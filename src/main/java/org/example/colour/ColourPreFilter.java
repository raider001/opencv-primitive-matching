package org.example.colour;

import org.example.factories.ReferenceId;
import org.example.factories.ReferenceImageFactory;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

/**
 * Colour isolation pre-processing layer.
 *
 * <p>Every matching technique is run in three modes:
 * <ol>
 *   <li><b>Base</b> — full-colour scene and reference, no pre-processing.</li>
 *   <li><b>CF_LOOSE</b> — scene and reference passed through
 *       {@link #applyMaskedBgrToScene} / {@link #applyMaskedBgrToReference} at
 *       {@link #LOOSE} tolerance (±15° hue).  Pixels outside the foreground colour
 *       range are zeroed; the result is still a 3-channel BGR image suitable for
 *       template matching.</li>
 *   <li><b>CF_TIGHT</b> — same but at {@link #TIGHT} tolerance (±8° hue).</li>
 * </ol>
 *
 * <p>The low-level {@link #apply} / {@link #applyToScene} / {@link #applyToReference}
 * methods return a <b>binary mask</b> (CV_8UC1) used by {@link ColourFirstLocator} for
 * blob detection.  The higher-level {@link #applyMaskedBgrToScene} /
 * {@link #applyMaskedBgrToReference} apply that mask and return a masked BGR image
 * (CV_8UC3) with non-matching pixels set to black — the correct input for matchers.
 *
 * <p><b>Red/orange hue wrap-around:</b> OpenCV encodes hue as 0–179 (half-degrees).
 * Red sits near H=0 and wraps at H=179. When the computed hue window crosses this boundary,
 * {@link #apply} automatically performs two {@code Core.inRange} calls and OR-combines them.
 * This is handled transparently via {@link ColourRange#wrapsHue()}.
 */
public final class ColourPreFilter {

    /** Loose hue tolerance — ±15° (OpenCV half-degrees: ±7.5 units). */
    public static final double LOOSE = 15.0;

    /** Tight hue tolerance — ±8° (OpenCV half-degrees: ±4 units). */
    public static final double TIGHT = 8.0;

    // Saturation and Value tolerance applied symmetrically around the reference colour
    private static final double SAT_TOLERANCE = 40.0;
    private static final double VAL_TOLERANCE = 40.0;

    private ColourPreFilter() {}

    // =========================================================================
    // Core apply methods
    // =========================================================================

    /**
     * Applies a colour isolation mask to {@code bgrImage}.
     *
     * <p>Converts BGR → HSV then calls {@code Core.inRange} with the supplied range.
     * Handles hue wrap-around automatically via {@link ColourRange#wrapsHue()}.
     *
     * @param bgrImage  input image (CV_8UC3 BGR)
     * @param range     HSV range to isolate — see {@link #extractReferenceColourRange}
     * @return binary mask (CV_8UC1): 255 = within range, 0 = outside
     */
    public static Mat apply(Mat bgrImage, ColourRange range) {
        Mat hsv  = new Mat();
        Mat mask = new Mat();
        Imgproc.cvtColor(bgrImage, hsv, Imgproc.COLOR_BGR2HSV);

        if (range.wrapsHue()) {
            // Split the window into two non-wrapping ranges and OR-combine
            // Lower segment: [0, hsvUpper]
            // Upper segment: [hsvLower, 179]
            Mat maskLow  = new Mat();
            Mat maskHigh = new Mat();
            Core.inRange(hsv,
                    new Scalar(0,                 range.hsvLower().val[1], range.hsvLower().val[2]),
                    new Scalar(range.hsvUpper().val[0], range.hsvUpper().val[1], range.hsvUpper().val[2]),
                    maskLow);
            Core.inRange(hsv,
                    new Scalar(range.hsvLower().val[0], range.hsvLower().val[1], range.hsvLower().val[2]),
                    new Scalar(179,                range.hsvUpper().val[1], range.hsvUpper().val[2]),
                    maskHigh);
            Core.bitwise_or(maskLow, maskHigh, mask);
            maskLow.release();
            maskHigh.release();
        } else {
            Core.inRange(hsv, range.hsvLower(), range.hsvUpper(), mask);
        }

        hsv.release();
        return mask;
    }

    /**
     * Convenience overload — builds the range from {@code hsvLower}/{@code hsvUpper} scalars
     * directly (no {@link ColourRange} needed by the caller).
     */
    public static Mat apply(Mat bgrImage, Scalar hsvLower, Scalar hsvUpper) {
        String label = String.format("manual_H%.0f-%.0f", hsvLower.val[0], hsvUpper.val[0]);
        return apply(bgrImage, new ColourRange(hsvLower, hsvUpper, label));
    }

    /**
     * Returns a binary mask (CV_8UC1) for the scene: 255 where the pixel colour
     * matches the reference foreground, 0 elsewhere.
     * Used by {@link ColourFirstLocator} for blob detection / region proposal.
     *
     * @param bgrScene     the full-colour scene (any size, CV_8UC3 BGR)
     * @param referenceId  which reference's foreground colour to isolate
     * @param hueTolerance use {@link #LOOSE} or {@link #TIGHT}
     * @return CV_8UC1 binary mask — caller must release
     */
    public static Mat applyToScene(Mat bgrScene, ReferenceId referenceId, double hueTolerance) {
        ColourRange range = extractReferenceColourRange(referenceId, hueTolerance);
        return apply(bgrScene, range);
    }

    /**
     * Returns a binary mask (CV_8UC1) for the reference: 255 where the pixel colour
     * matches the reference foreground, 0 elsewhere.
     *
     * @param bgrReference the 128×128 reference Mat
     * @param referenceId  the ID (used to look up the foreground colour)
     * @param hueTolerance use {@link #LOOSE} or {@link #TIGHT}
     * @return CV_8UC1 binary mask — caller must release
     */
    public static Mat applyToReference(Mat bgrReference, ReferenceId referenceId,
                                       double hueTolerance) {
        ColourRange range = extractReferenceColourRange(referenceId, hueTolerance);
        return apply(bgrReference, range);
    }

    /**
     * Returns a masked <b>BGR</b> image (CV_8UC3) for use as a template-matching search
     * image: pixels whose colour matches the reference foreground are kept; all other pixels
     * are set to black (zero).
     *
     * <p>This is the correct input for template matchers — passing a binary mask to
     * {@code Imgproc.matchTemplate} produces incorrect single-channel comparisons.
     *
     * @param bgrScene     the full-colour scene (any size, CV_8UC3 BGR)
     * @param referenceId  which reference's foreground colour to isolate
     * @param hueTolerance use {@link #LOOSE} or {@link #TIGHT}
     * @return new CV_8UC3 BGR Mat with non-matching pixels zeroed — caller must release
     */
    public static Mat applyMaskedBgrToScene(Mat bgrScene, ReferenceId referenceId,
                                            double hueTolerance) {
        Mat mask   = applyToScene(bgrScene, referenceId, hueTolerance);
        Mat masked = Mat.zeros(bgrScene.size(), bgrScene.type());
        bgrScene.copyTo(masked, mask);
        mask.release();
        return masked;
    }

    /**
     * Returns a masked <b>BGR</b> image (CV_8UC3) for use as a template: pixels whose colour
     * matches the reference foreground are kept; all other pixels are set to black.
     *
     * @param bgrReference the 128×128 reference Mat (CV_8UC3 BGR)
     * @param referenceId  the ID (used to look up the foreground colour)
     * @param hueTolerance use {@link #LOOSE} or {@link #TIGHT}
     * @return new CV_8UC3 BGR Mat with non-matching pixels zeroed — caller must release
     */
    public static Mat applyMaskedBgrToReference(Mat bgrReference, ReferenceId referenceId,
                                                double hueTolerance) {
        Mat mask   = applyToReference(bgrReference, referenceId, hueTolerance);
        Mat masked = Mat.zeros(bgrReference.size(), bgrReference.type());
        bgrReference.copyTo(masked, mask);
        mask.release();
        return masked;
    }

    // =========================================================================
    // Colour range extraction
    // =========================================================================

    /**
     * Builds a {@link ColourRange} for the given reference's foreground colour.
     *
     * <p>The foreground colour is obtained from {@link ReferenceImageFactory#foregroundColour}
     * (no image loading required). The colour is converted to HSV and a symmetric tolerance
     * window is applied:
     * <ul>
     *   <li>Hue: ±{@code hueTolerance / 2} in OpenCV half-degree units (÷2 because OpenCV
     *       H is 0–179, representing 0–360°)</li>
     *   <li>Saturation: ±{@link #SAT_TOLERANCE} clamped to [0, 255]</li>
     *   <li>Value: ±{@link #VAL_TOLERANCE} clamped to [0, 255]</li>
     * </ul>
     *
     * <p><b>White/near-grey references</b> (slot 0 in the 8-colour cycle) have very low
     * saturation, so the colour range will be wide but the mask will still work — just with
     * lower discrimination power.
     *
     * <p><b>Red/orange hue wrap-around</b> is handled by returning a range where
     * {@code hsvLower.val[0] > hsvUpper.val[0]}; the caller (or {@link #apply}) detects this
     * via {@link ColourRange#wrapsHue()} and performs two inRange passes.
     *
     * @param id           the reference shape
     * @param hueTolerance full window width in degrees (÷2 per side); use {@link #LOOSE} or
     *                     {@link #TIGHT}
     * @return a {@link ColourRange} ready to pass to {@link #apply}
     */
    public static ColourRange extractReferenceColourRange(ReferenceId id, double hueTolerance) {
        Scalar bgrColour = ReferenceImageFactory.foregroundColour(id);
        String colourName = ReferenceImageFactory.foregroundColourName(id);
        String label = colourName + (hueTolerance == LOOSE ? "_LOOSE" : "_TIGHT");

        // Convert the single BGR pixel to HSV
        Mat bgr1x1 = new Mat(1, 1, CvType.CV_8UC3);
        bgr1x1.put(0, 0, bgrColour.val[0], bgrColour.val[1], bgrColour.val[2]);
        Mat hsv1x1 = new Mat();
        Imgproc.cvtColor(bgr1x1, hsv1x1, Imgproc.COLOR_BGR2HSV);
        double[] hsv = hsv1x1.get(0, 0);
        bgr1x1.release();
        hsv1x1.release();

        double h = hsv[0]; // 0–179
        double s = hsv[1]; // 0–255
        double v = hsv[2]; // 0–255

        // OpenCV hue is half-degree — tolerance given in full degrees, so halve it
        double hTol = hueTolerance / 2.0;

        double hLow  = h - hTol;
        double hHigh = h + hTol;

        // Clamp S and V
        double sLow  = Math.max(0,   s - SAT_TOLERANCE);
        double sHigh = Math.min(255, s + SAT_TOLERANCE);
        double vLow  = Math.max(0,   v - VAL_TOLERANCE);
        double vHigh = Math.min(255, v + VAL_TOLERANCE);

        // Handle hue wrap-around
        if (hLow < 0) {
            // Wraps below 0 → lower = hLow+180, upper = hHigh (positive)
            return new ColourRange(
                    new Scalar(hLow + 180, sLow, vLow),
                    new Scalar(hHigh,      sHigh, vHigh),
                    label);
        } else if (hHigh > 179) {
            // Wraps above 179 → lower = hLow, upper = hHigh-180
            return new ColourRange(
                    new Scalar(hLow,       sLow, vLow),
                    new Scalar(hHigh - 180, sHigh, vHigh),
                    label);
        } else {
            return new ColourRange(
                    new Scalar(hLow,  sLow, vLow),
                    new Scalar(hHigh, sHigh, vHigh),
                    label);
        }
    }

    // =========================================================================
    // Utility
    // =========================================================================

    /**
     * Builds a {@link ColourRange} from an arbitrary BGR {@link Scalar} and hue tolerance.
     *
     * <p>Generalised form of {@link #extractReferenceColourRange} for callers
     * that need to threshold on an explicit colour rather than a registered reference ID.
     *
     * @param bgrColour    the colour to threshold, as a BGR {@link Scalar}
     * @param hueTolerance full window width in degrees; use {@link #LOOSE} or {@link #TIGHT}
     * @return a {@link ColourRange} ready to pass to {@link #apply}
     */
    public static ColourRange extractColourRange(Scalar bgrColour, double hueTolerance) {
        String label = String.format("bgr(%.0f,%.0f,%.0f)_H+/-%.0f",
                bgrColour.val[0], bgrColour.val[1], bgrColour.val[2], hueTolerance);

        Mat bgr1x1 = new Mat(1, 1, CvType.CV_8UC3);
        bgr1x1.put(0, 0, bgrColour.val[0], bgrColour.val[1], bgrColour.val[2]);
        Mat hsv1x1 = new Mat();
        Imgproc.cvtColor(bgr1x1, hsv1x1, Imgproc.COLOR_BGR2HSV);
        double[] hsv = hsv1x1.get(0, 0);
        bgr1x1.release(); hsv1x1.release();

        double h = hsv[0], s = hsv[1], v = hsv[2];
        double hTol  = hueTolerance / 2.0;
        double sLow  = Math.max(0,   s - SAT_TOLERANCE);
        double sHigh = Math.min(255, s + SAT_TOLERANCE);
        double vLow  = Math.max(0,   v - VAL_TOLERANCE);
        double vHigh = Math.min(255, v + VAL_TOLERANCE);
        double hLow  = h - hTol;
        double hHigh = h + hTol;

        if (hLow < 0) {
            return new ColourRange(
                    new Scalar(hLow + 180, sLow, vLow),
                    new Scalar(hHigh,      sHigh, vHigh), label);
        } else if (hHigh > 179) {
            return new ColourRange(
                    new Scalar(hLow,        sLow, vLow),
                    new Scalar(hHigh - 180, sHigh, vHigh), label);
        } else {
            return new ColourRange(
                    new Scalar(hLow,  sLow, vLow),
                    new Scalar(hHigh, sHigh, vHigh), label);
        }
    }


    /**
     * Counts white pixels (value == 255) in a single-channel binary mask and returns the
     * fraction as 0.0–1.0.
     */
    public static double whitePixelFraction(Mat mask) {
        if (mask == null || mask.empty()) return 0.0;
        return Core.countNonZero(mask) / (double)(mask.rows() * mask.cols());
    }
}

