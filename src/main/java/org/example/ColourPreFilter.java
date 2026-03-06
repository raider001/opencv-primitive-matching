package org.example;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

/**
 * Colour isolation pre-processing layer.
 *
 * <p>Every matching technique is run in three modes:
 * <ol>
 *   <li><b>Base</b> — full-colour scene and reference, no pre-processing.</li>
 *   <li><b>CF_LOOSE</b> — both images passed through {@link #apply} at {@link #LOOSE} tolerance
 *       (±15° hue) before matching.</li>
 *   <li><b>CF_TIGHT</b> — both images passed through {@link #apply} at {@link #TIGHT} tolerance
 *       (±8° hue) before matching.</li>
 * </ol>
 *
 * <p>The binary mask produced by {@link #apply} has the same width and height as the input;
 * white pixels (255) fall within the colour range and black pixels (0) fall outside.
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
     * Applies the colour pre-filter to a <em>scene</em> image, deriving the colour range
     * from the given reference ID.
     *
     * <p>This is the primary entry point for matcher {@code _CF} variants:
     * <pre>{@code
     *   Mat sceneMask = ColourPreFilter.applyToScene(sceneMat, refId, ColourPreFilter.LOOSE);
     * }</pre>
     *
     * @param bgrScene     the full-colour 640×480 scene
     * @param referenceId  which reference's foreground colour to isolate
     * @param hueTolerance use {@link #LOOSE} or {@link #TIGHT}
     * @return binary mask aligned to the scene (same size)
     */
    public static Mat applyToScene(Mat bgrScene, ReferenceId referenceId, double hueTolerance) {
        ColourRange range = extractReferenceColourRange(referenceId, hueTolerance);
        return apply(bgrScene, range);
    }

    /**
     * Applies the colour pre-filter to a <em>reference</em> image using its own foreground
     * colour range — useful for masking the reference before passing it to a matcher.
     *
     * @param bgrReference the 128×128 reference Mat
     * @param referenceId  the ID (used to look up the foreground colour)
     * @param hueTolerance use {@link #LOOSE} or {@link #TIGHT}
     * @return binary mask aligned to the reference (same size)
     */
    public static Mat applyToReference(Mat bgrReference, ReferenceId referenceId,
                                       double hueTolerance) {
        ColourRange range = extractReferenceColourRange(referenceId, hueTolerance);
        return apply(bgrReference, range);
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
     * Counts white pixels (value == 255) in a single-channel binary mask and returns the
     * fraction as 0.0–1.0.
     */
    public static double whitePixelFraction(Mat mask) {
        if (mask == null || mask.empty()) return 0.0;
        return Core.countNonZero(mask) / (double)(mask.rows() * mask.cols());
    }
}

