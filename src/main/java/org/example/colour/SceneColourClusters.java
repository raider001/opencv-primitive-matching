package org.example.colour;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.List;
/**
 * Decomposes a BGR scene into colour-isolated binary masks.
 *
 * Chromatic pixels (sufficient saturation) are grouped by hue peak via inRange
 * windows. Achromatic pixels are split into two brightness bands:
 *   BRIGHT_ACHROMATIC — white, light grey (val >= BRIGHT_VAL_THRESHOLD)
 *   DARK_ACHROMATIC   — black, dark grey, near-black (val < BRIGHT_VAL_THRESHOLD)
 *
 * Keeping these separate means a black shape on a white background produces two
 * distinct masks whose boundaries give the shape contour, regardless of whether
 * the shape is dark-on-light or light-on-dark.
 */
public final class SceneColourClusters {
    public static final int    MAX_CLUSTERS        = 12;
    public static final double HUE_TOLERANCE       = 14.0;
    private static final int   PEAK_MIN_SEPARATION = 18;
    public static final int    MIN_CONTOUR_AREA     = 64;
    private static final double MIN_SAT          = 35.0;
    private static final double MIN_VAL          = 25.0;   // below = near-black
    private static final double BRIGHT_VAL_THRESHOLD = 100.0; // split achromatic here
    private static final int    MIN_PIXEL_COUNT  = 64;
    // -------------------------------------------------------------------------
    public static final class Cluster {
        public final Mat     mask;
        public final double  hue;
        public final boolean achromatic;
        /**
         * True if this is a BRIGHT achromatic cluster (white / light-grey, val >= threshold).
         * False for chromatic clusters or dark-achromatic clusters (black / dark-grey).
         * Used by the matcher to distinguish the foreground shape cluster from the
         * background cluster when both are achromatic.
         */
        public final boolean brightAchromatic;
        Cluster(Mat mask, double hue, boolean achromatic, boolean brightAchromatic) {
            this.mask             = mask;
            this.hue              = hue;
            this.achromatic       = achromatic;
            this.brightAchromatic = brightAchromatic;
        }
        public void release() { mask.release(); }
    }
    private SceneColourClusters() {}

    // =========================================================================
    // Border-pixel variant
    // =========================================================================

    /**
     * Identical to {@link #extract(Mat)} but restricts the chromatic hue-histogram
     * to <em>border pixels only</em> — i.e. the pixels that sit on the outline of
     * each shape rather than its filled interior.
     *
     * <p>This is achieved by computing a morphological gradient (dilate − erode)
     * on the full chromatic mask before building the hue histogram.  The resulting
     * set of peaks therefore reflects what colours appear <em>at the edges</em> of
     * shapes, which is more robust for:
     * <ul>
     *   <li>Filled shapes whose interior and outline share a colour (no change).</li>
     *   <li>Ring / outline shapes where the fill is background — avoids counting the
     *       background colour as a foreground cluster.</li>
     *   <li>Complex scenes where a large background fill would dominate a full-image
     *       histogram but is absent from the actual shape boundaries.</li>
     * </ul>
     *
     * <p>Achromatic cluster detection (bright / dark split) is unchanged — it still
     * operates on the full image because achromatic regions are identified by
     * brightness, not hue, and the border vs fill distinction is less meaningful
     * for greyscale shapes.
     */
    public static List<Cluster> extractFromBorderPixels(Mat bgrScene) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(bgrScene, hsv, Imgproc.COLOR_BGR2HSV);

        // Full chromatic mask (saturation + brightness thresholds)
        Mat chromaticMask = new Mat();
        Core.inRange(hsv,
                new Scalar(0,   MIN_SAT, MIN_VAL),
                new Scalar(179, 255,     255),
                chromaticMask);

        // Morphological gradient = dilation − erosion → border pixels only
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Mat borderMask = new Mat();
        Imgproc.morphologyEx(chromaticMask, borderMask, Imgproc.MORPH_GRADIENT, kernel);
        chromaticMask.release();

        // Zero the 1px image border so edge artefacts from the morphological
        // gradient don't create spurious hue peaks or frame-spanning contours.
        borderMask.row(0).setTo(Scalar.all(0));
        borderMask.row(borderMask.rows() - 1).setTo(Scalar.all(0));
        borderMask.col(0).setTo(Scalar.all(0));
        borderMask.col(borderMask.cols() - 1).setTo(Scalar.all(0));

        // Build hue histogram restricted to border pixels
        float[] hueHist = buildHueHistogram(hsv, borderMask);
        borderMask.release();

        List<Integer> peaks = findPeaks(hueHist, MIN_PIXEL_COUNT);
        List<Cluster> clusters = new ArrayList<>();

        // Chromatic clusters (full mask per hue — border only for discovery)
        for (int peakHue : peaks) {
            if (clusters.size() >= MAX_CLUSTERS) break;
            Mat clusterMask = hueRangeMask(hsv, peakHue, HUE_TOLERANCE);
            if (Core.countNonZero(clusterMask) < MIN_PIXEL_COUNT) {
                clusterMask.release();
                continue;
            }
            zeroImageBorder(clusterMask);
            clusters.add(new Cluster(clusterMask, peakHue, false, false));
        }

        // Achromatic clusters — apply morphological gradient so masks represent
        // the BORDER of bright/dark regions, not the full filled interior.
        // This makes achromatic contours edge-aligned, consistent with chromatic.
        Mat brightFull = new Mat();
        Core.inRange(hsv,
                new Scalar(0,   0,       BRIGHT_VAL_THRESHOLD),
                new Scalar(179, MIN_SAT, 255),
                brightFull);
        Mat darkFull = new Mat();
        Core.inRange(hsv,
                new Scalar(0,   0, 0),
                new Scalar(179, 255, BRIGHT_VAL_THRESHOLD),
                darkFull);
        Mat brightBorder = new Mat();
        Mat darkBorder   = new Mat();
        Imgproc.morphologyEx(brightFull, brightBorder, Imgproc.MORPH_GRADIENT, kernel);
        Imgproc.morphologyEx(darkFull,   darkBorder,   Imgproc.MORPH_GRADIENT, kernel);
        kernel.release();
        brightFull.release();
        darkFull.release();

        if (Core.countNonZero(brightBorder) >= MIN_PIXEL_COUNT) {
            zeroImageBorder(brightBorder);
            clusters.add(new Cluster(brightBorder, Double.NaN, true, true));   // bright border
        } else { brightBorder.release(); }

        if (Core.countNonZero(darkBorder) >= MIN_PIXEL_COUNT) {
            zeroImageBorder(darkBorder);
            clusters.add(new Cluster(darkBorder, Double.NaN, true, false));    // dark border
        } else { darkBorder.release(); }

        hsv.release();
        return clusters;
    }

    /** Zero the 1-pixel image border of a mask in-place to suppress edge artefacts. */
    private static void zeroImageBorder(Mat mask) {
        mask.row(0).setTo(Scalar.all(0));
        mask.row(mask.rows() - 1).setTo(Scalar.all(0));
        mask.col(0).setTo(Scalar.all(0));
        mask.col(mask.cols() - 1).setTo(Scalar.all(0));
    }

    // =========================================================================
    public static List<Cluster> extract(Mat bgrScene) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(bgrScene, hsv, Imgproc.COLOR_BGR2HSV);
        // Bright achromatic: white / light grey (high value, low saturation)
        Mat brightAchromatic = new Mat();
        Core.inRange(hsv,
                new Scalar(0,   0,       BRIGHT_VAL_THRESHOLD),
                new Scalar(179, MIN_SAT, 255),
                brightAchromatic);
        // Dark achromatic: near-black / dark grey (low value regardless of sat)
        Mat darkAchromatic = new Mat();
        Core.inRange(hsv,
                new Scalar(0,   0, 0),
                new Scalar(179, 255, BRIGHT_VAL_THRESHOLD),
                darkAchromatic);
        // Chromatic: sufficient saturation + brightness
        Mat chromaticMask = new Mat();
        Core.inRange(hsv,
                new Scalar(0,   MIN_SAT, MIN_VAL),
                new Scalar(179, 255,     255),
                chromaticMask);
        float[] hueHist = buildHueHistogram(hsv, chromaticMask);
        chromaticMask.release();
        List<Integer> peaks = findPeaks(hueHist, MIN_PIXEL_COUNT);
        List<Cluster> clusters = new ArrayList<>();
        // Chromatic clusters
        for (int peakHue : peaks) {
            if (clusters.size() >= MAX_CLUSTERS) break;
            Mat clusterMask = hueRangeMask(hsv, peakHue, HUE_TOLERANCE);
            if (Core.countNonZero(clusterMask) < MIN_PIXEL_COUNT) {
                clusterMask.release();
                continue;
            }
            zeroImageBorder(clusterMask);
            clusters.add(new Cluster(clusterMask, peakHue, false, false)); // chromatic
        }
        // Bright achromatic cluster
        if (Core.countNonZero(brightAchromatic) >= MIN_PIXEL_COUNT) {
            zeroImageBorder(brightAchromatic);
            clusters.add(new Cluster(brightAchromatic, Double.NaN, true, true));  // bright
        } else {
            brightAchromatic.release();
        }
        // Dark achromatic cluster
        if (Core.countNonZero(darkAchromatic) >= MIN_PIXEL_COUNT) {
            zeroImageBorder(darkAchromatic);
            clusters.add(new Cluster(darkAchromatic, Double.NaN, true, false));   // dark
        } else {
            darkAchromatic.release();
        }
        hsv.release();
        return clusters;
    }
    public static Mat applyMask(Mat bgrScene, Cluster cluster) {
        Mat result = Mat.zeros(bgrScene.size(), bgrScene.type());
        bgrScene.copyTo(result, cluster.mask);
        return result;
    }
    // =========================================================================
    private static float[] buildHueHistogram(Mat hsv, Mat chromaticMask) {
        List<Mat> hsvChannels = new ArrayList<>();
        Core.split(hsv, hsvChannels);
        Mat hueChannel = hsvChannels.get(0);
        Mat hist = new Mat();
        Imgproc.calcHist(List.of(hueChannel), new MatOfInt(0), chromaticMask,
                hist, new MatOfInt(180), new MatOfFloat(0f, 180f));
        float[] result = new float[180];
        for (int i = 0; i < 180; i++) result[i] = (float) hist.get(i, 0)[0];
        hist.release();
        hueChannel.release();
        for (int i = 1; i < hsvChannels.size(); i++) hsvChannels.get(i).release();
        return result;
    }
    private static List<Integer> findPeaks(float[] hist, int minCount) {
        float[] smoothed = new float[180];
        int smoothR = 2;
        for (int i = 0; i < 180; i++) {
            float sum = 0;
            for (int d = -smoothR; d <= smoothR; d++)
                sum += hist[(i + d + 180) % 180];
            smoothed[i] = sum / (2 * smoothR + 1);
        }
        List<int[]> peaks = new ArrayList<>();
        for (int i = 0; i < 180; i++) {
            float prev = smoothed[(i - 1 + 180) % 180];
            float curr = smoothed[i];
            float next = smoothed[(i + 1) % 180];
            if (curr > prev && curr >= next && curr >= minCount)
                peaks.add(new int[]{i, (int) curr});
        }
        peaks.sort((a, b) -> b[1] - a[1]);
        List<Integer> result = new ArrayList<>();
        boolean[] suppressed = new boolean[peaks.size()];
        for (int i = 0; i < peaks.size() && result.size() < MAX_CLUSTERS; i++) {
            if (suppressed[i]) continue;
            result.add(peaks.get(i)[0]);
            for (int j = i + 1; j < peaks.size(); j++) {
                int d = Math.abs(peaks.get(i)[0] - peaks.get(j)[0]);
                d = Math.min(d, 180 - d);
                if (d < PEAK_MIN_SEPARATION) suppressed[j] = true;
            }
        }
        return result;
    }
    private static Mat hueRangeMask(Mat hsv, int peakHue, double tolerance) {
        double lo = peakHue - tolerance;
        double hi = peakHue + tolerance;
        Mat mask = new Mat();
        if (lo < 0) {
            Mat m1 = new Mat(), m2 = new Mat();
            Core.inRange(hsv, new Scalar(0, MIN_SAT, MIN_VAL), new Scalar(hi, 255, 255), m1);
            Core.inRange(hsv, new Scalar(180 + lo, MIN_SAT, MIN_VAL), new Scalar(179, 255, 255), m2);
            Core.bitwise_or(m1, m2, mask); m1.release(); m2.release();
        } else if (hi > 179) {
            Mat m1 = new Mat(), m2 = new Mat();
            Core.inRange(hsv, new Scalar(lo, MIN_SAT, MIN_VAL), new Scalar(179, 255, 255), m1);
            Core.inRange(hsv, new Scalar(0, MIN_SAT, MIN_VAL), new Scalar(hi - 180, 255, 255), m2);
            Core.bitwise_or(m1, m2, mask); m1.release(); m2.release();
        } else {
            Core.inRange(hsv, new Scalar(lo, MIN_SAT, MIN_VAL), new Scalar(hi, 255, 255), mask);
        }
        return mask;
    }
}