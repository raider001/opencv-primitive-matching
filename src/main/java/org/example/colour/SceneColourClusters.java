package org.example.colour;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Decomposes a BGR scene into a small number of distinct colour clusters,
 * each represented as a binary mask of pixels belonging to that cluster.
 *
 * <h2>Purpose</h2>
 * <p>Rather than extracting contours from a flat greyscale edge map (where all
 * colours compete and noise edges merge with target shape edges), we isolate
 * each distinct colour into its own layer and extract contours independently.
 * This means a rectangle's contour only competes with other same-coloured edges —
 * background noise of different colours simply does not exist in that layer.
 *
 * <h2>How it works</h2>
 * <ol>
 *   <li>Convert BGR → HSV</li>
 *   <li>Build a hue histogram (180 bins, one per OpenCV half-degree)</li>
 *   <li>Find the top N_MAX hue peaks (ignoring achromatic/low-saturation pixels)</li>
 *   <li>For each peak: produce a binary mask via {@code Core.inRange} with
 *       {@link #HUE_TOLERANCE} tolerance around that peak hue</li>
 *   <li>Also produce one achromatic mask (low-saturation pixels — white/grey/black)</li>
 * </ol>
 *
 * <h2>Cost</h2>
 * <p>One BGR→HSV conversion + one histogram + N_MAX inRange calls.
 * Typically completes in &lt;5 ms on a 640×480 scene.
 * N_MAX is capped at {@link #MAX_CLUSTERS} to bound the total cost.
 *
 * <h2>Colour agnostic matching</h2>
 * <p>The clusters are derived from the <em>scene</em>, not the reference.
 * This means a red-triangle reference will match a blue triangle in the scene
 * because the blue cluster's contour geometry is compared against the reference
 * geometry independently of colour.
 */
public final class SceneColourClusters {

    /** Maximum number of chromatic clusters extracted from the scene. */
    public static final int MAX_CLUSTERS = 12;

    /**
     * Hue window half-width for mask generation (OpenCV half-degrees, 10 = ±10°).
     * Tighter than before so adjacent hues (e.g. yellow vs green) are not merged.
     */
    public static final double HUE_TOLERANCE = 10.0;

    /**
     * Minimum hue separation between two peaks to be treated as distinct colours.
     * Must be > HUE_TOLERANCE to avoid gaps between adjacent cluster masks.
     */
    private static final int   PEAK_MIN_SEPARATION = 12;

    /** Minimum saturation to be considered chromatic (0–255). */
    private static final double MIN_SAT = 35.0;

    /** Minimum value to be considered non-black (0–255). */
    private static final double MIN_VAL = 25.0;

    /** Minimum number of chromatic pixels a peak must have to be kept. */
    private static final int MIN_PIXEL_COUNT = 64;

    // -------------------------------------------------------------------------

    /** One isolated colour layer from the scene. */
    public static final class Cluster {
        /** Binary mask: 255 = pixels belonging to this cluster. */
        public final Mat    mask;
        /** Centre hue of this cluster (OpenCV half-degrees, 0–179). NaN for achromatic. */
        public final double hue;
        /** True if this is the achromatic (white/grey/black) cluster. */
        public final boolean achromatic;

        Cluster(Mat mask, double hue, boolean achromatic) {
            this.mask       = mask;
            this.hue        = hue;
            this.achromatic = achromatic;
        }

        /** Releases the underlying OpenCV Mat. */
        public void release() { mask.release(); }
    }

    private SceneColourClusters() {}

    // =========================================================================
    // Public API
    // =========================================================================

    /**
     * Extracts up to {@link #MAX_CLUSTERS} colour clusters from {@code bgrScene}.
     *
     * <p>The returned list always includes one achromatic cluster (last element)
     * plus up to {@link #MAX_CLUSTERS} chromatic clusters.
     * Caller is responsible for releasing each {@link Cluster#mask}.
     *
     * @param bgrScene  source scene (CV_8UC3 BGR)
     * @return list of colour clusters, ordered by pixel coverage (largest first)
     */
    public static List<Cluster> extract(Mat bgrScene) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(bgrScene, hsv, Imgproc.COLOR_BGR2HSV);

        // ── Achromatic mask (low saturation or near-black) ─────────────────
        Mat achromaticMask = new Mat();
        Core.inRange(hsv,
                new Scalar(0,   0,       MIN_VAL),
                new Scalar(179, MIN_SAT, 255),
                achromaticMask);

        // ── Chromatic mask (sufficient saturation + brightness) ────────────
        Mat chromaticMask = new Mat();
        Core.inRange(hsv,
                new Scalar(0,   MIN_SAT, MIN_VAL),
                new Scalar(179, 255,     255),
                chromaticMask);

        // ── Hue histogram over chromatic pixels only ──────────────────────
        float[] hueHist = buildHueHistogram(hsv, chromaticMask);
        chromaticMask.release();

        // ── Find hue peaks ─────────────────────────────────────────────────
        List<Integer> peaks = findPeaks(hueHist, MIN_PIXEL_COUNT);

        // ── Build cluster masks ────────────────────────────────────────────
        List<Cluster> clusters = new ArrayList<>();
        for (int peakHue : peaks) {
            if (clusters.size() >= MAX_CLUSTERS) break;
            Mat clusterMask = hueRangeMask(hsv, peakHue, HUE_TOLERANCE);
            int coverage = Core.countNonZero(clusterMask);
            if (coverage < MIN_PIXEL_COUNT) {
                clusterMask.release();
                continue;
            }
            clusters.add(new Cluster(clusterMask, peakHue, false));
        }

        // Always add achromatic cluster if it has coverage
        if (Core.countNonZero(achromaticMask) >= MIN_PIXEL_COUNT) {
            clusters.add(new Cluster(achromaticMask, Double.NaN, true));
        } else {
            achromaticMask.release();
        }

        hsv.release();
        return clusters;
    }

    /**
     * Applies a cluster mask to a BGR scene, returning a masked BGR image
     * (non-cluster pixels set to black).  The returned Mat must be released
     * by the caller.
     */
    public static Mat applyMask(Mat bgrScene, Cluster cluster) {
        Mat result = Mat.zeros(bgrScene.size(), bgrScene.type());
        bgrScene.copyTo(result, cluster.mask);
        return result;
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    /** Builds a 180-bin hue histogram over chromatic pixels only using OpenCV calcHist. */
    private static float[] buildHueHistogram(Mat hsv, Mat chromaticMask) {
        // Extract hue channel only
        List<Mat> hsvChannels = new ArrayList<>();
        Core.split(hsv, hsvChannels);
        Mat hueChannel = hsvChannels.get(0);

        // calcHist: 180 bins, range [0, 180), masked to chromatic pixels only
        Mat hist = new Mat();
        Imgproc.calcHist(
                List.of(hueChannel),
                new MatOfInt(0),
                chromaticMask,
                hist,
                new MatOfInt(180),
                new MatOfFloat(0f, 180f)
        );

        // Copy into float array
        float[] result = new float[180];
        for (int i = 0; i < 180; i++) {
            result[i] = (float) hist.get(i, 0)[0];
        }

        hist.release();
        hueChannel.release();
        for (int i = 1; i < hsvChannels.size(); i++) hsvChannels.get(i).release();

        return result;
    }

    /**
     * Finds hue peaks in the histogram.
     * Peaks closer than {@link #PEAK_MIN_SEPARATION} to a stronger peak are suppressed.
     * Returns peak hues sorted by pixel count (highest first).
     */
    private static List<Integer> findPeaks(float[] hist, int minCount) {
        // Smooth with a small window to merge adjacent single-bin spikes
        float[] smoothed = new float[180];
        int smoothR = 2;
        for (int i = 0; i < 180; i++) {
            float sum = 0;
            for (int d = -smoothR; d <= smoothR; d++)
                sum += hist[(i + d + 180) % 180];
            smoothed[i] = sum / (2 * smoothR + 1);
        }

        // Find local maxima above minCount
        List<int[]> peaks = new ArrayList<>();
        for (int i = 0; i < 180; i++) {
            float prev = smoothed[(i - 1 + 180) % 180];
            float curr = smoothed[i];
            float next = smoothed[(i + 1) % 180];
            if (curr > prev && curr >= next && curr >= minCount)
                peaks.add(new int[]{i, (int) curr});
        }

        // Sort by pixel count descending
        peaks.sort((a, b) -> b[1] - a[1]);

        // Suppress peaks within PEAK_MIN_SEPARATION of a stronger peak
        List<Integer> result = new ArrayList<>();
        boolean[] suppressed = new boolean[peaks.size()];
        for (int i = 0; i < peaks.size() && result.size() < MAX_CLUSTERS; i++) {
            if (suppressed[i]) continue;
            result.add(peaks.get(i)[0]);
            for (int j = i + 1; j < peaks.size(); j++) {
                int hueDiff = Math.abs(peaks.get(i)[0] - peaks.get(j)[0]);
                hueDiff = Math.min(hueDiff, 180 - hueDiff);
                if (hueDiff < PEAK_MIN_SEPARATION) suppressed[j] = true;
            }
        }
        return result;
    }

    /** Builds a binary mask for pixels within HUE_TOLERANCE of peakHue, handling wrap. */
    private static Mat hueRangeMask(Mat hsv, int peakHue, double tolerance) {
        double lo = peakHue - tolerance;
        double hi = peakHue + tolerance;

        Mat mask = new Mat();
        if (lo < 0) {
            // Wraps below 0
            Mat m1 = new Mat(), m2 = new Mat();
            Core.inRange(hsv, new Scalar(0,       MIN_SAT, MIN_VAL),
                              new Scalar(hi,       255,     255), m1);
            Core.inRange(hsv, new Scalar(180 + lo, MIN_SAT, MIN_VAL),
                              new Scalar(179,       255,     255), m2);
            Core.bitwise_or(m1, m2, mask);
            m1.release(); m2.release();
        } else if (hi > 179) {
            // Wraps above 179
            Mat m1 = new Mat(), m2 = new Mat();
            Core.inRange(hsv, new Scalar(lo,       MIN_SAT, MIN_VAL),
                              new Scalar(179,       255,     255), m1);
            Core.inRange(hsv, new Scalar(0,         MIN_SAT, MIN_VAL),
                              new Scalar(hi - 180,  255,     255), m2);
            Core.bitwise_or(m1, m2, mask);
            m1.release(); m2.release();
        } else {
            Core.inRange(hsv, new Scalar(lo, MIN_SAT, MIN_VAL),
                              new Scalar(hi, 255,     255), mask);
        }
        return mask;
    }
}

