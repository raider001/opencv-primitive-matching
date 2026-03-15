package org.example.colour;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Experimental alternative to {@link SceneColourClusters}.
 *
 * <h2>Algorithm differences from production</h2>
 * <ul>
 *   <li><b>No cluster cap</b> — production limits output to {@code MAX_CLUSTERS=12};
 *       this implementation returns as many clusters as actually exist.</li>
 *   <li><b>Spatial separation</b> — within each hue/brightness band, any two
 *       regions that are not connected are returned as <em>separate</em> clusters.
 *       Production merges them all into one mask per peak.</li>
 *   <li><b>Auto-heal</b> — before connectivity analysis, a morphological closing
 *       (radius {@link #HEAL_RADIUS} px) bridges small disconnects so minor
 *       anti-aliasing gaps or thin separations do not cause spurious splits.
 *       The returned cluster masks contain only <em>original</em> pixels
 *       (the healed pixels are used purely for grouping decisions).</li>
 * </ul>
 *
 * <h2>Cluster identity</h2>
 * <p>A cluster is uniquely defined by the combination of
 * <em>(hue band, brightness category, spatial region)</em>.  Achromatic regions
 * are split into BRIGHT and DARK, exactly as in production.
 *
 * <h2>How to activate</h2>
 * <p>Replace any {@code SceneColourClusters.INSTANCE} reference with
 * {@code ExperimentalSceneColourClusters.INSTANCE}.  Both implement
 * {@link SceneColourExtractor}.
 *
 * @see SceneColourClusters
 * @see SceneColourExtractor
 */
public final class ExperimentalSceneColourClusters implements SceneColourExtractor {

    /** Singleton — use wherever a {@link SceneColourExtractor} is required. */
    public static final ExperimentalSceneColourClusters INSTANCE =
            new ExperimentalSceneColourClusters();

    // =========================================================================
    // Tunable constants
    // =========================================================================

    /**
     * Half-radius (px) of the morphological closing kernel applied before
     * connected-component analysis.  Closing bridges gaps up to
     * {@code 2 × HEAL_RADIUS} pixels wide.  Increase to merge regions
     * separated by wider gaps; decrease for finer spatial separation.
     */
    public static final int HEAL_RADIUS = 8;

    /**
     * Maximum half-width (OpenCV hue units, 0–179) a cluster's hue range may
     * extend on either side of its peak.  Prevents a cluster from claiming the
     * "long arc" around the hue wheel when valley detection runs between two
     * isolated peaks (e.g. only red+orange detected → red's far boundary would
     * otherwise sweep through blue, green and purple).
     * <p>1 OpenCV unit ≈ 2°, so the default 22 ≈ ±44°.
     */
    public static final int MAX_HUE_HALF_WIDTH = 22;

    // Constants shared with / kept in sync with SceneColourClusters
    private static final double HUE_TOLERANCE       = SceneColourClusters.HUE_TOLERANCE;
    private static final int    PEAK_MIN_SEPARATION  = 18;
    private static final double MIN_SAT              = 35.0;
    private static final double MIN_VAL              = 25.0;
    private static final double BRIGHT_VAL_THRESHOLD = 100.0;
    private static final int    MIN_PIXEL_COUNT      = 64;

    private ExperimentalSceneColourClusters() {}

    // =========================================================================
    // SceneColourExtractor — primary entry points
    // =========================================================================

    /**
     * Border-pixel histogram variant.
     *
     * <p>Hue peaks are detected from the morphological-gradient border pixels
     * (consistent with {@link SceneColourClusters#extractFromBorderPixels}).
     * Full-pixel masks are built for each peak, then auto-healed and split
     * spatially by connected components.
     */
    @Override
    public List<ColourCluster> extractFromBorderPixels(Mat bgrScene) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(bgrScene, hsv, Imgproc.COLOR_BGR2HSV);

        // Build border-pixel mask for histogram (same approach as production)
        Mat chromaticRaw = buildChromaticMask(hsv);
        Mat border3 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Mat borderMask = new Mat();
        Imgproc.morphologyEx(chromaticRaw, borderMask, Imgproc.MORPH_GRADIENT, border3);
        border3.release();
        chromaticRaw.release();
        zeroImageBorder(borderMask);

        float[]       hueHist     = buildHueHistogram(hsv, borderMask);
        borderMask.release();
        float[]       smoothed    = computeSmoothedHist(hueHist, 1);
        List<Integer> peaks       = findPeaks(smoothed, MIN_PIXEL_COUNT);   // no cap
        int[][]       valleyBounds = computeValleyBounds(smoothed, peaks);

        Mat healKernel = buildHealKernel();
        List<ColourCluster> clusters = new ArrayList<>();

        // Chromatic bands → spatially split
        // Valley bounds prevent overlap with actual neighbours; MAX_HUE_HALF_WIDTH
        // prevents a cluster from claiming hues on the far side of the wheel when
        // only a few peaks were detected (e.g. red+orange leaving a huge gap).
        for (int pi = 0; pi < peaks.size(); pi++) {
            int peakHue = peaks.get(pi);
            int[] vb    = clampBounds(valleyBounds[pi][0], valleyBounds[pi][1], peakHue);
            int lo = vb[0], hi = vb[1];
            Mat rawMask = hueRangeMaskByBounds(hsv, lo, hi);
            if (Core.countNonZero(rawMask) < MIN_PIXEL_COUNT) { rawMask.release(); continue; }
            zeroImageBorder(rawMask);
            clusters.addAll(splitSpatially(rawMask, peakHue, false, false, lo, hi, healKernel));
            rawMask.release();
        }

        // Achromatic bands → spatially split
        Mat brightRaw = buildAchromaticMask(hsv, true);
        zeroImageBorder(brightRaw);
        clusters.addAll(splitSpatially(brightRaw, Double.NaN, true, true, 0, 179, healKernel));
        brightRaw.release();

        Mat darkRaw = buildAchromaticMask(hsv, false);
        zeroImageBorder(darkRaw);
        clusters.addAll(splitSpatially(darkRaw, Double.NaN, true, false, 0, 179, healKernel));
        darkRaw.release();

        healKernel.release();
        hsv.release();
        return clusters;
    }

    /**
     * Full-pixel histogram variant.
     *
     * <p>Uses the entire chromatic pixel set for peak detection (valley-based
     * bounds), then applies the same auto-heal + connected-component split.
     */
    @Override
    public List<ColourCluster> extract(Mat bgrScene) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(bgrScene, hsv, Imgproc.COLOR_BGR2HSV);

        Mat chromaticRaw = buildChromaticMask(hsv);
        float[]       hueHist     = buildHueHistogram(hsv, chromaticRaw);
        chromaticRaw.release();
        float[]       smoothed    = computeSmoothedHist(hueHist, 1);
        List<Integer> peaks       = findPeaks(smoothed, MIN_PIXEL_COUNT);   // no cap
        int[][]       valleyBounds = computeValleyBounds(smoothed, peaks);

        Mat healKernel = buildHealKernel();
        List<ColourCluster> clusters = new ArrayList<>();

        for (int pi = 0; pi < peaks.size(); pi++) {
            int peakHue = peaks.get(pi);
            int[] vb    = clampBounds(valleyBounds[pi][0], valleyBounds[pi][1], peakHue);
            int lo = vb[0], hi = vb[1];
            Mat rawMask = hueRangeMaskByBounds(hsv, lo, hi);
            if (Core.countNonZero(rawMask) < MIN_PIXEL_COUNT) { rawMask.release(); continue; }
            zeroImageBorder(rawMask);
            clusters.addAll(splitSpatially(rawMask, peakHue, false, false, lo, hi, healKernel));
            rawMask.release();
        }

        Mat brightRaw = buildAchromaticMask(hsv, true);
        zeroImageBorder(brightRaw);
        clusters.addAll(splitSpatially(brightRaw, Double.NaN, true, true, 0, 179, healKernel));
        brightRaw.release();

        Mat darkRaw = buildAchromaticMask(hsv, false);
        zeroImageBorder(darkRaw);
        clusters.addAll(splitSpatially(darkRaw, Double.NaN, true, false, 0, 179, healKernel));
        darkRaw.release();

        healKernel.release();
        hsv.release();
        return clusters;
    }

    // =========================================================================
    // Spatial splitting — the core difference from production
    // =========================================================================

    /**
     * Applies auto-heal (morphological closing) to {@code rawMask}, then
     * finds connected components in the healed image.  Each component with
     * sufficient area is returned as a separate {@link ColourCluster}.
     *
     * <p>The mask stored in each returned cluster contains only the
     * <em>original</em> pixels (intersection of {@code rawMask} with the
     * healed component region).  The healed pixels are used solely to decide
     * which original pixels belong together.
     *
     * <p>Up to 254 distinct spatial components are supported per colour band
     * (safe for all practical scene sizes).
     *
     * @param rawMask    CV_8UC1 binary mask — 255 = pixel in this colour band
     * @param hue        peak hue (NaN for achromatic)
     * @param achromatic true for bright/dark achromatic bands
     * @param bright     true for BRIGHT achromatic; ignored when !achromatic
     * @param lo         valley lower hue bound (0 for achromatic)
     * @param hi         valley upper hue bound (179 for achromatic)
     * @param healKernel pre-built ellipse closing kernel
     */
    private static List<ColourCluster> splitSpatially(
            Mat rawMask, double hue, boolean achromatic, boolean bright,
            int lo, int hi, Mat healKernel) {

        // Step 1 — auto-heal: close small gaps so thin breaks don't split a region
        Mat healed = new Mat();
        Imgproc.morphologyEx(rawMask, healed, Imgproc.MORPH_CLOSE, healKernel);

        // Step 2 — connected-component labelling on the healed mask
        Mat labels32 = new Mat();
        int numComponents = Imgproc.connectedComponents(healed, labels32); // label 0 = background
        healed.release();

        // Convert to 8-bit labels (practical limit: 254 distinct regions per band)
        Mat labels = new Mat();
        labels32.convertTo(labels, CvType.CV_8U);
        labels32.release();

        List<ColourCluster> result = new ArrayList<>();
        int maxComp = Math.min(numComponents, 255); // guard against unlikely overflow

        for (int comp = 1; comp < maxComp; comp++) {
            // Pixels belonging to this component in the healed image
            Mat compRegion = new Mat();
            Core.inRange(labels, new Scalar(comp), new Scalar(comp), compRegion);

            // Intersect with original pixels — only real (non-healed) pixels in the cluster
            Mat compMask = new Mat();
            Core.bitwise_and(rawMask, compRegion, compMask);
            compRegion.release();

            if (Core.countNonZero(compMask) < MIN_PIXEL_COUNT) {
                compMask.release();
                continue;
            }
            result.add(new ColourCluster(compMask, hue, achromatic, bright, lo, hi));
        }
        labels.release();
        return result;
    }

    // =========================================================================
    // Mask builders
    // =========================================================================

    private static Mat buildChromaticMask(Mat hsv) {
        Mat m = new Mat();
        Core.inRange(hsv, new Scalar(0, MIN_SAT, MIN_VAL), new Scalar(179, 255, 255), m);
        return m;
    }

    private static Mat buildAchromaticMask(Mat hsv, boolean bright) {
        Mat m = new Mat();
        // Cap saturation at MIN_SAT-1 for both achromatic types so they are
        // strictly non-overlapping with the chromatic mask (sat >= MIN_SAT).
        // Without this, dark-saturated pixels (e.g. dark orange, dark green)
        // appear in both a chromatic cluster AND the dark achromatic cluster.
        double satMax = MIN_SAT - 1; // 34 → achromatic; 35+ → chromatic
        if (bright) {
            // Bright achromatic: low saturation, high value (white / light grey)
            Core.inRange(hsv,
                    new Scalar(0, 0,      BRIGHT_VAL_THRESHOLD),
                    new Scalar(179, satMax, 255), m);
        } else {
            // Dark achromatic: low saturation, low value (black / dark grey)
            // Cap val at BRIGHT_VAL_THRESHOLD-1 to avoid 1-bin overlap with bright.
            Core.inRange(hsv,
                    new Scalar(0, 0, 0),
                    new Scalar(179, satMax, BRIGHT_VAL_THRESHOLD - 1), m);
        }
        return m;
    }

    private static Mat buildHealKernel() {
        int d = 2 * HEAL_RADIUS + 1;
        return Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(d, d));
    }

    // =========================================================================
    // Histogram helpers (identical internals to SceneColourClusters)
    // =========================================================================

    private static void zeroImageBorder(Mat mask) {
        mask.row(0).setTo(Scalar.all(0));
        mask.row(mask.rows() - 1).setTo(Scalar.all(0));
        mask.col(0).setTo(Scalar.all(0));
        mask.col(mask.cols() - 1).setTo(Scalar.all(0));
    }

    private static float[] buildHueHistogram(Mat hsv, Mat mask) {
        List<Mat> ch = new ArrayList<>();
        Core.split(hsv, ch);
        Mat hueChannel = ch.get(0);
        Mat hist = new Mat();
        Imgproc.calcHist(List.of(hueChannel), new MatOfInt(0), mask,
                hist, new MatOfInt(180), new MatOfFloat(0f, 180f));
        float[] result = new float[180];
        for (int i = 0; i < 180; i++) result[i] = (float) hist.get(i, 0)[0];
        hist.release();
        hueChannel.release();
        for (int i = 1; i < ch.size(); i++) ch.get(i).release();
        return result;
    }

    private static float[] computeSmoothedHist(float[] hist, int r) {
        float[] s = new float[180];
        for (int i = 0; i < 180; i++) {
            float sum = 0;
            for (int d = -r; d <= r; d++) sum += hist[(i + d + 180) % 180];
            s[i] = sum / (2 * r + 1);
        }
        return s;
    }

    /**
     * Finds all local hue peaks — <b>no cap on count</b>.
     * Non-maximum suppression uses {@link #PEAK_MIN_SEPARATION} to avoid
     * adjacent bins of the same peak being returned multiple times.
     */
    private static List<Integer> findPeaks(float[] smoothed, int minCount) {
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
        for (int i = 0; i < peaks.size(); i++) {
            if (suppressed[i]) continue;
            result.add(peaks.get(i)[0]);
            for (int j = i + 1; j < peaks.size(); j++) {
                int d = Math.abs(peaks.get(i)[0] - peaks.get(j)[0]);
                if (Math.min(d, 180 - d) < PEAK_MIN_SEPARATION) suppressed[j] = true;
            }
        }
        return result; // no MAX_CLUSTERS cap
    }

    private static int[][] computeValleyBounds(float[] hist, List<Integer> ampPeaks) {
        int n = ampPeaks.size();
        int[][] result = new int[n][2];
        if (n == 0) return result;
        if (n == 1) {
            int p = ampPeaks.get(0);
            result[0][0] = Math.max(0,   (int)(p - HUE_TOLERANCE));
            result[0][1] = Math.min(179, (int)(p + HUE_TOLERANCE));
            return result;
        }
        List<Integer> hueSorted = new ArrayList<>(ampPeaks);
        Collections.sort(hueSorted);
        int hn = hueSorted.size();
        int[] valleyAfter = new int[hn];
        for (int i = 0; i < hn; i++)
            valleyAfter[i] = findValleyBetween(hist, hueSorted.get(i), hueSorted.get((i + 1) % hn));
        for (int ai = 0; ai < n; ai++) {
            int hPos  = hueSorted.indexOf(ampPeaks.get(ai));
            int loPos = (hPos - 1 + hn) % hn;
            result[ai][0] = valleyAfter[loPos];
            result[ai][1] = valleyAfter[hPos];
        }
        return result;
    }

    private static int findValleyBetween(float[] hist, int from, int to) {
        int dist = (to - from + 180) % 180;
        if (dist <= 2) return (from + 1) % 180;
        float minVal = Float.MAX_VALUE;
        for (int d = 1; d < dist; d++) minVal = Math.min(minVal, hist[(from + d) % 180]);
        int midDist = dist / 2, bestPos = (from + midDist) % 180, bestClose = Integer.MAX_VALUE;
        for (int d = 1; d < dist; d++) {
            if (hist[(from + d) % 180] <= minVal + 0.5f) {
                int close = Math.abs(d - midDist);
                if (close < bestClose) { bestClose = close; bestPos = (from + d) % 180; }
            }
        }
        return bestPos;
    }

    private static Mat hueRangeMask(Mat hsv, int peakHue, double tolerance) {
        double lo = peakHue - tolerance, hi = peakHue + tolerance;
        Mat mask = new Mat();
        if (lo < 0) {
            Mat m1 = new Mat(), m2 = new Mat();
            Core.inRange(hsv, new Scalar(0, MIN_SAT, MIN_VAL), new Scalar(hi, 255, 255), m1);
            Core.inRange(hsv, new Scalar(180 + lo, MIN_SAT, MIN_VAL), new Scalar(179, 255, 255), m2);
            Core.bitwise_or(m1, m2, mask); m1.release(); m2.release();
        } else if (hi > 179) {
            Mat m1 = new Mat(), m2 = new Mat();
            Core.inRange(hsv, new Scalar(lo, MIN_SAT, MIN_VAL), new Scalar(179, 255, 255), m1);
            Core.inRange(hsv, new Scalar(0,  MIN_SAT, MIN_VAL), new Scalar(hi - 180, 255, 255), m2);
            Core.bitwise_or(m1, m2, mask); m1.release(); m2.release();
        } else {
            Core.inRange(hsv, new Scalar(lo, MIN_SAT, MIN_VAL), new Scalar(hi, 255, 255), mask);
        }
        return mask;
    }

    /**
     * Clamps valley-derived {@code lo}/{@code hi} so neither boundary is
     * more than {@link #MAX_HUE_HALF_WIDTH} hue units away from {@code peak}.
     *
     * <p>Valley detection assigns the midpoint of the empty arc between two
     * detected peaks as the boundary for each cluster.  When only a small
     * number of peaks are present (e.g. red + orange in an otherwise
     * achromatic scene), one cluster ends up with a boundary on the far side
     * of the hue wheel — its range sweeps through blue, green and purple
     * even though none of those hues were detected.  This cap prevents that.
     *
     * <p>The valley boundary toward a <em>real neighbour</em> is kept even if
     * it is closer than {@code MAX_HUE_HALF_WIDTH}; the cap only shrinks a
     * boundary that is too wide, never widens it.
     *
     * @return {@code int[2]} = {@code {clampedLo, clampedHi}}
     */
    private static int[] clampBounds(int lo, int hi, int peak) {
        // Arc distance from lo to peak (clockwise: lo → peak)
        int distBefore = (peak - lo + 180) % 180;
        int clampedLo  = distBefore > MAX_HUE_HALF_WIDTH
                ? (peak - MAX_HUE_HALF_WIDTH + 180) % 180
                : lo;

        // Arc distance from peak to hi (clockwise: peak → hi)
        int distAfter  = (hi - peak + 180) % 180;
        int clampedHi  = distAfter > MAX_HUE_HALF_WIDTH
                ? (peak + MAX_HUE_HALF_WIDTH) % 180
                : hi;

        return new int[]{ clampedLo, clampedHi };
    }

    private static Mat hueRangeMaskByBounds(Mat hsv, int lo, int hi) {
        // hi is exclusive: the valley hue bin itself is not assigned to either
        // neighbouring cluster, preventing the same pixel from appearing in two masks.
        if (lo == hi) return Mat.zeros(hsv.rows(), hsv.cols(), CvType.CV_8UC1); // zero-width
        // Map exclusive hi to inclusive: hi=0 wraps to 179 (i.e. "up to but not including hue 0")
        int hiIncl = (hi == 0) ? 179 : hi - 1;
        Mat mask = new Mat();
        if (lo <= hiIncl) {
            Core.inRange(hsv, new Scalar(lo, MIN_SAT, MIN_VAL), new Scalar(hiIncl, 255, 255), mask);
        } else {
            // Wrap-around (e.g. lo=159, hi=14 → hiIncl=13)
            Mat m1 = new Mat(), m2 = new Mat();
            Core.inRange(hsv, new Scalar(lo, MIN_SAT, MIN_VAL), new Scalar(179, 255, 255), m1);
            Core.inRange(hsv, new Scalar(0,  MIN_SAT, MIN_VAL), new Scalar(hiIncl, 255, 255), m2);
            Core.bitwise_or(m1, m2, mask); m1.release(); m2.release();
        }
        return mask;
    }
}

