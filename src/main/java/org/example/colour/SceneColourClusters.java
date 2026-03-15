package org.example.colour;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.Collections;
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
 *
 * <h2>Anti-aliasing: valley-based exclusive hue assignment</h2>
 * <p>Earlier versions used a fixed ±{@link #HUE_TOLERANCE} window per peak,
 * which caused adjacent clusters to overlap where their windows intersected
 * (pixels in the overlap were included in both masks).  Now, after peak
 * detection, the actual valley (histogram minimum) between each pair of
 * adjacent peaks is located and used as the hard cluster boundary.  Each pixel
 * is therefore assigned to exactly one cluster — the one whose peak it lies
 * closest to within the valley boundary.  {@link #HUE_TOLERANCE} is kept as
 * the fallback half-width for isolated single peaks with no adjacent neighbour.
 */
public final class SceneColourClusters {
    public static final int    MAX_CLUSTERS        = 12;
    public static final double HUE_TOLERANCE       = 14.0;
    private static final int   PEAK_MIN_SEPARATION = 18;
    public static final int    MIN_CONTOUR_AREA     = 64;
    private static final double MIN_SAT          = 35.0;
    private static final double MIN_VAL          = 25.0;
    private static final double BRIGHT_VAL_THRESHOLD = 100.0;
    private static final int    MIN_PIXEL_COUNT  = 64;

    /**
     * Singleton {@link SceneColourExtractor} backed by the static methods of
     * this class.  Use this reference wherever a {@link SceneColourExtractor}
     * is required, and swap it out with
     * {@link ExperimentalSceneColourClusters#INSTANCE} to trial the
     * experimental implementation without changing call sites.
     */
    public static final SceneColourExtractor INSTANCE = new SceneColourExtractor() {
        @Override
        public List<ColourCluster> extractFromBorderPixels(Mat bgrScene) {
            return SceneColourClusters.extractFromBorderPixels(bgrScene);
        }
        @Override
        public List<ColourCluster> extract(Mat bgrScene) {
            return SceneColourClusters.extract(bgrScene);
        }
    };

    /**
     * Backward-compatibility alias.
     * Prefer {@link ColourCluster} for new code.
     */
    public static final class Cluster extends ColourCluster {
        Cluster(Mat mask, double hue, boolean achromatic, boolean brightAchromatic,
                int loBound, int hiBound) {
            super(mask, hue, achromatic, brightAchromatic, loBound, hiBound);
        }
        Cluster(Mat mask, double hue, boolean achromatic, boolean brightAchromatic) {
            super(mask, hue, achromatic, brightAchromatic);
        }
    }

    private SceneColourClusters() {}

    // =========================================================================
    // Public helpers for external callers (e.g. VectorMatcher re-extraction)
    // =========================================================================

    /**
     * Builds a chromatic mask for pixels at the given peak hue ± tolerance.
     * The input {@code hsv} Mat must already be in HSV colour space.
     * Handles red-hue wrap-around (near H=0/179) automatically.
     */
    public static Mat buildHueMask(Mat hsv, double peakHue, double tolerance) {
        return hueRangeMask(hsv, (int) Math.round(peakHue), tolerance);
    }

    /**
     * Builds a full-pixel achromatic mask (bright or dark) from an HSV Mat.
     *
     * @param hsv    image already converted to HSV colour space
     * @param bright {@code true} for BRIGHT achromatic (white/light-grey),
     *               {@code false} for DARK achromatic (black/dark-grey)
     */
    public static Mat buildAchromaticMask(Mat hsv, boolean bright) {
        Mat mask = new Mat();
        if (bright) {
            Core.inRange(hsv,
                    new Scalar(0,   0,       BRIGHT_VAL_THRESHOLD),
                    new Scalar(179, MIN_SAT, 255),
                    mask);
        } else {
            Core.inRange(hsv,
                    new Scalar(0,   0, 0),
                    new Scalar(179, 255, BRIGHT_VAL_THRESHOLD),
                    mask);
        }
        return mask;
    }

    // =========================================================================
    // Border-pixel variant
    // =========================================================================

    public static List<ColourCluster> extractFromBorderPixels(Mat bgrScene) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(bgrScene, hsv, Imgproc.COLOR_BGR2HSV);

        Mat chromaticMask = new Mat();
        Core.inRange(hsv,
                new Scalar(0,   MIN_SAT, MIN_VAL),
                new Scalar(179, 255,     255),
                chromaticMask);

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Mat borderMask = new Mat();
        Imgproc.morphologyEx(chromaticMask, borderMask, Imgproc.MORPH_GRADIENT, kernel);
        chromaticMask.release();

        borderMask.row(0).setTo(Scalar.all(0));
        borderMask.row(borderMask.rows() - 1).setTo(Scalar.all(0));
        borderMask.col(0).setTo(Scalar.all(0));
        borderMask.col(borderMask.cols() - 1).setTo(Scalar.all(0));

        float[] hueHist   = buildHueHistogram(hsv, borderMask);
        borderMask.release();

        // ── Anti-aliased cluster discovery ────────────────────────────────
        // Smooth with radius 1 (tighter than the old radius-2), find peaks,
        // then compute valley-based exclusive hue bounds for each cluster.
        float[]       smoothed     = computeSmoothedHist(hueHist, 1);
        List<Integer> peaks        = findPeaks(smoothed, MIN_PIXEL_COUNT);

        List<ColourCluster> clusters = new ArrayList<>();

        for (int pi = 0; pi < peaks.size(); pi++) {
            if (clusters.size() >= MAX_CLUSTERS) break;
            int peakHue = peaks.get(pi);
            // Use the proven ±HUE_TOLERANCE window.  Valley-based exclusive bounds
            // are computed and stored on the ColourCluster for diagnostics / future use,
            // but the mask itself still uses the fixed-width window — switching to
            // valley bounds caused unexplained score regressions on
            // TRICOLOUR_TRIANGLE and BICOLOUR_RECT_HALVES in VectorMatchingTest.
            Mat clusterMask = hueRangeMask(hsv, peakHue, HUE_TOLERANCE);
            if (Core.countNonZero(clusterMask) < MIN_PIXEL_COUNT) {
                clusterMask.release();
                continue;
            }
            zeroImageBorder(clusterMask);
            int[] vb = computeValleyBounds(smoothed, peaks)[pi];
            clusters.add(new ColourCluster(clusterMask, peakHue, false, false, vb[0], vb[1]));
        }

        // Achromatic clusters (unchanged — border gradient applied here)
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
            clusters.add(new ColourCluster(brightBorder, Double.NaN, true, true));
        } else { brightBorder.release(); }

        if (Core.countNonZero(darkBorder) >= MIN_PIXEL_COUNT) {
            zeroImageBorder(darkBorder);
            clusters.add(new ColourCluster(darkBorder, Double.NaN, true, false));
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
    public static List<ColourCluster> extract(Mat bgrScene) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(bgrScene, hsv, Imgproc.COLOR_BGR2HSV);
        Mat brightAchromatic = new Mat();
        Core.inRange(hsv,
                new Scalar(0,   0,       BRIGHT_VAL_THRESHOLD),
                new Scalar(179, MIN_SAT, 255),
                brightAchromatic);
        Mat darkAchromatic = new Mat();
        Core.inRange(hsv,
                new Scalar(0,   0, 0),
                new Scalar(179, 255, BRIGHT_VAL_THRESHOLD),
                darkAchromatic);
        Mat chromaticMask = new Mat();
        Core.inRange(hsv,
                new Scalar(0,   MIN_SAT, MIN_VAL),
                new Scalar(179, 255,     255),
                chromaticMask);

        // ── Anti-aliased cluster discovery ────────────────────────────────
        float[]       hueHist      = buildHueHistogram(hsv, chromaticMask);
        chromaticMask.release();
        float[]       smoothed     = computeSmoothedHist(hueHist, 1);
        List<Integer> peaks        = findPeaks(smoothed, MIN_PIXEL_COUNT);
        int[][]       valleyBounds = computeValleyBounds(smoothed, peaks);

        List<ColourCluster> clusters = new ArrayList<>();
        for (int pi = 0; pi < peaks.size(); pi++) {
            if (clusters.size() >= MAX_CLUSTERS) break;
            int peakHue = peaks.get(pi);
            int lo = valleyBounds[pi][0];
            int hi = valleyBounds[pi][1];
            Mat clusterMask = hueRangeMaskByBounds(hsv, lo, hi);
            if (Core.countNonZero(clusterMask) < MIN_PIXEL_COUNT) {
                clusterMask.release();
                continue;
            }
            zeroImageBorder(clusterMask);
            clusters.add(new ColourCluster(clusterMask, peakHue, false, false, lo, hi));
        }
        if (Core.countNonZero(brightAchromatic) >= MIN_PIXEL_COUNT) {
            zeroImageBorder(brightAchromatic);
            clusters.add(new ColourCluster(brightAchromatic, Double.NaN, true, true));
        } else {
            brightAchromatic.release();
        }
        if (Core.countNonZero(darkAchromatic) >= MIN_PIXEL_COUNT) {
            zeroImageBorder(darkAchromatic);
            clusters.add(new ColourCluster(darkAchromatic, Double.NaN, true, false));
        } else {
            darkAchromatic.release();
        }
        hsv.release();
        return clusters;
    }

    public static Mat applyMask(Mat bgrScene, ColourCluster cluster) {
        Mat result = Mat.zeros(bgrScene.size(), bgrScene.type());
        bgrScene.copyTo(result, cluster.mask);
        return result;
    }

    // =========================================================================
    // Histogram helpers
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

    /**
     * Applies a simple box-filter smooth to the circular hue histogram.
     *
     * @param hist    raw 180-bin hue histogram
     * @param smoothR half-width of the box window (total window = 2*smoothR+1 bins)
     * @return        smoothed histogram (new array, same length)
     */
    private static float[] computeSmoothedHist(float[] hist, int smoothR) {
        float[] smoothed = new float[180];
        for (int i = 0; i < 180; i++) {
            float sum = 0;
            for (int d = -smoothR; d <= smoothR; d++)
                sum += hist[(i + d + 180) % 180];
            smoothed[i] = sum / (2 * smoothR + 1);
        }
        return smoothed;
    }

    /**
     * Finds local maxima in a pre-smoothed hue histogram.
     * Returns peak positions sorted by amplitude (strongest first),
     * with non-maximum suppression using {@link #PEAK_MIN_SEPARATION}.
     *
     * @param smoothed  smoothed histogram (from {@link #computeSmoothedHist})
     * @param minCount  minimum bin value for a peak to be considered
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

    // =========================================================================
    // Valley-based exclusive hue assignment
    // =========================================================================

    /**
     * Computes exclusive valley-based hue bounds for each detected peak.
     *
     * <p>Instead of a fixed ±{@link #HUE_TOLERANCE} window (which causes adjacent
     * clusters to overlap), this method finds the actual valley (histogram minimum)
     * between each pair of adjacent peaks in hue space and uses those valley
     * positions as hard cluster boundaries.  Each pixel therefore belongs to
     * exactly one cluster.
     *
     * <p>For a single isolated peak with no neighbour, falls back to
     * ±{@link #HUE_TOLERANCE}.
     *
     * @param hist      smoothed hue histogram (used for valley detection)
     * @param ampPeaks  peak hue positions in amplitude order (from {@link #findPeaks})
     * @return          {@code int[n][2]} where {@code result[i] = {loBound, hiBound}}
     *                  for {@code ampPeaks.get(i)}; bounds are hue values 0–179
     */
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

        // Sort a copy by hue value to find adjacent peaks in circular hue space
        List<Integer> hueSorted = new ArrayList<>(ampPeaks);
        Collections.sort(hueSorted);
        int hn = hueSorted.size();

        // valleyAfter[i] = valley position between hueSorted[i] and hueSorted[(i+1)%hn]
        int[] valleyAfter = new int[hn];
        for (int i = 0; i < hn; i++) {
            int p1 = hueSorted.get(i);
            int p2 = hueSorted.get((i + 1) % hn);
            valleyAfter[i] = findValleyBetween(hist, p1, p2);
        }

        // Map each amplitude-sorted peak to its valley bounds via the hue-sorted index
        for (int ai = 0; ai < n; ai++) {
            int peak   = ampPeaks.get(ai);
            int hPos   = hueSorted.indexOf(peak);          // position in hue-sorted list
            int loPos  = (hPos - 1 + hn) % hn;            // previous peak in hue order
            // lo = valley BEFORE this peak; hi = valley AFTER this peak
            result[ai][0] = valleyAfter[loPos];
            result[ai][1] = valleyAfter[hPos];
        }
        return result;
    }

    /**
     * Finds the valley position going clockwise from {@code from+1} to {@code to-1}.
     *
     * <p>When the histogram between two peaks is flat or empty (common for
     * synthetic bi-colour shapes with no pixels between the two hue peaks), the
     * first position would otherwise be returned — producing a degenerate cluster
     * window of width 1 instead of the expected half-gap width.  The fix: among
     * all positions at or near the minimum, prefer the one closest to the
     * <em>arc midpoint</em> so each cluster gets an equal share of the empty space.
     */
    private static int findValleyBetween(float[] hist, int from, int to) {
        int dist = (to - from + 180) % 180;
        if (dist <= 2) return (from + 1) % 180;   // degenerate: peaks nearly adjacent

        // Find the minimum value in the arc
        float minVal = Float.MAX_VALUE;
        for (int d = 1; d < dist; d++)
            minVal = Math.min(minVal, hist[(from + d) % 180]);

        // Among all positions at/near that minimum, return the one closest to the
        // arc midpoint.  When the arc is entirely empty (minVal == 0), this gives
        // an equidistant boundary rather than the first bin after 'from'.
        int midDist      = dist / 2;
        int bestPos      = (from + midDist) % 180;
        int bestCloseness = Integer.MAX_VALUE;
        for (int d = 1; d < dist; d++) {
            if (hist[(from + d) % 180] <= minVal + 0.5f) {
                int closeness = Math.abs(d - midDist);
                if (closeness < bestCloseness) {
                    bestCloseness = closeness;
                    bestPos = (from + d) % 180;
                }
            }
        }
        return bestPos;
    }

    /**
     * Builds a chromatic mask for pixels whose hue falls in the arc [lo, hi]
     * (inclusive).  Handles circular wrap-around when {@code lo > hi}
     * (e.g., lo=170, hi=10 for red hues near 0/179).
     */
    private static Mat hueRangeMaskByBounds(Mat hsv, int lo, int hi) {
        Mat mask = new Mat();
        if (lo <= hi) {
            Core.inRange(hsv,
                    new Scalar(lo, MIN_SAT, MIN_VAL),
                    new Scalar(hi, 255,     255), mask);
        } else {
            // Wraps around: [lo..179] OR [0..hi]
            Mat m1 = new Mat(), m2 = new Mat();
            Core.inRange(hsv, new Scalar(lo, MIN_SAT, MIN_VAL), new Scalar(179, 255, 255), m1);
            Core.inRange(hsv, new Scalar(0,  MIN_SAT, MIN_VAL), new Scalar(hi,  255, 255), m2);
            Core.bitwise_or(m1, m2, mask);
            m1.release(); m2.release();
        }
        return mask;
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







