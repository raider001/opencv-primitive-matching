package org.example.colour;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
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
     *
     * <p>Item 14: reduced from 8→3 (17×17→7×7).  Anti-aliasing and thin
     * separation gaps are 1–4 px wide; a 7×7 kernel bridges up to 6 px,
     * which is sufficient.  The 17×17 kernel over-heals on noisy backgrounds,
     * spuriously merging distinct components.
     */
    public static final int HEAL_RADIUS = 3;

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
    private static final int    PEAK_MIN_SEPARATION  = SceneColourClusters.PEAK_MIN_SEPARATION;
    private static final double MIN_SAT              = 35.0;
    private static final double MIN_VAL              = 25.0;
    private static final double BRIGHT_VAL_THRESHOLD = 100.0;
    private static final int    MIN_PIXEL_COUNT      = 64;

    /**
     * Minimum bin separation (0–255 saturation scale) required between two
     * distinct S peaks within a single hue cluster.  Peaks closer than this
     * are treated as the same saturation mode and merged into one sub-cluster.
     * 32 ≈ 12.5 % of the full saturation range.
     */
    private static final int S_PEAK_MIN_SEP = 32;

    /**
     * Maximum number of spatial components returned per cluster from
     * {@link #splitSpatially}.  When more than this many components survive
     * {@link #MIN_PIXEL_COUNT}, the effective pixel-count floor is raised so
     * only the {@code MAX_SPLIT_COMPONENTS} largest survive.
     *
     * <p>Item 16: bounds Mat allocations, {@code put()} calls, and downstream
     * {@code contoursFromMask} calls regardless of background complexity.
     * Small noise fragments are never matched against reference shapes; shapes
     * are always larger.
     */
    private static final int MAX_SPLIT_COMPONENTS = 24;

    /**
     * Maximum recursion depth for Otsu-based S sub-cluster detection.
     * Depth 1 → at most a binary split (2 sub-clusters per hue cluster).
     */
    private static final int S_MAX_DEPTH = 1;

    // Integer-typed copies of the thresholds used in the hot single-pass loop
    // to avoid repeated double→int casts.
    private static final int MIN_SAT_I    = (int) MIN_SAT;              // 35
    private static final int MIN_VAL_I    = (int) MIN_VAL;              // 25
    private static final int BRIGHT_VAL_I = (int) BRIGHT_VAL_THRESHOLD; // 100
    private static final int SAT_MAX_I    = MIN_SAT_I - 1;              // 34

    private ExperimentalSceneColourClusters() {}

    // =========================================================================
    // SceneColourExtractor — primary entry points
    // =========================================================================

    /**
     * Border-pixel histogram variant.
     *
     * <p>Hue peaks are detected from the morphological-gradient border pixels.
     * Full-pixel masks are built for each peak via the single-pass O(n) helper,
     * then auto-healed and split spatially by connected components.
     */
    @Override
    public List<ColourCluster> extractFromBorderPixels(Mat bgrScene) {
        return extractFromBorderPixels(bgrScene, null);
    }

    /**
     * Border-pixel histogram variant with optional chromatic accumulation.
     *
     * <p>When {@code chromaticOut} is non-null it must be a pre-allocated
     * {@code byte[rows * cols]} array.  Step 7 of {@link #buildClustersOnePass}
     * will write {@code (byte) 255} for every pixel that falls into a chromatic
     * cluster, allowing the caller to build {@code combinedChromaticMask} with a
     * single {@code Mat.put()} instead of N {@code Core.bitwise_or} calls.
     */
    public List<ColourCluster> extractFromBorderPixels(Mat bgrScene, byte[] chromaticOut) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(bgrScene, hsv, Imgproc.COLOR_BGR2HSV);

        // ── Pure-Java border detection ────────────────────────────────────
        // Replaces buildChromaticMask (Core.inRange) + morphologyEx (MORPH_GRADIENT)
        // + borderMask.get() — eliminates 2 native full-image passes + 1 JNI bulk
        // read + 2 Mat allocations.  The chromatic classification and 3×3
        // morphological gradient are computed entirely in Java from the already-
        // bulk-read HSV data.
        int rows = hsv.rows(), cols = hsv.cols();
        int n    = rows * cols;
        byte[] hsvData = new byte[n * 3];
        hsv.get(0, 0, hsvData);

        // Pass A: classify each pixel as chromatic (1) or not (0).
        byte[] chromatic = new byte[n];
        for (int r = 1; r < rows - 1; r++) {
            int base = r * cols;
            for (int c = 1; c < cols - 1; c++) {
                int idx = (base + c) * 3;
                int s = hsvData[idx + 1] & 0xFF;
                int v = hsvData[idx + 2] & 0xFF;
                if (s >= MIN_SAT_I && v >= MIN_VAL_I) chromatic[base + c] = 1;
            }
        }

        // Pass B: morphological gradient — a pixel is on the border if any of
        // its 8-neighbors has a different chromatic status (replicates the 3×3
        // MORPH_GRADIENT of the binary chromatic mask).  Simultaneously count
        // border pixels for the degenerate-case fallback.
        byte[] borderData = new byte[n];
        int borderCount = 0;
        for (int r = 1; r < rows - 1; r++) {
            int base = r * cols;
            int prevRow = (r - 1) * cols;
            int nextRow = (r + 1) * cols;
            for (int c = 1; c < cols - 1; c++) {
                int i = base + c;
                byte me = chromatic[i];
                // Check 8-neighbors: if any differs, this pixel is on the gradient
                if (me != chromatic[prevRow + c - 1] || me != chromatic[prevRow + c] ||
                    me != chromatic[prevRow + c + 1] || me != chromatic[base + c - 1] ||
                    me != chromatic[base + c + 1]    || me != chromatic[nextRow + c - 1] ||
                    me != chromatic[nextRow + c]      || me != chromatic[nextRow + c + 1]) {
                    borderData[i] = (byte) 255;
                    borderCount++;
                }
            }
        }

        // Degenerate-case fallback: if too few border pixels, use all pixels.
        if (borderCount < MIN_PIXEL_COUNT) {
            borderData = null;
        }

        List<ColourCluster> result = buildClustersOnePass(hsv, hsvData, borderData, chromaticOut);
        hsv.release();
        return result;
    }

    /**
     * Full-pixel histogram variant.
     *
     * <p>Uses the entire chromatic pixel set for peak detection, then applies
     * the same auto-heal + connected-component split via the single-pass helper.
     */
    @Override
    public List<ColourCluster> extract(Mat bgrScene) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(bgrScene, hsv, Imgproc.COLOR_BGR2HSV);
        int n = hsv.rows() * hsv.cols();
        byte[] hsvData = new byte[n * 3];
        hsv.get(0, 0, hsvData);
        List<ColourCluster> result = buildClustersOnePass(hsv, hsvData, null, null);
        hsv.release();
        return result;
    }

    // =========================================================================
    // O(n) single-pass cluster builder
    // =========================================================================

    /**
     * Full-spectrum O(n) cluster builder with saturation sub-clustering.
     *
     * <p>Each chromatic hue cluster is further split along the <b>S axis</b>:
     * the saturation histogram of the cluster's border pixels is smoothed and
     * searched for local maxima separated by ≥ {@link #S_PEAK_MIN_SEP} units.
     * Each distinct S peak produces an independent sub-cluster with its own
     * valley-derived S window, so two pixels that share a hue but differ
     * significantly in saturation (e.g. steel-blue S≈135 vs vivid-blue S≈255)
     * land in <em>separate</em> clusters.
     *
     * <p>Cluster membership requires <b>all three</b> of:
     * <ol>
     *   <li>H in the hue valley window.</li>
     *   <li>S in the sub-cluster's S window [sLo, sHi].</li>
     *   <li>V ≥ per-hue-cluster 10th-percentile lower bound.</li>
     * </ol>
     *
     * <p>The BRIGHT/DARK achromatic split uses a data-driven V valley
     * (see {@link #dynamicAchromaticThreshold}).
     */
    private List<ColourCluster> buildClustersOnePass(Mat hsv, byte[] hsvData, byte[] borderData, byte[] chromaticOut) {
        int rows = hsv.rows(), cols = hsv.cols();
        int n    = rows * cols;

        // ── 1. HSV data already bulk-read by caller ───────────────────────────

        // ── 2. Hue histogram + achromatic-V histogram (single loop) ──────────
        float[] hueHist   = new float[180];
        int[]   achrVHist = new int[256];

        if (borderData != null) {
            for (int r = 1; r < rows - 1; r++) {
                int off = r * cols;
                for (int c = 1; c < cols - 1; c++) {
                    int i = off + c;
                    if ((borderData[i] & 0xFF) == 0) continue;
                    int idx = i * 3;
                    int h = hsvData[idx]     & 0xFF;
                    int s = hsvData[idx + 1] & 0xFF;
                    int v = hsvData[idx + 2] & 0xFF;
                    if      (s >= MIN_SAT_I && v >= MIN_VAL_I && h < 180) hueHist[h]++;
                    else if (s <= SAT_MAX_I)                               achrVHist[v]++;
                }
            }
        } else {
            for (int r = 1; r < rows - 1; r++) {
                int off = r * cols;
                for (int c = 1; c < cols - 1; c++) {
                    int i = off + c;
                    int idx = i * 3;
                    int h = hsvData[idx]     & 0xFF;
                    int s = hsvData[idx + 1] & 0xFF;
                    int v = hsvData[idx + 2] & 0xFF;
                    if      (s >= MIN_SAT_I && v >= MIN_VAL_I && h < 180) hueHist[h]++;
                    else if (s <= SAT_MAX_I)                               achrVHist[v]++;
                }
            }
        }

        // ── 3. Chromatic hue peak detection + hue LUT ───────────────────────
        float[]       smoothed     = computeSmoothedHist(hueHist, 1);
        List<Integer> peaks        = findPeaks(smoothed, MIN_PIXEL_COUNT);
        int[][]       valleyBounds = computeValleyBounds(smoothed, peaks);
        int           numChromatic = peaks.size();

        int[] hueLUT       = new int[180];
        int[] clusterPeaks = new int[numChromatic];
        int[] clusterLo    = new int[numChromatic];
        int[] clusterHi    = new int[numChromatic];
        Arrays.fill(hueLUT, -1);

        for (int pi = 0; pi < numChromatic; pi++) {
            int peakHue = peaks.get(pi);
            int[] vb    = clampBounds(valleyBounds[pi][0], valleyBounds[pi][1], peakHue);
            int lo = vb[0], hi = vb[1];
            clusterPeaks[pi] = peakHue;
            clusterLo[pi]    = lo;
            clusterHi[pi]    = hi;
            int hiIncl = (hi == 0) ? 179 : hi - 1;
            if (lo <= hiIncl) {
                for (int h = lo; h <= hiIncl; h++) if (hueLUT[h] < 0) hueLUT[h] = pi;
            } else {
                for (int h = lo; h < 180;     h++) if (hueLUT[h] < 0) hueLUT[h] = pi;
                for (int h = 0;  h <= hiIncl; h++) if (hueLUT[h] < 0) hueLUT[h] = pi;
            }
        }

        // ── 4. Per-cluster S histogram (all interior pixels) ──────────────
        int[][] sHistByCi = new int[numChromatic][256];

        for (int r = 1; r < rows - 1; r++) {
            int off = r * cols;
            for (int c = 1; c < cols - 1; c++) {
                int i   = off + c;
                int idx = i * 3;
                int h = hsvData[idx]     & 0xFF;
                int s = hsvData[idx + 1] & 0xFF;
                int v = hsvData[idx + 2] & 0xFF;
                if (s >= MIN_SAT_I && v >= MIN_VAL_I && h < 180) {
                    int ci = hueLUT[h];
                    if (ci >= 0) sHistByCi[ci][s]++;
                }
            }
        }

        // ── 5. S sub-cluster windows via recursive Otsu thresholding ─────────
        //
        // Replaces histogram peak detection.  Peak detection requires a sharp
        // local maximum in the S histogram to fire, which fails when one S mode
        // is broad/diffuse (e.g. a background whose orange tiles vary from
        // S=100 to S=170 with no single dominant bin).  Otsu thresholding
        // finds the optimal binary split between two classes regardless of
        // whether either distribution is sharp or wide — it only requires that
        // the two class *means* differ by ≥ 2×S_PEAK_MIN_SEP (= 64 S units)
        // and each class holds ≥ MIN_PIXEL_COUNT pixels.
        int[][] subSLo   = new int[numChromatic][];
        int[][] subSHi   = new int[numChromatic][];
        int[][] subGIdx  = new int[numChromatic][];
        int[]   subCount = new int[numChromatic];
        int     nextGIdx = 0;

        for (int ci = 0; ci < numChromatic; ci++) {
            List<int[]> bands = new ArrayList<>();
            computeSBands(sHistByCi[ci], MIN_SAT_I, 255, S_MAX_DEPTH, bands);

            int K = bands.size();
            subSLo[ci]   = new int[K];
            subSHi[ci]   = new int[K];
            subGIdx[ci]  = new int[K];
            subCount[ci] = K;

            for (int si = 0; si < K; si++) {
                subSLo[ci][si]  = bands.get(si)[0];
                subSHi[ci][si]  = bands.get(si)[1];
                subGIdx[ci][si] = nextGIdx++;
            }
        }

        int brightIdx  = nextGIdx++;
        int darkIdx    = nextGIdx++;
        int totalSlots = nextGIdx;

        // ── 6. Dynamic achromatic BRIGHT/DARK threshold ───────────────────────
        int achrVThreshold = dynamicAchromaticThreshold(achrVHist);

        // ── 7. Single-pass pixel classification ──────────────────────────────
        // Chromatic: hue window → S sub-cluster linear scan.
        // The only V floor is the global MIN_VAL_I=25 check in the outer if.
        // A per-cluster V percentile gate was removed: it was computed from the
        // combined V distribution of all S populations in the hue cluster, so
        // it broke when the vivid-S pixels (high V) outnumbered the muted-S
        // pixels (lower V), wrongly blocking the lower-V population before
        // Otsu sub-clustering could separate them.
        //
        // If chromaticOut is non-null, each chromatic pixel also sets
        // chromaticOut[i] = 255 — letting the caller build combinedChromaticMask
        // with one Mat.put() instead of N bitwise_or JNI calls (Item 12).
        byte[][] maskData = new byte[totalSlots][n];
        int[]    pixCount = new int[totalSlots];

        for (int r = 1; r < rows - 1; r++) {
            int off = r * cols;
            for (int c = 1; c < cols - 1; c++) {
                int i   = off + c;
                int idx = i * 3;
                int h = hsvData[idx]     & 0xFF;
                int s = hsvData[idx + 1] & 0xFF;
                int v = hsvData[idx + 2] & 0xFF;

                if (s >= MIN_SAT_I && v >= MIN_VAL_I) {
                    if (h < 180) {
                        int ci = hueLUT[h];
                        if (ci >= 0) {
                            int[] sLo = subSLo[ci], sHi = subSHi[ci], gIdx = subGIdx[ci];
                            for (int si = 0; si < subCount[ci]; si++) {
                                if (s >= sLo[si] && s <= sHi[si]) {
                                    maskData[gIdx[si]][i] = (byte) 255;
                                    pixCount[gIdx[si]]++;
                                    if (chromaticOut != null) chromaticOut[i] = (byte) 255;
                                    break;
                                }
                            }
                        }
                    }
                } else if (s <= SAT_MAX_I) {
                    int ci = (v >= achrVThreshold) ? brightIdx : darkIdx;
                    maskData[ci][i] = (byte) 255;
                    pixCount[ci]++;
                }
            }
        }

        // ── 8. Spatial splitting ──────────────────────────────────────────────
        // maskData[gIdx] is passed directly as rawBytes — splitSpatially uses it
        // instead of calling rawMask.get(), eliminating a redundant native→Java
        // bulk read per cluster (Item 10).
        Mat healKernel = buildHealKernel();
        List<ColourCluster> clusters = new ArrayList<>();

        for (int ci = 0; ci < numChromatic; ci++) {
            for (int si = 0; si < subCount[ci]; si++) {
                int gIdx = subGIdx[ci][si];
                if (pixCount[gIdx] < MIN_PIXEL_COUNT) continue;
                Mat rawMask = new Mat(rows, cols, CvType.CV_8UC1);
                rawMask.put(0, 0, maskData[gIdx]);
                clusters.addAll(splitSpatially(rawMask, maskData[gIdx], clusterPeaks[ci], false, false,
                        clusterLo[ci], clusterHi[ci], subSLo[ci][si], subSHi[ci][si], healKernel));
                rawMask.release();
            }
        }

        if (pixCount[brightIdx] >= MIN_PIXEL_COUNT) {
            Mat rawMask = new Mat(rows, cols, CvType.CV_8UC1);
            rawMask.put(0, 0, maskData[brightIdx]);
            clusters.addAll(splitSpatially(rawMask, maskData[brightIdx], Double.NaN, true, true, 0, 179, 0, 255, healKernel));
            rawMask.release();
        }

        if (pixCount[darkIdx] >= MIN_PIXEL_COUNT) {
            Mat rawMask = new Mat(rows, cols, CvType.CV_8UC1);
            rawMask.put(0, 0, maskData[darkIdx]);
            clusters.addAll(splitSpatially(rawMask, maskData[darkIdx], Double.NaN, true, false, 0, 179, 0, 255, healKernel));
            rawMask.release();
        }

        // healKernel is the static cachedHealKernel — do not release it here.
        // The static field owns the kernel's lifetime across all calls.
        return clusters;
    }

    // =========================================================================
    // Full-HSV helpers
    // =========================================================================

    /**
     * Returns the histogram bin index at the {@code p}-th percentile.
     *
     * @param hist  frequency counts (non-negative)
     * @param total pre-computed sum of {@code hist}
     * @param p     percentile 0–100
     */
    private static int percentileIndex(int[] hist, int total, int p) {
        int target = Math.max(1, total * p / 100);
        int cum    = 0;
        for (int i = 0; i < hist.length; i++) {
            cum += hist[i];
            if (cum >= target) return i;
        }
        return hist.length - 1;
    }

    /**
     * Computes the Otsu threshold for the histogram slice {@code hist[lo..hi]}.
     *
     * <p>Returns the split point {@code T} such that {@code [lo, T]} is "class A"
     * and {@code [T+1, hi]} is "class B", maximising between-class variance.
     * Works for both symmetric bimodal <em>and</em> asymmetric distributions
     * (one sharp peak + one broad/diffuse cloud).
     */
    private static int otsuThreshold(int[] hist, int lo, int hi) {
        long total = 0, wSum = 0;
        for (int i = lo; i <= hi; i++) { total += hist[i]; wSum += (long) i * hist[i]; }
        if (total == 0) return (lo + hi) / 2;

        long   wB = 0, sumB = 0;
        double maxVar = -1;
        int    best   = lo;
        for (int i = lo; i < hi; i++) {           // T < hi so the right class is non-empty
            wB   += hist[i];
            sumB += (long) i * hist[i];
            long wF = total - wB;
            if (wB == 0 || wF == 0) continue;
            double mB  = (double) sumB         / wB;
            double mF  = (double)(wSum - sumB) / wF;
            double var = (double) wB * wF * (mB - mF) * (mB - mF);
            if (var > maxVar) { maxVar = var; best = i; }
        }
        return best;
    }

    /**
     * Recursively splits {@code hist[lo..hi]} into S sub-cluster bands using
     * Otsu thresholding, appending {@code int[]{lo, hi}} pairs to {@code out}.
     *
     * <p>A split is accepted only when both resulting sides have ≥
     * {@link #MIN_PIXEL_COUNT} pixels <em>and</em> the two class means differ
     * by ≥ {@code 2 × S_PEAK_MIN_SEP} (= 64 S units).  The mean-difference
     * guard rejects spurious splits of unimodal distributions: Otsu always
     * places a threshold somewhere, even for a single broad Gaussian, but the
     * resulting class means are too close to represent distinct saturation levels.
     *
     * @param depth max recursion depth (1 → binary split, 2 → up to 4 bands)
     */
    private static void computeSBands(int[] hist, int lo, int hi, int depth, List<int[]> out) {
        int total = 0;
        for (int s = lo; s <= hi; s++) total += hist[s];

        if (depth <= 0 || hi - lo < S_PEAK_MIN_SEP || total < MIN_PIXEL_COUNT * 2) {
            out.add(new int[]{lo, hi});
            return;
        }

        int  T          = otsuThreshold(hist, lo, hi);
        int  leftCount  = 0;
        long leftSum    = 0;
        for (int s = lo;    s <= T;  s++) { leftCount  += hist[s]; leftSum  += (long) s * hist[s]; }
        int  rightCount = total - leftCount;
        long rightSum   = 0;
        for (int s = T + 1; s <= hi; s++) rightSum += (long) s * hist[s];

        if (leftCount < MIN_PIXEL_COUNT || rightCount < MIN_PIXEL_COUNT) {
            out.add(new int[]{lo, hi});
            return;
        }

        double mLeft  = (double) leftSum  / leftCount;
        double mRight = (double) rightSum / rightCount;

        // Guard: only split when the two class means are genuinely far apart.
        // Prevents splitting a unimodal (single-material) hue cluster just because
        // Otsu found the midpoint of its S distribution.
        if (mRight - mLeft < S_PEAK_MIN_SEP * 2) {
            out.add(new int[]{lo, hi});
            return;
        }

        computeSBands(hist, lo,    T,  depth - 1, out);
        computeSBands(hist, T + 1, hi, depth - 1, out);
    }

    /**
     * Finds a data-driven V threshold between DARK and BRIGHT achromatic pixels
     * using Otsu thresholding on the achromatic V histogram.
     *
     * <p>Falls back to {@link #BRIGHT_VAL_I} when the histogram is unimodal,
     * either class has too few pixels, or the class V-means are &lt; 40 units apart.
     */
    private static int dynamicAchromaticThreshold(int[] achrVHist) {
        int total = 0;
        for (int v = 0; v < 256; v++) total += achrVHist[v];
        if (total < MIN_PIXEL_COUNT * 2) return BRIGHT_VAL_I;

        int  T          = otsuThreshold(achrVHist, 0, 255);
        int  darkCount  = 0;
        long darkSum    = 0;
        for (int v = 0;     v <= T;  v++) { darkCount += achrVHist[v]; darkSum += (long) v * achrVHist[v]; }
        int  brightCount = total - darkCount;
        long brightSum   = 0;
        for (int v = T + 1; v < 256; v++) brightSum += (long) v * achrVHist[v];

        if (darkCount < MIN_PIXEL_COUNT || brightCount < MIN_PIXEL_COUNT) return BRIGHT_VAL_I;

        double mDark   = (double) darkSum   / darkCount;
        double mBright = (double) brightSum / brightCount;
        // Guard against splitting a unimodal V distribution
        return (mBright - mDark >= 40) ? T + 1 : BRIGHT_VAL_I;
    }

    // =========================================================================
    // Spatial splitting — the core difference from production
    // =========================================================================

    /**
     * O(n) variant: two bulk reads replace K×3 native full-image scans.
     *
     * <p>Old approach: for each of K connected components, call
     * {@code Core.inRange} + {@code Core.bitwise_and} + {@code Core.countNonZero}
     * (three full W×H native scans each).  On a noisy background with 30–50
     * dark-achromatic components that was 90–150 full-image native calls per band.
     *
     * <p>New approach:</p>
     * <ol>
     *   <li>{@code connectedComponents} on raw mask first — cheap linear scan.</li>
     *   <li>{@code morphologyEx} (MORPH_CLOSE) <em>only</em> when raw component
     *       count &gt; 2, i.e. when healing may actually merge disconnected regions.
     *       Single-component clusters skip closing entirely.</li>
     *   <li>{@code connectedComponents} on healed mask (only if healing ran).</li>
     *   <li>Two bulk reads: {@code labels32.get()} + {@code rawMask.get()} — 2 JNI calls.</li>
     *   <li>One Java loop over all pixels: builds every component's {@code byte[]}
     *       mask and pixel count simultaneously.</li>
     *   <li>One {@code Mat.put()} per valid component — plain memcpy, no per-pixel work.</li>
     * </ol>
     * Total native work is now {@code O(W×H)} independent of component count K,
     * and the expensive closing kernel is skipped for already-connected clusters.
     */
    private static List<ColourCluster> splitSpatially(
            Mat rawMask, byte[] rawBytes, double hue, boolean achromatic, boolean bright,
            int lo, int hi, int sLo, int sHi, Mat healKernel) {

        int rows = rawMask.rows(), cols = rawMask.cols(), n = rows * cols;

        // ── 1. Fast-path: check raw component count before paying for MORPH_CLOSE ─
        Mat labels32 = new Mat();
        int numComponents = Imgproc.connectedComponents(rawMask, labels32);

        if (numComponents <= 1) {
            labels32.release();
            return Collections.emptyList();
        }

        // ── 2. Auto-heal only when multiple raw components exist ──────────────
        if (numComponents > 2) {
            Mat healed = new Mat();
            Imgproc.morphologyEx(rawMask, healed, Imgproc.MORPH_CLOSE, healKernel);
            labels32.release();
            labels32 = new Mat();
            numComponents = Imgproc.connectedComponents(healed, labels32);
            healed.release();
            if (numComponents <= 1) {
                labels32.release();
                return Collections.emptyList();
            }
        }

        // ── 3. Bulk read labels; use rawBytes directly — no rawMask.get() needed ──
        // rawBytes is maskData[gIdx] from buildClustersOnePass, which is the exact
        // same data that was just put() into rawMask.  Reusing it eliminates one
        // ~300KB native→Java bulk read per splitSpatially call (Item 10).
        int[] labelsData = new int[n];
        labels32.get(0, 0, labelsData);
        labels32.release();
        byte[] rawData = rawBytes;

        // ── 4. Two-pass lazy compMasks — count first, allocate only valid ─────
        // Old approach allocated byte[maxComp][n] upfront, wasting up to
        // maxComp×n bytes when most components are noise (Item 11).
        int maxComp = Math.min(numComponents, 255);

        // Pass 1: count pixels per component label
        int[] compCount = new int[maxComp];
        for (int i = 0; i < n; i++) {
            int label = labelsData[i];
            if (label > 0 && label < maxComp && (rawData[i] & 0xFF) > 0)
                compCount[label]++;
        }

        // Item 16: cap output to top-MAX_SPLIT_COMPONENTS components.
        // If more components survive MIN_PIXEL_COUNT, raise the effective floor
        // so only the K largest are materialised — bounds allocations and
        // contoursFromMask calls regardless of background complexity.
        int effectiveMin = MIN_PIXEL_COUNT;
        int validCount = 0;
        for (int comp = 1; comp < maxComp; comp++) {
            if (compCount[comp] >= MIN_PIXEL_COUNT) validCount++;
        }
        if (validCount > MAX_SPLIT_COMPONENTS) {
            // Collect valid pixel-counts, sort ascending, take the K-th from top
            int[] validCounts = new int[validCount];
            int vi = 0;
            for (int comp = 1; comp < maxComp; comp++) {
                if (compCount[comp] >= MIN_PIXEL_COUNT) validCounts[vi++] = compCount[comp];
            }
            Arrays.sort(validCounts);
            // validCounts[validCount - MAX_SPLIT_COMPONENTS] is the smallest
            // count that still makes the top-K cut.
            effectiveMin = Math.max(MIN_PIXEL_COUNT, validCounts[validCount - MAX_SPLIT_COMPONENTS]);
        }

        // Allocate byte[n] only for components that will survive effectiveMin
        byte[][] compMasks = new byte[maxComp][];
        for (int comp = 1; comp < maxComp; comp++) {
            if (compCount[comp] >= effectiveMin)
                compMasks[comp] = new byte[n];
        }

        // Pass 2: fill only the allocated masks
        for (int i = 0; i < n; i++) {
            int label = labelsData[i];
            if (label > 0 && label < maxComp && compMasks[label] != null && (rawData[i] & 0xFF) > 0)
                compMasks[label][i] = (byte) 255;
        }

        // ── 5. Materialise valid components (Mat.put = plain memcpy) ─────────
        List<ColourCluster> result = new ArrayList<>();
        for (int comp = 1; comp < maxComp; comp++) {
            if (compMasks[comp] == null) continue;
            Mat compMask = new Mat(rows, cols, CvType.CV_8UC1);
            compMask.put(0, 0, compMasks[comp]);
            result.add(new ColourCluster(compMask, hue, achromatic, bright, lo, hi, sLo, sHi));
        }
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

    // Cached heal kernel — HEAL_RADIUS is a compile-time constant so the kernel
    // shape never changes. Lazy-initialised on first use (OpenCV must be loaded).
    // Do NOT call release() on the returned Mat — the static field owns its lifetime.
    private static volatile Mat cachedHealKernel = null;

    private static synchronized Mat buildHealKernel() {
        if (cachedHealKernel == null || cachedHealKernel.empty()) {
            int d = 2 * HEAL_RADIUS + 1;
            // Item 13: MORPH_RECT uses OpenCV's O(n) running-sum decomposition
            // (two 1-D passes independent of kernel area), replacing the general
            // O(n × kernel_area) path used by MORPH_ELLIPSE.  For a 17×17
            // ellipse on a 640×480 image that was ~200 ms per close call; with
            // a 7×7 rect (Items 13+14 combined) it drops to ~1–2 ms.
            cachedHealKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(d, d));
        }
        return cachedHealKernel;
    }

    // Cached 3×3 MORPH_RECT kernel used for the morphological gradient border mask.
    // Shape never changes — owned by the static field, never released by callers.
    private static volatile Mat cachedBorderKernel = null;

    private static synchronized Mat buildBorderKernel() {
        if (cachedBorderKernel == null || cachedBorderKernel.empty()) {
            cachedBorderKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        }
        return cachedBorderKernel;
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

    // =========================================================================
    // Linear histogram helpers (non-circular — used for S and V dimensions)
    // =========================================================================

    /**
     * Non-circular boxcar smoothing of a frequency histogram.
     * Unlike {@link #computeSmoothedHist} (circular, for hue), this clamps
     * at the boundaries instead of wrapping.
     *
     * @param hist   source histogram (any length)
     * @param radius half-width of the boxcar (window = 2r+1)
     */
    private static float[] smoothLinear(float[] hist, int radius) {
        int     len = hist.length;
        float[] out = new float[len];
        for (int i = 0; i < len; i++) {
            float sum   = 0;
            int   count = 0;
            for (int d = -radius; d <= radius; d++) {
                int idx = i + d;
                if (idx >= 0 && idx < len) { sum += hist[idx]; count++; }
            }
            out[i] = (count > 0) ? sum / count : 0;
        }
        return out;
    }

    /**
     * Peak detection for a linear (non-circular) histogram.
     *
     * <p>Finds all local maxima above {@code minCount}, then applies
     * non-maximum suppression so peaks within {@code minSep} bins of a
     * taller peak are discarded.  Returns peak positions sorted ascending.
     *
     * @param hist     smoothed histogram
     * @param minCount minimum height to qualify as a peak
     * @param minSep   minimum bin distance between accepted peaks
     */
    private static List<Integer> findLinearPeaks(float[] hist, int minCount, int minSep) {
        List<int[]> candidates = new ArrayList<>();
        for (int i = 1; i < hist.length - 1; i++) {
            if (hist[i] > hist[i - 1] && hist[i] >= hist[i + 1] && hist[i] >= minCount)
                candidates.add(new int[]{i, (int) hist[i]});
        }
        candidates.sort((a, b) -> b[1] - a[1]);           // tallest first
        boolean[]     suppressed = new boolean[candidates.size()];
        List<Integer> result     = new ArrayList<>();
        for (int i = 0; i < candidates.size(); i++) {
            if (suppressed[i]) continue;
            result.add(candidates.get(i)[0]);
            for (int j = i + 1; j < candidates.size(); j++) {
                if (Math.abs(candidates.get(i)[0] - candidates.get(j)[0]) < minSep)
                    suppressed[j] = true;
            }
        }
        result.sort(null);                                  // ascending bin order
        return result;
    }

    /**
     * Returns the bin index of the minimum value in {@code hist(from+1 … to-1)}.
     * Falls back to the midpoint when the range has fewer than 2 interior bins.
     */
    private static int findLinearValley(float[] hist, int from, int to) {
        if (to - from < 2) return (from + to) / 2;
        int   pos = from + 1;
        float min = Float.MAX_VALUE;
        for (int i = from + 1; i < to; i++) {
            if (hist[i] < min) { min = hist[i]; pos = i; }
        }
        return pos;
    }
}

