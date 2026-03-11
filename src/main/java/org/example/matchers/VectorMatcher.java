package org.example.matchers;

import org.example.analytics.AnalysisResult;
import org.example.colour.SceneColourClusters;
import org.example.factories.ReferenceId;
import org.example.scene.SceneEntry;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Vector Matcher — scores a scene against a reference by decomposing both into
 * colour clusters, then comparing the contour geometry within each cluster.
 *
 * <h2>Three-layer scoring</h2>
 * <ol>
 *   <li><b>Cluster count</b> — how many distinct colour clusters the candidate
 *       region has versus the reference.  Mismatch is penalised exponentially.</li>
 *   <li><b>Per-cluster contour matching</b> — each reference cluster is paired
 *       with the best-matching scene cluster (by hue proximity), then every
 *       contour in the reference cluster is matched against the scene cluster's
 *       contours using geometric signatures.</li>
 *   <li><b>Geometry within clusters</b> — {@link VectorSignature#similarity}
 *       encodes circularity, solidity, vertex count, angle histogram, segment
 *       descriptor and topology — all scale- and rotation-invariant.</li>
 * </ol>
 *
 * <h2>Reference cluster discovery</h2>
 * <p>Reference clusters are identified from <em>border pixels only</em> via
 * {@link SceneColourClusters#extractFromBorderPixels(Mat)}.  This focuses on the
 * colours that actually form shape outlines, ignoring large filled interiors that
 * would skew the hue histogram.
 *
 * <h2>Variants</h2>
 * <p>Three variants — STRICT / NORMAL / LOOSE epsilon levels for polygon
 * approximation.  All three are always returned.
 */
public final class VectorMatcher {

    // Scoring layer weights — must sum to 1.0
    // Geometry carries the most weight: it is the only discriminator between
    // shapes that share the same cluster count and colour structure
    // (e.g. white circle vs white rectangle — both have 1 achromatic cluster).
    private static final double W_CLUSTER_COUNT  = 0.10;  // Layer 1: cluster count match
    private static final double W_CLUSTER_MATCH  = 0.20;  // Layer 2: per-cluster contour match
    private static final double W_GEOMETRY       = 0.70;  // Layer 3: geometry within clusters

    // Exponential decay constant for cluster-count penalty
    private static final double CLUSTER_PENALTY_K = 0.5;

    // Maximum hue distance (OpenCV half-degrees) to pair a scene cluster with a ref cluster
    private static final double MAX_HUE_DISTANCE = 20.0;

    private VectorMatcher() {}

    // =========================================================================
    // Public entry points
    // =========================================================================

    /**
     * Primary entry point — uses the {@link SceneDescriptor} pre-built inside
     * the {@link SceneEntry}.
     */
    public static List<AnalysisResult> match(ReferenceId referenceId,
                                             Mat refMat,
                                             SceneEntry scene,
                                             Set<String> saveVariants,
                                             Path outputDir) {
        SceneDescriptor descriptor = scene.descriptor();
        if (descriptor != null) {
            return matchWithDescriptor(referenceId, refMat, scene, descriptor, saveVariants, outputDir);
        }
        SceneDescriptor temp = SceneDescriptor.build(scene.sceneMat());
        try {
            return matchWithDescriptor(referenceId, refMat, scene, temp, saveVariants, outputDir);
        } finally {
            temp.release();
        }
    }

    /**
     * Overload for callers that supply their own {@link SceneDescriptor}.
     * The descriptor is <em>not</em> released by this method.
     */
    public static List<AnalysisResult> match(ReferenceId referenceId,
                                             Mat refMat,
                                             SceneEntry scene,
                                             SceneDescriptor descriptor,
                                             Set<String> saveVariants,
                                             Path outputDir) {
        return matchWithDescriptor(referenceId, refMat, scene, descriptor, saveVariants, outputDir);
    }

    // =========================================================================
    // Core matching
    // =========================================================================

    private static List<AnalysisResult> matchWithDescriptor(ReferenceId referenceId,
                                                             Mat refMat,
                                                             SceneEntry scene,
                                                             SceneDescriptor descriptor,
                                                             Set<String> saveVariants,
                                                             Path outputDir) {
        // Build reference clusters from border pixels (one set — reused across variants)
        List<RefCluster> refClusters = buildRefClusters(refMat);
        int refClusterCount = refClusters.size();

        List<AnalysisResult> out = new ArrayList<>(3);
        for (VectorVariant variant : VectorVariant.values()) {
            out.add(runVariant(variant, refClusters, refClusterCount,
                    descriptor, scene, referenceId, saveVariants, outputDir));
        }

        // Release reference cluster resources
        for (RefCluster rc : refClusters) rc.release();
        return out;
    }

    // =========================================================================
    // Single variant
    // =========================================================================

    private static AnalysisResult runVariant(VectorVariant variant,
                                             List<RefCluster> refClusters,
                                             int refClusterCount,
                                             SceneDescriptor descriptor,
                                             SceneEntry scene,
                                             ReferenceId referenceId,
                                             Set<String> saveVariants,
                                             Path outputDir) {
        long t0 = System.currentTimeMillis();
        try {
            double eps = variant.epsilonFactor();

            // Collect all candidate contours from the scene (excluding envelope entries)
            List<SceneContourEntry> candidates = collectSceneCandidates(descriptor, eps);

            if (candidates.isEmpty()) {
                return zeroResult(variant, referenceId, scene, descriptor, t0);
            }

            // Find best-scoring candidate region
            double bestScore = 0.0;
            Rect   bestBbox  = null;

            // Strategy: for each scene contour as an "anchor", build a candidate
            // bounding region by greedily pulling in the scene clusters that best
            // match the reference clusters, score the combination, keep the best.
            for (int anchorIdx = 0; anchorIdx < candidates.size(); anchorIdx++) {
                SceneContourEntry anchor = candidates.get(anchorIdx);
                Rect anchorBbox = Imgproc.boundingRect(anchor.contour);

                // Expand the anchor bbox to include the best scene cluster for each ref cluster
                Rect regionBbox = anchorBbox;
                List<SceneContourEntry> matched = new ArrayList<>();
                matched.add(anchor);

                // Track which scene cluster indices have been consumed
                Set<Integer> usedClusterIdx = new HashSet<>();
                usedClusterIdx.add(anchor.clusterIdx);

                for (RefCluster rc : refClusters) {
                    // Skip if this ref cluster is the one the anchor already covers
                    if (hueMatch(rc, anchor)) continue;
                    // Find the best scene contour from an unused cluster that matches this ref cluster
                    SceneContourEntry best = null;
                    double bestSim = 0.0;
                    for (SceneContourEntry ce : candidates) {
                        if (usedClusterIdx.contains(ce.clusterIdx)) continue;
                        if (!hueCompatible(rc, ce, descriptor)) continue;
                        double sim = rc.bestSig(eps).similarity(ce.sig);
                        if (sim > bestSim) { bestSim = sim; best = ce; }
                    }
                    if (best != null && bestSim > 0.05) {
                        matched.add(best);
                        usedClusterIdx.add(best.clusterIdx);
                        regionBbox = unionRect(regionBbox, Imgproc.boundingRect(best.contour));
                    }
                }

                // Score this candidate region
                double score = scoreRegion(refClusters, refClusterCount,
                        matched, descriptor, regionBbox, eps);

                if (score > bestScore) {
                    bestScore = score;
                    bestBbox  = regionBbox;
                }
            }

            double scorePercent = bestScore * 100.0;
            long   elapsed      = System.currentTimeMillis() - t0;

            Path savedPath = null;
            if (saveVariants.contains(variant.variantName())) {
                savedPath = writeAnnotated(scene.sceneMat(), bestBbox, variant.variantName(),
                        scorePercent, referenceId, scene, outputDir);
            }

            return new AnalysisResult(variant.variantName(), referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    scorePercent, bestBbox, elapsed, 0L,
                    (int) descriptor.sceneArea, savedPath, false, null);

        } catch (Exception e) {
            return AnalysisResult.error(variant.variantName(), referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    System.currentTimeMillis() - t0,
                    (int) descriptor.sceneArea,
                    e.getMessage());
        }
    }

    // =========================================================================
    // Three-layer score computation
    // =========================================================================

    /**
     * Scores a candidate set of matched scene contours against the reference clusters.
     *
     * <h3>Layer 1 — Cluster count match (weight 0.20)</h3>
     * Exponential penalty for mismatch in total colour-cluster count between the
     * candidate region and the reference.
     *
     * <h3>Layer 2 — Per-cluster contour coverage (weight 0.50)</h3>
     * For each reference cluster, find the matched scene entry and score contour
     * presence.  Missing clusters contribute 0.  Extra scene clusters are not penalised.
     *
     * <h3>Layer 3 — Geometry within matched clusters (weight 0.30)</h3>
     * Average {@link VectorSignature#similarity} across all matched ref→scene contour pairs.
     */
    private static double scoreRegion(List<RefCluster> refClusters,
                                      int refClusterCount,
                                      List<SceneContourEntry> matchedEntries,
                                      SceneDescriptor descriptor,
                                      Rect regionBbox,
                                      double eps) {

        // ── Layer 1: cluster count ────────────────────────────────────────
        int sceneClustersInRegion = countDistinctClusters(matchedEntries);
        double clusterCountScore;
        if (refClusterCount == 0) {
            clusterCountScore = 1.0;
        } else {
            int diff = Math.abs(sceneClustersInRegion - refClusterCount);
            clusterCountScore = Math.exp(-CLUSTER_PENALTY_K * diff);
        }

        // ── Layer 2 + 3: per-cluster contour matching + geometry ──────────
        double totalClusterScore   = 0.0;
        double totalGeometryScore  = 0.0;
        int    matchedClusterCount = 0;

        for (RefCluster rc : refClusters) {
            // Find matched scene entry for this ref cluster
            SceneContourEntry sceneEntry = findMatchForCluster(rc, matchedEntries, descriptor);
            if (sceneEntry == null) {
                // Missing cluster — zero contribution to Layer 2 and 3
                continue;
            }
            // Layer 2: cluster was found → full coverage credit for this cluster
            totalClusterScore += 1.0;
            // Layer 3: geometry similarity
            double geomSim = rc.bestSig(eps).similarity(sceneEntry.sig);
            totalGeometryScore += geomSim;
            matchedClusterCount++;
        }

        // Normalise Layer 2 and 3 by total ref cluster count (missing = 0)
        double clusterMatchScore = (refClusterCount > 0)
                ? totalClusterScore / refClusterCount : 1.0;
        double geometryScore = (matchedClusterCount > 0)
                ? totalGeometryScore / matchedClusterCount : 0.0;

        // ── Combine layers ────────────────────────────────────────────────
        // When there is only one cluster and it is achromatic (i.e. a plain
        // white/grey shape on a black background), the cluster-count and
        // cluster-match layers are identical for ALL such shapes — they all
        // have exactly one achromatic cluster.  Geometry is then the only
        // true discriminator.  Shift weights to give geometry 70% so that
        // a circle and a rectangle are properly separated by their shape metrics.
        double wCount, wMatch, wGeom;
        boolean singleAchromatic = (refClusterCount == 1)
                && refClusters.stream().allMatch(rc -> rc.achromatic);
        if (singleAchromatic) {
            wCount = 0.05;
            wMatch = 0.25;
            wGeom  = 0.70;
        } else {
            wCount = W_CLUSTER_COUNT;
            wMatch = W_CLUSTER_MATCH;
            wGeom  = W_GEOMETRY;
        }

        double combined = clusterCountScore  * wCount
                        + clusterMatchScore  * wMatch
                        + geometryScore      * wGeom;

        return Math.max(0.0, Math.min(1.0, combined));
    }

    // =========================================================================
    // Reference cluster builder
    // =========================================================================

    /**
     * Builds one {@link RefCluster} per colour cluster in the reference image,
     * discovering clusters from <em>border pixels only</em> via
     * {@link SceneColourClusters#extractFromBorderPixels(Mat)}.
     */
    static List<RefCluster> buildRefClusters(Mat refBgr) {
        List<SceneColourClusters.Cluster> rawClusters =
                SceneColourClusters.extractFromBorderPixels(refBgr);

        List<RefCluster> result = new ArrayList<>();
        double refArea = (double) refBgr.rows() * refBgr.cols();

        for (SceneColourClusters.Cluster cluster : rawClusters) {
            List<MatOfPoint> contours = SceneDescriptor.contoursFromMask(cluster.mask);
            if (!contours.isEmpty()) {
                result.add(new RefCluster(cluster.hue, cluster.achromatic, contours,
                        refArea, Core.countNonZero(cluster.mask)));
            }
            cluster.release();
        }

        // Fallback: if no clusters found at all, extract from full image
        if (result.isEmpty()) {
            List<SceneColourClusters.Cluster> fallback = SceneColourClusters.extract(refBgr);
            for (SceneColourClusters.Cluster cluster : fallback) {
                List<MatOfPoint> contours = SceneDescriptor.contoursFromMask(cluster.mask);
                if (!contours.isEmpty()) {
                    result.add(new RefCluster(cluster.hue, cluster.achromatic, contours,
                            refArea, Core.countNonZero(cluster.mask)));
                }
                cluster.release();
            }
        }

        return result;
    }

    // =========================================================================
    // Internal data structures
    // =========================================================================

    /** One colour cluster from the reference image with its contours. */
    static final class RefCluster {
        final double hue;
        final boolean achromatic;
        final List<MatOfPoint> contours;
        final double imageArea;
        final int pixelCount;
        // Cache of built signatures keyed by epsilon
        private final Map<Double, List<VectorSignature>> sigCache = new HashMap<>();

        RefCluster(double hue, boolean achromatic, List<MatOfPoint> contours,
                   double imageArea, int pixelCount) {
            this.hue        = hue;
            this.achromatic = achromatic;
            this.contours   = contours;
            this.imageArea  = imageArea;
            this.pixelCount = pixelCount;
        }

        /** Returns the best (largest area) signature for this cluster at the given epsilon. */
        VectorSignature bestSig(double eps) {
            List<VectorSignature> sigs = sigCache.computeIfAbsent(eps, e -> {
                List<VectorSignature> s = new ArrayList<>();
                for (MatOfPoint c : contours)
                    // Use NaN for imageArea so the normalised-area gate in
                    // VectorSignature.similarity() does not fire on cross-image
                    // comparisons (ref image is 128×128, scene is 640×480).
                    s.add(VectorSignature.buildFromContour(c, e, Double.NaN));
                return s;
            });
            // Return the sig with the highest solidity (most complete shape)
            VectorSignature best = null;
            double bestSolidity = -1;
            for (VectorSignature s : sigs) {
                if (s.solidity > bestSolidity) { bestSolidity = s.solidity; best = s; }
            }
            return best != null ? best : VectorSignature.build(
                    Mat.zeros(1, 1, CvType.CV_8UC1), eps, imageArea);
        }

        void release() {
            for (MatOfPoint c : contours) c.release();
        }
    }

    /** One contour from the scene with its cluster index and pre-built signature. */
    private record SceneContourEntry(MatOfPoint contour, int clusterIdx,
                                      boolean achromatic, double clusterHue,
                                      VectorSignature sig) {}

    // =========================================================================
    // Scene candidate collection
    // =========================================================================

    private static List<SceneContourEntry> collectSceneCandidates(SceneDescriptor descriptor,
                                                                   double eps) {
        List<SceneContourEntry> out = new ArrayList<>();
        List<SceneDescriptor.ClusterContours> clusters = descriptor.clusters();
        for (int ci = 0; ci < clusters.size(); ci++) {
            SceneDescriptor.ClusterContours cc = clusters.get(ci);
            if (cc.envelope) continue;
            for (MatOfPoint c : cc.contours) {
                VectorSignature sig = VectorSignature.buildFromContour(c, eps, descriptor.sceneArea);
                out.add(new SceneContourEntry(c, ci, cc.achromatic, cc.hue, sig));
            }
        }
        return out;
    }

    // =========================================================================
    // Cluster matching helpers
    // =========================================================================

    /** True if the scene contour's cluster matches the reference cluster by hue. */
    private static boolean hueMatch(RefCluster rc, SceneContourEntry ce) {
        if (rc.achromatic != ce.achromatic) return false;
        if (rc.achromatic) return true; // both achromatic — match
        return hueDist(rc.hue, ce.clusterHue) <= MAX_HUE_DISTANCE;
    }

    /** True if the scene contour could belong to the reference cluster (loose check). */
    private static boolean hueCompatible(RefCluster rc, SceneContourEntry ce,
                                          SceneDescriptor descriptor) {
        return hueMatch(rc, ce);
    }

    /** Circular hue distance in OpenCV half-degrees [0, 90]. */
    private static double hueDist(double a, double b) {
        if (Double.isNaN(a) || Double.isNaN(b)) return 0.0;
        double d = Math.abs(a - b);
        return Math.min(d, 180.0 - d);
    }

    /** Find the best-matching scene entry for a given reference cluster. */
    private static SceneContourEntry findMatchForCluster(RefCluster rc,
                                                          List<SceneContourEntry> entries,
                                                          SceneDescriptor descriptor) {
        SceneContourEntry best = null;
        double bestScore = -1;
        for (SceneContourEntry ce : entries) {
            if (!hueMatch(rc, ce)) continue;
            double score = rc.bestSig(0.04).similarity(ce.sig);
            if (score > bestScore) { bestScore = score; best = ce; }
        }
        return best;
    }

    /** Count how many distinct cluster indices appear in the matched entries. */
    private static int countDistinctClusters(List<SceneContourEntry> entries) {
        Set<Integer> seen = new HashSet<>();
        for (SceneContourEntry e : entries) seen.add(e.clusterIdx);
        return seen.size();
    }

    /** Returns the union (enclosing) rectangle of two rects. */
    private static Rect unionRect(Rect a, Rect b) {
        int x  = Math.min(a.x, b.x);
        int y  = Math.min(a.y, b.y);
        int x2 = Math.max(a.x + a.width,  b.x + b.width);
        int y2 = Math.max(a.y + a.height, b.y + b.height);
        return new Rect(x, y, x2 - x, y2 - y);
    }

    private static AnalysisResult zeroResult(VectorVariant variant, ReferenceId referenceId,
                                              SceneEntry scene, SceneDescriptor descriptor,
                                              long t0) {
        return new AnalysisResult(variant.variantName(), referenceId,
                scene.variantLabel(), scene.category(), scene.backgroundId(),
                0.0, null, System.currentTimeMillis() - t0, 0L,
                (int) descriptor.sceneArea, null, false, null);
    }

    // =========================================================================
    // Annotation writer
    // =========================================================================

    static Path writeAnnotated(Mat scene, Rect bbox, String variant, double score,
                                ReferenceId refId, SceneEntry sceneEntry, Path outputDir) {
        try {
            Path dir = outputDir.resolve("annotated").resolve(sanitise(variant));
            Files.createDirectories(dir);
            String sceneRef = sceneEntry.primaryReferenceId() != null
                    ? sanitise(sceneEntry.primaryReferenceId().name()) : "neg";
            String fname = sanitise(refId.name()) + "_vs_" + sceneRef
                    + "_" + sanitise(sceneEntry.variantLabel()) + ".png";
            Mat m = scene.clone();
            Scalar colour = score >= 70 ? new Scalar(0, 200, 0)
                          : score >= 40 ? new Scalar(0, 200, 200)
                          :               new Scalar(0, 0, 200);
            if (bbox != null && bbox.width > 1 && bbox.height > 1) {
                Imgproc.rectangle(m,
                        new Point(bbox.x, bbox.y),
                        new Point(bbox.x + bbox.width, bbox.y + bbox.height),
                        colour, 2);
            }
            Imgproc.putText(m, shortName(variant),
                    new Point(4, 13), Imgproc.FONT_HERSHEY_SIMPLEX, 0.32,
                    new Scalar(200, 200, 200), 1);
            Imgproc.putText(m, String.format("%.1f%%", score),
                    new Point(4, 28), Imgproc.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1);
            Path dest = dir.resolve(fname);
            Imgcodecs.imwrite(dest.toString(), m);
            m.release();
            return outputDir.toAbsolutePath().relativize(dest.toAbsolutePath());
        } catch (Exception e) {
            return null;
        }
    }

    // =========================================================================
    // Utility stubs — kept for compilation compatibility with callers
    // =========================================================================

    public static List<MatOfPoint> extractContoursFromBinary(Mat maskedBgr) {
        Mat grey = new Mat(), bin = new Mat();
        Imgproc.cvtColor(maskedBgr, grey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(grey, bin, 20, 255, Imgproc.THRESH_BINARY);
        grey.release();
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(bin, contours, hierarchy,
                Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        hierarchy.release(); bin.release();
        contours.removeIf(c -> Imgproc.contourArea(c) < SceneColourClusters.MIN_CONTOUR_AREA);
        return contours;
    }

    public static Mat extractBinaryRaw(Mat bgrScene) {
        Mat grey = new Mat(), bin = new Mat(), edge = new Mat();
        Imgproc.cvtColor(bgrScene, grey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(grey, bin, 20, 255, Imgproc.THRESH_BINARY);
        Imgproc.Canny(grey, edge, 40.0, 120.0);
        Core.bitwise_or(bin, edge, bin);
        grey.release(); edge.release();
        return bin;
    }

    public static Mat drawContourGraph(Size size, Mat binary,
                                       List<MatOfPoint> contours, double epsilon) {
        Mat out = Mat.zeros((int) size.height, (int) size.width, CvType.CV_8UC3);
        if (binary != null && !binary.empty()) {
            Mat grey3 = new Mat();
            Imgproc.cvtColor(binary, grey3, Imgproc.COLOR_GRAY2BGR);
            Core.multiply(grey3, new Scalar(0.4, 0.4, 0.4), grey3);
            Core.add(out, grey3, out);
            grey3.release();
        }
        int[][] palette = {
            {80,80,255},{80,255,80},{255,80,80},{80,255,255},
            {255,80,255},{255,255,80},{80,160,255},{255,80,160}
        };
        for (int ci = 0; ci < contours.size(); ci++) {
            int[] col = palette[ci % palette.length];
            Scalar edgeCol = new Scalar(col[0], col[1], col[2]);
            MatOfPoint c = contours.get(ci);
            Point[] pts;
            if (epsilon <= 0) {
                pts = c.toArray();
            } else {
                double perim = Imgproc.arcLength(new MatOfPoint2f(c.toArray()), true);
                double eps   = Math.max(epsilon * perim, 2.0);
                MatOfPoint2f approx = new MatOfPoint2f();
                Imgproc.approxPolyDP(new MatOfPoint2f(c.toArray()), approx, eps, true);
                pts = approx.toArray();
                approx.release();
            }
            int n = pts.length;
            if (n == 0) continue;
            for (int i = 0; i < n; i++)
                Imgproc.line(out, pts[i], pts[(i + 1) % n], edgeCol, 1);
            for (Point p : pts)
                Imgproc.circle(out, p, epsilon > 0 ? 5 : 2, new Scalar(255,255,255), -1);
        }
        return out;
    }

    public static List<VectorSignature> buildRefSignatures(Mat refBgr, double epsilonFactor) {
        List<RefCluster> clusters = buildRefClusters(refBgr);
        List<VectorSignature> sigs = new ArrayList<>();
        for (RefCluster rc : clusters) {
            sigs.add(rc.bestSig(epsilonFactor));
        }
        for (RefCluster rc : clusters) rc.release();
        return sigs.isEmpty() ? List.of(buildRefSignature(refBgr, epsilonFactor)) : sigs;
    }

    public static VectorSignature buildRefSignature(Mat refBgr, double epsilonFactor) {
        Mat grey = new Mat(), bin = new Mat();
        Imgproc.cvtColor(refBgr, grey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(grey, bin, 20, 255, Imgproc.THRESH_BINARY);
        grey.release();
        VectorSignature sig = VectorSignature.build(bin, epsilonFactor, Double.NaN);
        bin.release();
        return sig;
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private static String sanitise(String s) { return s.replaceAll("[^A-Za-z0-9_\\-]", "_"); }

    private static String shortName(String v) {
        return v.replace("VECTOR_STRICT", "VM_S")
                .replace("VECTOR_NORMAL", "VM_N")
                .replace("VECTOR_LOOSE",  "VM_L");
    }
}
