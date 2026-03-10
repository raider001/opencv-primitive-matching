package org.example.matchers;

import org.example.analytics.AnalysisResult;
import org.example.colour.SceneColourClusters;
import org.example.matchers.SceneDescriptor;
import org.example.factories.ReferenceId;
import org.example.scene.SceneEntry;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Vector Matcher — detects primitive shapes (lines, polygons, circles) using
 * colour-isolated contour analysis.
 *
 * <h2>Pipeline (per variant)</h2>
 * <ol>
 *   <li>Binarise the reference → build {@link VectorSignature}.</li>
 *   <li>Use a pre-computed {@link SceneDescriptor} (contours grouped by colour cluster).</li>
 *   <li>Score each candidate contour's {@link VectorSignature} against the reference; keep best.</li>
 *   <li>Return one {@link AnalysisResult} per variant (score 0–100, bounding rect).</li>
 * </ol>
 *
 * <h2>Performance</h2>
 * <p>When matching multiple references against the same scene, build a
 * {@link SceneDescriptor} once via {@link SceneDescriptor#build(Mat)} and pass it
 * to {@link #match(ReferenceId, Mat, SceneEntry, SceneDescriptor, Set, Path)}.
 * This avoids re-scanning the scene for every reference.
 *
 * <h2>Variants</h2>
 * <p>3 variants — STRICT / NORMAL / LOOSE epsilon levels.
 */
public final class VectorMatcher {

    private static final int MIN_AREA = 64;

    private VectorMatcher() {}

    // -------------------------------------------------------------------------
    // Public entry points
    // -------------------------------------------------------------------------

    /**
     * Primary entry point — uses the {@link SceneDescriptor} already built into the
     * {@link SceneEntry}.  The descriptor is computed once at scene construction and
     * reused across every reference matched against that scene.
     *
     * <p>If the scene has no descriptor (stub entry with null mat) a temporary one is
     * built and released after matching.
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
        // Fallback for stub entries that have no mat/descriptor
        SceneDescriptor temp = SceneDescriptor.build(scene.sceneMat());
        try {
            return matchWithDescriptor(referenceId, refMat, scene, temp, saveVariants, outputDir);
        } finally {
            temp.release();
        }
    }

    /**
     * Overload for callers that manage their own {@link SceneDescriptor} lifecycle —
     * e.g. when a descriptor is shared across multiple scenes or built externally.
     * The descriptor is not released by this method.
     */
    public static List<AnalysisResult> match(ReferenceId referenceId,
                                             Mat refMat,
                                             SceneEntry scene,
                                             SceneDescriptor descriptor,
                                             Set<String> saveVariants,
                                             Path outputDir) {
        return matchWithDescriptor(referenceId, refMat, scene, descriptor, saveVariants, outputDir);
    }

    private static List<AnalysisResult> matchWithDescriptor(ReferenceId referenceId,
                                                             Mat refMat,
                                                             SceneEntry scene,
                                                             SceneDescriptor descriptor,
                                                             Set<String> saveVariants,
                                                             Path outputDir) {
        List<VectorSignature> refStrict = buildRefSignatures(refMat, VectorVariant.VECTOR_STRICT.epsilonFactor());
        List<VectorSignature> refNormal = buildRefSignatures(refMat, VectorVariant.VECTOR_NORMAL.epsilonFactor());
        List<VectorSignature> refLoose  = buildRefSignatures(refMat, VectorVariant.VECTOR_LOOSE.epsilonFactor());

        // Count ALL clusters (chromatic + achromatic) in the ref image.
        // e.g. CIRCLE_FILLED (white on black) → 2, BICOLOUR_RECT_HALVES → 3
        int refContourCount = SceneDescriptor.countAllClusters(refMat);

        List<AnalysisResult> out = new ArrayList<>(3);
        out.add(runVariant(VectorVariant.VECTOR_STRICT, refStrict, refContourCount, descriptor,
                scene.sceneMat(), referenceId, scene, saveVariants, outputDir));
        out.add(runVariant(VectorVariant.VECTOR_NORMAL, refNormal, refContourCount, descriptor,
                scene.sceneMat(), referenceId, scene, saveVariants, outputDir));
        out.add(runVariant(VectorVariant.VECTOR_LOOSE,  refLoose,  refContourCount, descriptor,
                scene.sceneMat(), referenceId, scene, saveVariants, outputDir));
        return out;
    }

    // -------------------------------------------------------------------------
    // Single variant
    // -------------------------------------------------------------------------

    static AnalysisResult runVariant(VectorVariant variant,
                                      List<VectorSignature> refSigs,
                                      int refContourCount,
                                      SceneDescriptor descriptor,
                                      Mat sceneForAnnotation,
                                      ReferenceId referenceId,
                                      SceneEntry scene,
                                      Set<String> saveVariants,
                                      Path outputDir) {
        long t0 = System.currentTimeMillis();
        try {
            int refClusterCount = refSigs.size();
            List<SceneDescriptor.ClusterContours> clusters = descriptor.clusters();
            double eps = variant.epsilonFactor();

            // ── Flatten all scene contours (excluding envelope entries) ──
            record CEntry(MatOfPoint c, int clusterIdx, VectorSignature sig) {}
            List<CEntry> candidates = new ArrayList<>();
            for (int ci = 0; ci < clusters.size(); ci++) {
                SceneDescriptor.ClusterContours cc = clusters.get(ci);
                if (cc.envelope) continue; // envelope used only for penalty, not matching
                for (MatOfPoint c : cc.contours) {
                    VectorSignature sig = VectorSignature.buildFromContour(c, eps, descriptor.sceneArea);
                    candidates.add(new CEntry(c, ci, sig));
                }
            }

            // ── Score every possible assignment of scene contours to ref sigs ──
            //
            // Algorithm (greedy best-assignment):
            //   For each scene contour c_i, compute sim(c_i, refSig_j) for all j.
            //   Then greedily assign: pick the highest sim pair (c_i, refSig_j)
            //   where c_i comes from a cluster not yet used and refSig_j not yet
            //   matched. Repeat until no more matches.
            //
            // Score = (sum of matched sims / refClusterCount)
            //       × (matched / refClusterCount)          ← coverage fraction
            //
            // The coverage fraction means:
            //   - All N ref sigs matched          → multiplier = 1.0  (no degradation)
            //   - Only k of N matched             → multiplier = k/N  (degrades linearly)
            //   - 1 of N matched (single contour) → multiplier = 1/N
            //
            // This is fully algorithmic — works for any N without hard-coded loops.

            double bestScore      = 0.0;
            Rect   bestBbox       = null;
            int    bestClusterIdx = -1;

            // Build sim matrix: candidates × refSigs
            int nc = candidates.size();
            int nr = refSigs.size();
            double[][] simMatrix = new double[nc][nr];
            for (int i = 0; i < nc; i++)
                for (int j = 0; j < nr; j++)
                    simMatrix[i][j] = refSigs.get(j).similarity(candidates.get(i).sig());

            // Try greedy assignment starting from every candidate as the "anchor"
            // to avoid getting stuck in a single local optimum.
            for (int anchor = 0; anchor < nc; anchor++) {
                boolean[] usedCandidate = new boolean[nc];
                boolean[] usedRefSig    = new boolean[nr];
                List<Integer> matched   = new ArrayList<>();
                Rect combinedBbox       = null;

                // Build a flat list of all (sim, candidateIdx, refSigIdx) triples,
                // sorted descending by sim — greedy picks highest available pair
                List<int[]> triples = new ArrayList<>(nc * nr);
                for (int i = 0; i < nc; i++)
                    for (int j = 0; j < nr; j++)
                        triples.add(new int[]{i, j, (int)(simMatrix[i][j] * 1_000_000)});
                triples.sort((a, b) -> b[2] - a[2]);

                // Force the anchor candidate in as the first assignment
                int anchorBestJ = 0;
                for (int j = 1; j < nr; j++)
                    if (simMatrix[anchor][j] > simMatrix[anchor][anchorBestJ]) anchorBestJ = j;
                usedCandidate[anchor] = true;
                usedRefSig[anchorBestJ] = true;
                matched.add(anchor);
                combinedBbox = Imgproc.boundingRect(candidates.get(anchor).c());

                // Greedy fill remaining ref sigs from different clusters
                for (int[] triple : triples) {
                    int ci = triple[0], rj = triple[1];
                    if (usedCandidate[ci] || usedRefSig[rj]) continue;
                    // Must come from a cluster not already represented in the match set
                    int clusterOfCi = candidates.get(ci).clusterIdx();
                    boolean clusterAlreadyUsed = false;
                    for (int m : matched)
                        if (candidates.get(m).clusterIdx() == clusterOfCi) { clusterAlreadyUsed = true; break; }
                    if (clusterAlreadyUsed) continue;
                    usedCandidate[ci] = true;
                    usedRefSig[rj]    = true;
                    matched.add(ci);
                    combinedBbox = unionRect(combinedBbox, Imgproc.boundingRect(candidates.get(ci).c()));
                    if (matched.size() == nr) break; // all ref sigs covered
                }

                // Score: average sim of matched pairs × coverage fraction
                double sumSim = 0.0;
                for (int m : matched) {
                    // find which refSig this candidate was matched to
                    double best = 0;
                    for (int j = 0; j < nr; j++)
                        if (simMatrix[m][j] > best) best = simMatrix[m][j];
                    sumSim += best;
                }
                double coverage = (double) matched.size() / nr;
                double rawScore = (sumSim / nr) * coverage;

                // Intra-cluster noise penalty: if a single cluster has many
                // significant contours beyond what the ref expects, penalise
                int clusterOfAnchor = candidates.get(anchor).clusterIdx();
                SceneDescriptor.ClusterContours anchorCluster = clusters.get(clusterOfAnchor);
                double maxA = anchorCluster.contours.stream()
                        .mapToDouble(c -> { Rect r = Imgproc.boundingRect(c); return (double)r.width*r.height; })
                        .max().orElse(1);
                long sig = anchorCluster.contours.stream()
                        .filter(c -> { Rect r = Imgproc.boundingRect(c); return (double)r.width*r.height >= maxA*0.20; })
                        .count();
                int excess = Math.max(0, (int)sig - refClusterCount);
                double noisePenalty = excess > 0 ? 1.0 / (Math.log(excess + 2) / Math.log(2)) : 1.0;
                rawScore *= noisePenalty;

                if (rawScore > bestScore) {
                    bestScore      = rawScore;
                    bestBbox       = combinedBbox;
                    bestClusterIdx = clusterOfAnchor;
                }
            }

            // ── Contour-count penalty on the winning bbox ─────────────────
            // Compare total structural contours in the winning region against
            // the ref's expected count (from its union-of-clusters mask).
            if (bestBbox != null) {
                bestScore = applyClusterCountPenalty(
                        bestScore, bestBbox, refContourCount, sceneForAnnotation);
            }

            double scorePercent = bestScore * 100.0;
            long   elapsed      = System.currentTimeMillis() - t0;

            Path savedPath = null;
            if (saveVariants.contains(variant.variantName())) {
                savedPath = writeAnnotated(sceneForAnnotation, bestBbox, variant.variantName(),
                        scorePercent, referenceId, scene, outputDir);
            }

            return new AnalysisResult(variant.variantName(), referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    scorePercent, bestBbox, elapsed, 0L,
                    (int) descriptor.sceneArea,
                    savedPath, false, null);

        } catch (Exception e) {
            return AnalysisResult.error(variant.variantName(), referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    System.currentTimeMillis() - t0,
                    (int) descriptor.sceneArea,
                    e.getMessage());
        }
    }

    /** Returns the union (enclosing) rectangle of two rects. */
    private static Rect unionRect(Rect a, Rect b) {
        int x = Math.min(a.x, b.x);
        int y = Math.min(a.y, b.y);
        int x2 = Math.max(a.x + a.width,  b.x + b.width);
        int y2 = Math.max(a.y + a.height, b.y + b.height);
        return new Rect(x, y, x2 - x, y2 - y);
    }


    /**
     * Penalises a match score when the total cluster count inside the winning
     * bbox does not match the reference's expected cluster count.
     *
     * <p>Both ref and scene are counted the same way: ALL significant clusters
     * (chromatic + achromatic) from {@link SceneColourClusters#extract}.
     * This includes the background, so a plain white-on-black shape = 2 clusters.
     *
     * <p>The penalty is <b>exponential</b> based on the absolute difference:
     * <pre>
     *   multiplier = exp(-k * |sceneClusters - refClusters|)
     * </pre>
     * where k controls how steeply the score degrades per unit of mismatch.
     * This means:
     * <ul>
     *   <li>mismatch=0 → multiplier=1.0  (no penalty)</li>
     *   <li>mismatch=1 → multiplier≈0.74 (k=0.3)</li>
     *   <li>mismatch=2 → multiplier≈0.55</li>
     *   <li>mismatch=4 → multiplier≈0.30</li>
     * </ul>
     *
     * <p>Works for any N clusters from 1 to 100 without hard-coding.
     */
    private static double applyClusterCountPenalty(
            double score, Rect bbox, int refClusterCount, Mat scene) {

        // Single-colour refs (≤2 clusters: shape + background) — no penalty.
        // A tight bbox may only capture the foreground, so scene count of 1
        // vs ref count of 2 is a false alarm, not a structural mismatch.
        // Only penalise genuine multi-colour refs (3+ clusters).
        if (refClusterCount <= 2) return score;

        int x1 = Math.max(0, bbox.x);
        int y1 = Math.max(0, bbox.y);
        int x2 = Math.min(scene.cols(), bbox.x + bbox.width);
        int y2 = Math.min(scene.rows(), bbox.y + bbox.height);
        if (x2 <= x1 || y2 <= y1) return score;

        Mat roi = scene.submat(y1, y2, x1, x2);
        int sceneClusterCount = SceneDescriptor.countAllClusters(roi);
        roi.release();

        if (sceneClusterCount == 0 || sceneClusterCount >= refClusterCount) return score;

        // Scene has FEWER clusters than ref → missing colour regions / structure.
        // Extra clusters (noisy background) are NOT penalised — the shape is
        // still present even in a busy scene.
        // Exponential degradation per missing cluster, works for any N:
        //   missing=1 of ref=2 → exp(-0.3*1) ≈ 0.74
        //   missing=2 of ref=3 → exp(-0.3*2) ≈ 0.55
        //   missing=3 of ref=4 → exp(-0.3*3) ≈ 0.41
        int missing = refClusterCount - sceneClusterCount;
        double multiplier = Math.exp(-0.3 * missing);
        return score * multiplier;
    }


    // -------------------------------------------------------------------------
    // Contour extraction
    // -------------------------------------------------------------------------

    /**
     * Extracts contours from a greyscale/binary or BGR image using
     * threshold-only binarisation.  Used for visualisation and legacy callers.
     */
    public static List<MatOfPoint> extractContoursFromBinary(Mat maskedBgr) {
        Mat grey = new Mat();
        Mat bin  = new Mat();
        Imgproc.cvtColor(maskedBgr, grey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(grey, bin, 20, 255, Imgproc.THRESH_BINARY);
        grey.release();
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(bin, contours, hierarchy,
                Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        hierarchy.release();
        bin.release();
        contours.removeIf(c -> Imgproc.contourArea(c) < SceneColourClusters.MIN_CONTOUR_AREA);
        return contours;
    }

    /**
     * Returns the raw binary edge map (threshold OR Canny) for visualisation only.
     * Not used in matching — kept for the HTML report "Edges (ref)" step.
     */
    public static Mat extractBinaryRaw(Mat bgrScene) {
        Mat grey = new Mat();
        Mat bin  = new Mat();
        Mat edge = new Mat();
        Imgproc.cvtColor(bgrScene, grey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(grey, bin, 20, 255, Imgproc.THRESH_BINARY);
        Imgproc.Canny(grey, edge, 40.0, 120.0);
        Core.bitwise_or(bin, edge, bin);
        grey.release();
        edge.release();
        return bin;
    }

    /**
     * Renders a contour graph visualisation onto a black canvas.
     *
     * @param size      output canvas size
     * @param binary    binary edge map as grey underlay (may be null)
     * @param contours  contours to visualise
     * @param epsilon   0 = all raw points; > 0 (e.g. 0.02) = approxPolyDP vertices only
     */
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
            {255,80,255},{255,255,80},{80,160,255},{255,80,160},
            {160,255,80},{255,160,80}
        };

        for (int ci = 0; ci < contours.size(); ci++) {
            int[]  col     = palette[ci % palette.length];
            Scalar edgeCol = new Scalar(col[0], col[1], col[2]);
            Scalar vertCol = new Scalar(255, 255, 255);
            MatOfPoint c   = contours.get(ci);
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

            int dotR = (epsilon > 0) ? 5 : 2;
            for (int i = 0; i < n; i++)
                Imgproc.line(out, pts[i], pts[(i + 1) % n], edgeCol, 1);
            for (Point p : pts) {
                Imgproc.circle(out, p, dotR, vertCol, -1);
                Imgproc.circle(out, p, dotR, edgeCol, 1);
            }
        }
        return out;
    }

    // -------------------------------------------------------------------------
    // Reference signature builder
    // -------------------------------------------------------------------------

    /**
     * Builds one {@link VectorSignature} per distinct colour cluster in the reference
     * image.  For single-colour refs this returns a one-element list (identical
     * behaviour to the old greyscale path).  For multi-colour refs (BICOLOUR_*,
     * TRICOLOUR_*) it returns one signature per hue region so each colour's
     * geometry is described independently.
     *
     * <p>Falls back to the greyscale threshold path if no colour clusters are found.
     */
    public static List<VectorSignature> buildRefSignatures(Mat refBgr, double epsilonFactor) {
        List<VectorSignature> sigs = new ArrayList<>();

        List<SceneColourClusters.Cluster> clusters = SceneColourClusters.extract(refBgr);
        List<Mat> chromaticMasks = new ArrayList<>();

        for (SceneColourClusters.Cluster cluster : clusters) {
            if (cluster.achromatic) { cluster.release(); continue; }
            Mat dilated = new Mat();
            Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
            Imgproc.dilate(cluster.mask, dilated, kernel);
            kernel.release();
            chromaticMasks.add(cluster.mask.clone());
            List<MatOfPoint> contours = SceneDescriptor.contoursFromMask(dilated);
            dilated.release();
            cluster.release();
            for (MatOfPoint c : contours)
                sigs.add(VectorSignature.buildFromContour(c, epsilonFactor, Double.NaN));
        }
        for (Mat m : chromaticMasks) m.release();

        // Fallback: if colour extraction found nothing useful, use greyscale
        if (sigs.isEmpty()) sigs.add(buildRefSignature(refBgr, epsilonFactor));
        return sigs;
    }


    /**
     * Builds a single {@link VectorSignature} from the greyscale threshold of the
     * reference image.  Used as a fallback and by callers that need a single
     * canonical signature (e.g. report visualisation, {@code allScoredBboxes}).
     */
    public static VectorSignature buildRefSignature(Mat refBgr, double epsilonFactor) {
        Mat grey = new Mat();
        Mat bin  = new Mat();
        Imgproc.cvtColor(refBgr, grey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(grey, bin, 20, 255, Imgproc.THRESH_BINARY);
        grey.release();
        VectorSignature sig = VectorSignature.build(bin, epsilonFactor, Double.NaN);
        bin.release();
        return sig;
    }

    // -------------------------------------------------------------------------
    // Annotation writer
    // -------------------------------------------------------------------------

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

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private static String sanitise(String s) { return s.replaceAll("[^A-Za-z0-9_\\-]", "_"); }

    private static String shortName(String v) {
        return v.replace("VECTOR_STRICT", "VM_S")
                .replace("VECTOR_NORMAL", "VM_N")
                .replace("VECTOR_LOOSE",  "VM_L");
    }
}
