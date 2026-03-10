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
        VectorSignature refStrict = buildRefSignature(refMat, VectorVariant.VECTOR_STRICT.epsilonFactor());
        VectorSignature refNormal = buildRefSignature(refMat, VectorVariant.VECTOR_NORMAL.epsilonFactor());
        VectorSignature refLoose  = buildRefSignature(refMat, VectorVariant.VECTOR_LOOSE.epsilonFactor());

        List<AnalysisResult> out = new ArrayList<>(3);
        out.add(runVariant(VectorVariant.VECTOR_STRICT, refStrict, descriptor,
                scene.sceneMat(), referenceId, scene, saveVariants, outputDir));
        out.add(runVariant(VectorVariant.VECTOR_NORMAL, refNormal, descriptor,
                scene.sceneMat(), referenceId, scene, saveVariants, outputDir));
        out.add(runVariant(VectorVariant.VECTOR_LOOSE,  refLoose,  descriptor,
                scene.sceneMat(), referenceId, scene, saveVariants, outputDir));
        return out;
    }

    // -------------------------------------------------------------------------
    // Single variant
    // -------------------------------------------------------------------------

    static AnalysisResult runVariant(VectorVariant variant,
                                      VectorSignature refSig,
                                      SceneDescriptor descriptor,
                                      Mat sceneForAnnotation,
                                      ReferenceId referenceId,
                                      SceneEntry scene,
                                      Set<String> saveVariants,
                                      Path outputDir) {
        long t0 = System.currentTimeMillis();
        try {
            double bestScore      = 0.0;
            Rect   bestBbox       = null;
            int    bestClusterIdx = -1;

            List<SceneDescriptor.ClusterContours> clusters = descriptor.clusters();
            for (int ci = 0; ci < clusters.size(); ci++) {
                SceneDescriptor.ClusterContours cc = clusters.get(ci);
                // Pre-compute significant competitor count for this cluster
                // (contours whose bbox area >= 20% of the largest contour in the cluster)
                double maxArea = 0;
                for (MatOfPoint cx : cc.contours) {
                    Rect rx = Imgproc.boundingRect(cx);
                    maxArea = Math.max(maxArea, (double) rx.width * rx.height);
                }
                int significant = 0;
                for (MatOfPoint cx : cc.contours) {
                    Rect rx = Imgproc.boundingRect(cx);
                    if ((double) rx.width * rx.height >= maxArea * 0.20) significant++;
                }
                double clusterPenalty = significant > 1
                        ? 1.0 / (Math.log(significant + 1) / Math.log(2)) : 1.0;

                for (MatOfPoint c : cc.contours) {
                    VectorSignature sceneSig = VectorSignature.buildFromContour(
                            c, variant.epsilonFactor(), descriptor.sceneArea);
                    double sim = refSig.similarity(sceneSig) * clusterPenalty;
                    if (sim > bestScore) {
                        bestScore      = sim;
                        bestBbox       = Imgproc.boundingRect(c);
                        bestClusterIdx = ci;
                    }
                }
            }

            // ── Multi-cluster penalty ─────────────────────────────────────
            // If the best bbox region is significantly covered by pixels from
            // other clusters, the "shape" spans multiple colours — it is likely
            // background noise, not a real single-colour shape.
            if (bestBbox != null && bestClusterIdx >= 0 && clusters.size() > 1) {
                bestScore = applyMultiClusterPenalty(
                        bestScore, bestBbox, bestClusterIdx, clusters, sceneForAnnotation);
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

    /**
     * Penalises a match score when the candidate bbox region contains significant
     * pixel coverage from clusters other than the one the match came from.
     *
     * <p>Logic:
     * <ol>
     *   <li>Count pixels in the bbox belonging to the winning cluster (own pixels).</li>
     *   <li>Count pixels belonging to every other cluster in the same bbox (foreign pixels).</li>
     *   <li>Compute {@code foreignRatio = foreignPx / (ownPx + foreignPx)}.</li>
     *   <li>Apply penalty: {@code score *= (1 - foreignRatio * PENALTY_WEIGHT)}.</li>
     * </ol>
     *
     * <p>A foreignRatio of 0 means the shape is cleanly one colour — no penalty.
     * A foreignRatio of 0.5 means half the bbox is other colours — substantial penalty.
     */
    private static double applyMultiClusterPenalty(
            double score, Rect bbox, int winnerIdx,
            List<SceneDescriptor.ClusterContours> clusters,
            Mat scene) {

        // Re-extract cluster masks just for the bbox ROI.
        // This is the same SceneColourClusters.extract() that built the descriptor,
        // but operating only on the small submat — lightweight.
        int x1 = Math.max(0, bbox.x);
        int y1 = Math.max(0, bbox.y);
        int x2 = Math.min(scene.cols(), bbox.x + bbox.width);
        int y2 = Math.min(scene.rows(), bbox.y + bbox.height);
        if (x2 <= x1 || y2 <= y1) return score;

        Mat roi = scene.submat(y1, y2, x1, x2);
        List<org.example.colour.SceneColourClusters.Cluster> roiClusters =
                org.example.colour.SceneColourClusters.extract(roi);
        roi.release();

        // Match roi clusters back to descriptor clusters by hue proximity.
        // The winning descriptor cluster index maps to the roi cluster whose
        // hue is closest; its pixel count = ownPx. All others = foreignPx.
        SceneDescriptor.ClusterContours winner = clusters.get(winnerIdx);

        long ownPx = 0, foreignPx = 0;
        for (org.example.colour.SceneColourClusters.Cluster rc : roiClusters) {
            long px = Core.countNonZero(rc.mask);
            boolean isOwn;
            if (winner.achromatic) {
                isOwn = rc.achromatic;
            } else {
                if (rc.achromatic) {
                    isOwn = false;
                } else {
                    double d = Math.abs(winner.hue - rc.hue);
                    d = Math.min(d, 180 - d);
                    isOwn = d < SceneColourClusters.HUE_TOLERANCE * 2;
                }
            }
            if (isOwn) ownPx += px;
            else        foreignPx += px;
            rc.release();
        }

        long total = ownPx + foreignPx;
        if (total == 0) return score;

        double foreignRatio = (double) foreignPx / total;
        double penalty = foreignRatio * 0.80;
        return score * (1.0 - penalty);
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

    public static VectorSignature buildRefSignature(Mat refBgr, double epsilonFactor) {
        Mat grey = new Mat();
        Mat bin  = new Mat();
        Imgproc.cvtColor(refBgr, grey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(grey, bin, 20, 255, Imgproc.THRESH_BINARY);
        grey.release();
        // Pass NaN for imageArea so normalisedArea = NaN on the reference.
        // The area-ratio gate in VectorSignature.similarity() only fires when
        // BOTH sides have a finite normalisedArea — this prevents false-gating
        // when the reference is drawn at a different scale than the scene.
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
