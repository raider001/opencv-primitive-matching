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
 * <h2>Why it is scale and rotation invariant</h2>
 * <p>Shapes are encoded as {@link VectorSignature} objects whose fields
 * (circularity, vertex count, angle histogram, segment descriptor) are all
 * computed from normalised geometry — absolute position and scale are discarded.
 *
 * <h2>Pipeline (per variant)</h2>
 * <ol>
 *   <li>Binarise the reference → build {@link VectorSignature}.</li>
 *   <li>Decompose the scene into per-colour clusters via {@link SceneColourClusters}.</li>
 *   <li>For each cluster: threshold → {@code findContours} (CHAIN_APPROX_SIMPLE).</li>
 *   <li>Score each candidate contour's {@link VectorSignature} against the reference; keep best.</li>
 *   <li>Return one {@link AnalysisResult} per variant (score 0–100, bounding rect).</li>
 * </ol>
 *
 * <h2>Variants</h2>
 * <p>3 variants — one per approximation epsilon level (STRICT/NORMAL/LOOSE).
 * Colour isolation is applied automatically to all variants via {@link SceneColourClusters};
 * no separate CF variants are needed.
 */
public final class VectorMatcher {

    private static final int MIN_AREA = 64;

    private VectorMatcher() {}

    // -------------------------------------------------------------------------
    // Public entry point
    // -------------------------------------------------------------------------

    public static List<AnalysisResult> match(ReferenceId referenceId,
                                             Mat refMat,
                                             SceneEntry scene,
                                             Set<String> saveVariants,
                                             Path outputDir) {
        List<AnalysisResult> out = new ArrayList<>(3);
        Mat sceneMat = scene.sceneMat();

        // Build reference signatures once per epsilon level
        VectorSignature refStrict = buildRefSignature(refMat, VectorVariant.VECTOR_STRICT.epsilonFactor());
        VectorSignature refNormal = buildRefSignature(refMat, VectorVariant.VECTOR_NORMAL.epsilonFactor());
        VectorSignature refLoose  = buildRefSignature(refMat, VectorVariant.VECTOR_LOOSE.epsilonFactor());

        // Extract colour clusters once — shared across all 3 variants
        List<SceneColourClusters.Cluster> clusters = SceneColourClusters.extract(sceneMat);
        List<List<MatOfPoint>> contoursPerCluster  = new ArrayList<>(clusters.size());
        for (SceneColourClusters.Cluster cluster : clusters) {
            Mat masked = SceneColourClusters.applyMask(sceneMat, cluster);
            contoursPerCluster.add(extractContoursFromBinary(masked));
            masked.release();
            cluster.release();
        }

        out.add(runVariant(VectorVariant.VECTOR_STRICT, refStrict, contoursPerCluster,
                sceneMat, referenceId, scene, saveVariants, outputDir));
        out.add(runVariant(VectorVariant.VECTOR_NORMAL, refNormal, contoursPerCluster,
                sceneMat, referenceId, scene, saveVariants, outputDir));
        out.add(runVariant(VectorVariant.VECTOR_LOOSE,  refLoose,  contoursPerCluster,
                sceneMat, referenceId, scene, saveVariants, outputDir));

        return out;
    }

    // -------------------------------------------------------------------------
    // Single variant
    // -------------------------------------------------------------------------

    static AnalysisResult runVariant(VectorVariant variant,
                                      VectorSignature refSig,
                                      List<List<MatOfPoint>> contoursPerCluster,
                                      Mat sceneForAnnotation,
                                      ReferenceId referenceId,
                                      SceneEntry scene,
                                      Set<String> saveVariants,
                                      Path outputDir) {
        long t0 = System.currentTimeMillis();
        try {
            double sceneArea = (double) sceneForAnnotation.rows() * sceneForAnnotation.cols();
            double bestScore = 0.0;
            Rect   bestBbox  = null;

            for (List<MatOfPoint> contours : contoursPerCluster) {
                for (MatOfPoint c : contours) {
                    VectorSignature sceneSig = VectorSignature.buildFromContour(
                            c, variant.epsilonFactor(), sceneArea);
                    double sim = refSig.similarity(sceneSig);
                    if (sim > bestScore) {
                        bestScore = sim;
                        bestBbox  = Imgproc.boundingRect(c);
                    }
                }
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
                    scene.sceneMat().cols() * scene.sceneMat().rows(),
                    savedPath, false, null);

        } catch (Exception e) {
            return AnalysisResult.error(variant.variantName(), referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    System.currentTimeMillis() - t0,
                    scene.sceneMat().cols() * scene.sceneMat().rows(),
                    e.getMessage());
        }
    }

    // -------------------------------------------------------------------------
    // Contour extraction
    // -------------------------------------------------------------------------

    /**
     * Extracts contours from a colour-isolated masked BGR image using
     * threshold-only binarisation.  Because the input has only one colour cluster,
     * the threshold produces clean filled shapes.  {@code CHAIN_APPROX_SIMPLE} on
     * a filled shape gives exact corner points with no aliased stepping.
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

        contours.removeIf(c -> Imgproc.contourArea(c) < MIN_AREA);
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

            boolean isApprox = (epsilon > 0);
            int dotR  = isApprox ? 5 : 2;
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
        double refArea = (double) refBgr.rows() * refBgr.cols();
        VectorSignature sig = VectorSignature.build(bin, epsilonFactor, refArea);
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
