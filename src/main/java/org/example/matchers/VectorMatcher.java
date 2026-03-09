package org.example.matchers;

import org.example.analytics.AnalysisResult;
import org.example.colour.ColourPreFilter;
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
 * contour-based structural analysis.
 *
 * <h2>Why it is scale and rotation invariant</h2>
 * <p>Unlike pixel-based matchers, this technique encodes shapes as
 * {@link VectorSignature} objects whose fields (circularity, vertex count,
 * angle histogram) are all computed from normalised geometry.  Absolute
 * position and scale are discarded at descriptor-build time, and the angle
 * histogram uses absolute angle magnitudes so rotation changes nothing.
 *
 * <h2>Pipeline (per variant)</h2>
 * <ol>
 *   <li>Binarise the reference at 128×128 → build {@link VectorSignature}.</li>
 *   <li>Apply CF pre-filter to the scene if the variant requires it.</li>
 *   <li>Convert scene to greyscale, threshold, Canny edges.</li>
 *   <li>Extract contours; discard noise (&lt; 64 px²).</li>
 *   <li>Score each candidate contour's {@link VectorSignature} against the
 *       reference signature; keep the best match.</li>
 *   <li>Return one {@link AnalysisResult} per variant (score 0–100, bounding rect).</li>
 * </ol>
 *
 * <h2>Variants</h2>
 * <p>9 variants — 3 approximation epsilon levels (STRICT/NORMAL/LOOSE)
 * × 3 CF modes (NONE/LOOSE/TIGHT).  See {@link VectorVariant}.
 */
public final class VectorMatcher {

    private static final double CANNY_LO  = 40.0;
    private static final double CANNY_HI  = 120.0;
    private static final int    MIN_AREA  = 64;

    private VectorMatcher() {}

    // -------------------------------------------------------------------------
    // Public entry point
    // -------------------------------------------------------------------------

    public static List<AnalysisResult> match(ReferenceId referenceId,
                                             Mat refMat,
                                             SceneEntry scene,
                                             Set<String> saveVariants,
                                             Path outputDir) {
        List<AnalysisResult> out = new ArrayList<>(9);
        Mat sceneMat = scene.sceneMat();

        // Build reference signatures once per epsilon level (CF does not change ref)
        VectorSignature refStrict = buildRefSignature(refMat, VectorVariant.VECTOR_STRICT.epsilonFactor());
        VectorSignature refNormal = buildRefSignature(refMat, VectorVariant.VECTOR_NORMAL.epsilonFactor());
        VectorSignature refLoose  = buildRefSignature(refMat, VectorVariant.VECTOR_LOOSE.epsilonFactor());

        // Pre-compute CF scene mats (deferred to avoid building them when not needed)
        Mat sceneLoose = null, sceneTight = null;
        long cfLMs = 0, cfTMs = 0;

        long t0 = System.currentTimeMillis();
        sceneLoose = ColourPreFilter.applyMaskedBgrToScene(sceneMat, referenceId, ColourPreFilter.LOOSE);
        cfLMs = System.currentTimeMillis() - t0;

        t0 = System.currentTimeMillis();
        sceneTight = ColourPreFilter.applyMaskedBgrToScene(sceneMat, referenceId, ColourPreFilter.TIGHT);
        cfTMs = System.currentTimeMillis() - t0;

        // Run all 9 variants
        for (VectorVariant v : VectorVariant.values()) {
            VectorSignature refSig = pickRefSig(v, refStrict, refNormal, refLoose);
            Mat             scnMat = pickSceneMat(v, sceneMat, sceneLoose, sceneTight);
            long            cfMs   = pickCfMs(v, cfLMs, cfTMs);

            out.add(runVariant(v, refSig, scnMat, sceneMat, cfMs,
                    referenceId, scene, saveVariants, outputDir));
        }

        if (sceneLoose != null) sceneLoose.release();
        if (sceneTight != null) sceneTight.release();

        return out;
    }

    // -------------------------------------------------------------------------
    // Single variant
    // -------------------------------------------------------------------------

    static AnalysisResult runVariant(VectorVariant variant,
                                      VectorSignature refSig,
                                      Mat sceneForExtraction,
                                      Mat sceneForAnnotation,
                                      long preFilterMs,
                                      ReferenceId referenceId,
                                      SceneEntry scene,
                                      Set<String> saveVariants,
                                      Path outputDir) {
        long t0 = System.currentTimeMillis();
        try {
            double sceneArea = (double) sceneForExtraction.rows() * sceneForExtraction.cols();

            double bestScore = 0.0;
            Rect   bestBbox  = null;

            // ── Colour-isolated contour extraction ───────────────────────
            // Decompose the scene into per-colour clusters and extract
            // contours from each independently.  This keeps each contour
            // clean — noise of a different colour never merges with the
            // target shape's contour.
            List<SceneColourClusters.Cluster> clusters =
                    SceneColourClusters.extract(sceneForExtraction);

            for (SceneColourClusters.Cluster cluster : clusters) {
                Mat masked = SceneColourClusters.applyMask(sceneForExtraction, cluster);
                List<MatOfPoint> contours = extractContoursFromBinary(masked);
                masked.release();
                cluster.release();

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
            long elapsed = System.currentTimeMillis() - t0;

            Path savedPath = null;
            if (saveVariants.contains(variant.variantName())) {
                savedPath = writeAnnotated(sceneForAnnotation, bestBbox, variant.variantName(),
                        scorePercent, referenceId, scene, outputDir);
            }

            return new AnalysisResult(variant.variantName(), referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    scorePercent, bestBbox, elapsed, preFilterMs,
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
     * Converts a BGR scene into a list of candidate contours.
     * Contours smaller than {@code minArea} pixels are discarded.
     */
    public static List<MatOfPoint> extractContours(Mat bgrScene, double minArea) {
        Mat grey = new Mat();
        Mat bin  = new Mat();
        Mat edge = new Mat();

        Imgproc.cvtColor(bgrScene, grey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(grey, bin, 20, 255, Imgproc.THRESH_BINARY);
        Imgproc.Canny(grey, edge, CANNY_LO, CANNY_HI);
        Core.bitwise_or(bin, edge, bin);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(bin, contours, hierarchy,
                Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        hierarchy.release();
        grey.release();
        bin.release();
        edge.release();

        contours.removeIf(c -> Imgproc.contourArea(c) < minArea);
        return contours;
    }

    /** Extracts contours using the default minimum area (64 px²). */
    public static List<MatOfPoint> extractContours(Mat bgrScene) {
        return extractContours(bgrScene, MIN_AREA);
    }

    /**
     * Extracts contours from a colour-isolated (masked) BGR image using
     * threshold-only binarisation — no Canny.
     *
     * <p>Because the input is already colour-isolated (only one colour cluster
     * is present), the threshold produces a clean binary mask with true filled
     * shapes. {@code CHAIN_APPROX_SIMPLE} on a filled shape gives exact corner
     * points (4 for a rect, 3 for a triangle) with no aliased intermediate points.
     * This is the primary contour extraction path for the colour-isolated matcher.
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
     * Returns the raw binary edge map (threshold OR Canny) before any noise reduction.
     * Used for the "All Points &amp; Connections" visualisation step in the HTML report.
     */
    public static Mat extractBinaryRaw(Mat bgrScene) {
        Mat grey = new Mat();
        Mat bin  = new Mat();
        Mat edge = new Mat();
        Imgproc.cvtColor(bgrScene, grey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(grey, bin, 20, 255, Imgproc.THRESH_BINARY);
        Imgproc.Canny(grey, edge, CANNY_LO, CANNY_HI);
        Core.bitwise_or(bin, edge, bin);
        grey.release();
        edge.release();
        return bin; // caller must release
    }

    /**
     * Returns the binary edge map after morphological opening — erode then dilate
     * with a 3×3 kernel.  This breaks thin 1–2 px noise bridges that connect
     * background edges to the target shape's contour, without affecting the thick
     * edges of real filled shapes.
     * Used for the "Reduced Points &amp; Connections" visualisation step.
     */
    public static Mat extractBinaryReduced(Mat bgrScene) {
        Mat raw    = extractBinaryRaw(bgrScene);
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Mat opened = new Mat();
        Imgproc.morphologyEx(raw, opened, Imgproc.MORPH_OPEN, kernel);
        raw.release();
        kernel.release();
        return opened; // caller must release
    }

    /**
     * Renders a contour graph visualisation onto a black canvas.
     *
     * <p>Layer 1 (dark grey): the raw binary edge map as underlay.
     * <p>Layer 2 (coloured, per-contour): connecting edges as thick lines.
     * <p>Layer 3 (white dots): vertices as filled circles.
     *
     * @param size      output canvas size
     * @param binary    binary edge map to show as grey underlay (single-channel, may be null)
     * @param contours  contours to visualise
     * @param epsilon   approxPolyDP factor. Pass {@code 0} to show ALL raw contour
     *                  points without any approximation (the "All Points" view).
     *                  Pass a non-zero value (e.g. 0.02) to show only the approximated
     *                  polygon vertices (the "Reduced Points" view).
     */
    public static Mat drawContourGraph(Size size, Mat binary,
                                        List<MatOfPoint> contours, double epsilon) {
        Mat out = Mat.zeros((int) size.height, (int) size.width, CvType.CV_8UC3);

        int dotRaw     = 2;
        int dotApprox  = 5;
        int lineApprox = 1;

        // Layer 1 — raw edge map as dark grey underlay
        if (binary != null && !binary.empty()) {
            Mat grey3 = new Mat();
            Imgproc.cvtColor(binary, grey3, Imgproc.COLOR_GRAY2BGR);
            Core.multiply(grey3, new Scalar(0.4, 0.4, 0.4), grey3);
            Core.add(out, grey3, out);
            grey3.release();
        }

        // Colour palette (BGR)
        int[][] palette = {
            {80,80,255},{80,255,80},{255,80,80},{80,255,255},
            {255,80,255},{255,255,80},{80,160,255},{255,80,160},
            {160,255,80},{255,160,80}
        };

        for (int ci = 0; ci < contours.size(); ci++) {
            int[]  col     = palette[ci % palette.length];
            Scalar edgeCol = new Scalar(col[0], col[1], col[2]);
            Scalar vertCol = new Scalar(255, 255, 255);

            MatOfPoint c = contours.get(ci);
            Point[] pts;

            if (epsilon <= 0) {
                // ALL raw contour points — no approximation
                pts = c.toArray();
            } else {
                // Approximated polygon
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
            int lineW = isApprox ? lineApprox : 1;
            int dotR  = isApprox ? dotApprox  : dotRaw;

            // Edges
            for (int i = 0; i < n; i++) {
                Imgproc.line(out, pts[i], pts[(i + 1) % n], edgeCol, lineW);
            }
            // Vertices
            for (Point p : pts) {
                Imgproc.circle(out, p, dotR, vertCol, -1);
                Imgproc.circle(out, p, dotR, edgeCol, lineW);
            }
        }
        return out;
    }

    /**
     * Renders a single contour into a binary mask the same size as the scene,
     * so {@link VectorSignature#build} can analyse it in isolation.
     */
    static Mat contourToBinary(MatOfPoint contour, Size sceneSize) {
        Mat m = Mat.zeros((int) sceneSize.height, (int) sceneSize.width, CvType.CV_8UC1);
        Imgproc.drawContours(m, List.of(contour), 0, new Scalar(255), -1);
        return m;
    }

    // -------------------------------------------------------------------------
    // Reference signature builder
    // -------------------------------------------------------------------------

    /**
     * Binarises the reference BGR mat and builds a {@link VectorSignature}.
     * The reference is always treated without colour pre-filtering because
     * the CF filter is applied to the scene only.
     */
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
    // Selection helpers
    // -------------------------------------------------------------------------

    private static VectorSignature pickRefSig(VectorVariant v,
                                               VectorSignature strict,
                                               VectorSignature normal,
                                               VectorSignature loose) {
        return switch (v) {
            case VECTOR_STRICT, VECTOR_STRICT_CF_LOOSE, VECTOR_STRICT_CF_TIGHT -> strict;
            case VECTOR_NORMAL, VECTOR_NORMAL_CF_LOOSE, VECTOR_NORMAL_CF_TIGHT -> normal;
            default -> loose;
        };
    }

    private static Mat pickSceneMat(VectorVariant v, Mat base, Mat loose, Mat tight) {
        return switch (v.cfMode()) {
            case LOOSE -> loose != null ? loose : base;
            case TIGHT -> tight != null ? tight : base;
            default    -> base;
        };
    }

    private static long pickCfMs(VectorVariant v, long cfLMs, long cfTMs) {
        return switch (v.cfMode()) {
            case LOOSE -> cfLMs;
            case TIGHT -> cfTMs;
            default    -> 0L;
        };
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
    // Tiny helpers
    // -------------------------------------------------------------------------

    private static String sanitise(String s) { return s.replaceAll("[^A-Za-z0-9_\\-]", "_"); }

    private static String shortName(String v) {
        return v.replace("VECTOR_STRICT", "VM_S")
                .replace("VECTOR_NORMAL", "VM_N")
                .replace("VECTOR_LOOSE",  "VM_L")
                .replace("_CF_LOOSE", "-CFL")
                .replace("_CF_TIGHT", "-CFT");
    }
}












