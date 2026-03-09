package org.example.matchers;

import org.example.analytics.AnalysisResult;
import org.example.colour.ColourPreFilter;
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
            List<MatOfPoint> contours = extractContours(sceneForExtraction);

            double sceneArea = (double) sceneForExtraction.rows() * sceneForExtraction.cols();

            double bestScore = 0.0;
            Rect   bestBbox  = null;

            for (MatOfPoint c : contours) {
                // Fast path: build signature directly from tight bounding-box crop (no full-scene render)
                VectorSignature sceneSig = VectorSignature.buildFromContour(
                        c, variant.epsilonFactor(), sceneArea);

                // Full similarity (includes vertex graph)
                double sim = refSig.similarity(sceneSig);
                if (sim > bestScore) {
                    bestScore = sim;
                    bestBbox  = Imgproc.boundingRect(c);
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
     * Converts a BGR scene (or CF-masked BGR scene) into a list of candidate
     * contours that are large enough to be real shapes.
     */
    public static List<MatOfPoint> extractContours(Mat bgrScene) {
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

        contours.removeIf(c -> Imgproc.contourArea(c) < MIN_AREA);
        return contours;
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






