package org.example.matchers;

import org.example.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Morphological Shape Analysis — Milestone 14.
 *
 * <p>3 base variants × base / CF_LOOSE / CF_TIGHT = <b>9 variants total</b>:
 *
 * <ul>
 *   <li>{@code MORPH_POLY}     — vertex-count similarity via {@code approxPolyDP}.
 *       Compares how many polygon vertices the dominant contour approximates to.</li>
 *   <li>{@code MORPH_CIRC}     — circularity similarity:
 *       {@code 4π × area / perimeter²} → 1.0 for a perfect circle, < 1 otherwise.</li>
 *   <li>{@code MORPH_COMBINED} — composite of four shape descriptors:
 *       vertex count, circularity, convexity (isContourConvex), and aspect ratio.
 *       Each component contributes 25% to the final score.</li>
 * </ul>
 *
 * <h2>Pipeline (per variant)</h2>
 * <ol>
 *   <li>Binarise the 128×128 reference; extract its largest contour; compute the
 *       reference shape descriptors (vertex count, circularity, convexity, aspect ratio).</li>
 *   <li>Binarise the full scene; run {@link Imgproc#findContours} to get all external
 *       contours.</li>
 *   <li>For each scene contour above {@value #MIN_CONTOUR_AREA} px²: compute the same
 *       descriptors and score its similarity to the reference descriptors.</li>
 *   <li>The contour with the best score is the detection.  Its bounding rect is
 *       reported as the result bbox.</li>
 * </ol>
 *
 * <p>CF variants apply a colour pre-filter mask to the scene binary before contour
 * extraction, eliminating background contours of the wrong colour.
 *
 * <h2>Scoring formulas</h2>
 * <ul>
 *   <li><b>MORPH_POLY:</b>
 *       {@code score = exp(-|refVerts - sceneVerts| / max(1, refVerts)) × 100}</li>
 *   <li><b>MORPH_CIRC:</b>
 *       {@code score = (1 − |refCirc − sceneCirc|) × 100}, clamped to [0,100]</li>
 *   <li><b>MORPH_COMBINED:</b>
 *       weighted average of the four normalised component scores × 100</li>
 * </ul>
 */
public final class MorphologyAnalyzer {

    // -------------------------------------------------------------------------
    // Variant names
    // -------------------------------------------------------------------------
    public static final String VAR_POLY     = "MORPH_POLY";
    public static final String VAR_CIRC     = "MORPH_CIRC";
    public static final String VAR_COMBINED = "MORPH_COMBINED";

    /** Minimum contour area (px²) to consider. Filters single-pixel noise. */
    private static final double MIN_CONTOUR_AREA = 16.0;

    /** approxPolyDP epsilon as a fraction of the contour arc length. */
    private static final double POLY_EPSILON_FACTOR = 0.04;

    /** Binary threshold for binarisation. */
    private static final int BINARISE_THRESH = 30;

    private MorphologyAnalyzer() {}

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

        // --- Binarise reference (once for base and as mask source for CF) ---
        Mat refBin = binarise(refMat, null);
        ShapeDescriptor refDesc = dominantDescriptor(refBin);
        refBin.release();

        // --- Pre-compute scene binaries for all 3 modes ---
        Mat sceneBin = binarise(sceneMat, null);

        long t0 = System.currentTimeMillis();
        Mat looseMask    = ColourPreFilter.applyToScene(sceneMat, referenceId, ColourPreFilter.LOOSE);
        long cfLMs = System.currentTimeMillis() - t0;
        Mat sceneBinLoose = binarise(sceneMat, looseMask);
        looseMask.release();

        t0 = System.currentTimeMillis();
        Mat tightMask    = ColourPreFilter.applyToScene(sceneMat, referenceId, ColourPreFilter.TIGHT);
        long cfTMs = System.currentTimeMillis() - t0;
        Mat sceneBinTight = binarise(sceneMat, tightMask);
        tightMask.release();

        // --- Extract scene contours once per mode ---
        List<MatOfPoint> sceneContours      = findExternalContours(sceneBin);
        List<MatOfPoint> sceneContoursLoose = findExternalContours(sceneBinLoose);
        List<MatOfPoint> sceneContoursTight = findExternalContours(sceneBinTight);

        sceneBin.release();
        sceneBinLoose.release();
        sceneBinTight.release();

        // --- Run all 3 variants × 3 modes ---
        for (String base : new String[]{ VAR_POLY, VAR_CIRC, VAR_COMBINED }) {
            out.add(runVariant(base, sceneMat, sceneContours,      refDesc, 0L,    referenceId, scene, saveVariants, outputDir));
            out.add(runVariant(base + "_CF_LOOSE", sceneMat, sceneContoursLoose, refDesc, cfLMs, referenceId, scene, saveVariants, outputDir));
            out.add(runVariant(base + "_CF_TIGHT", sceneMat, sceneContoursTight, refDesc, cfTMs, referenceId, scene, saveVariants, outputDir));
        }

        releaseContours(sceneContours);
        releaseContours(sceneContoursLoose);
        releaseContours(sceneContoursTight);

        return out;
    }

    // -------------------------------------------------------------------------
    // Single variant
    // -------------------------------------------------------------------------

    private static AnalysisResult runVariant(String variantName,
                                              Mat sceneMat,
                                              List<MatOfPoint> sceneContours,
                                              ShapeDescriptor refDesc,
                                              long preFilterMs,
                                              ReferenceId referenceId,
                                              SceneEntry scene,
                                              Set<String> saveVariants,
                                              Path outputDir) {
        long t0 = System.currentTimeMillis();
        try {
            if (refDesc == null) {
                return AnalysisResult.error(variantName, referenceId,
                        scene.variantLabel(), scene.category(), scene.backgroundId(),
                        0L, scenePx(scene), "No reference descriptor");
            }

            double bestScore = -1;
            Rect   bestBbox  = null;

            for (MatOfPoint c : sceneContours) {
                if (Imgproc.contourArea(c) < MIN_CONTOUR_AREA) continue;
                ShapeDescriptor sd = describe(c);
                double score = score(variantName, refDesc, sd);
                if (score > bestScore) {
                    bestScore = score;
                    bestBbox  = Imgproc.boundingRect(c);
                }
            }

            long elapsed = System.currentTimeMillis() - t0;
            if (bestBbox == null) {
                bestScore = 0.0;
                bestBbox  = new Rect(0, 0, 1, 1);
            }
            double finalScore = Math.max(0, Math.min(100, bestScore));

            Path savedPath = null;
            if (saveVariants.contains(variantName)) {
                savedPath = writeAnnotated(sceneMat, bestBbox, variantName, finalScore,
                        referenceId, scene, outputDir);
            }

            return new AnalysisResult(variantName, referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    finalScore, bestBbox, elapsed, preFilterMs,
                    scenePx(scene), savedPath, false, null);

        } catch (Exception e) {
            return AnalysisResult.error(variantName, referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    System.currentTimeMillis() - t0, scenePx(scene), e.getMessage());
        }
    }

    // -------------------------------------------------------------------------
    // Scoring
    // -------------------------------------------------------------------------

    /**
     * Returns a 0–100 score for how well {@code scene} matches {@code ref}
     * under the named variant.
     */
    private static double score(String variant, ShapeDescriptor ref, ShapeDescriptor scene) {
        return switch (variant) {
            case VAR_POLY     -> scoreVerts(ref, scene);
            case VAR_CIRC     -> scoreCirc(ref, scene);
            case VAR_COMBINED -> scoreCombined(ref, scene);
            default -> {
                // CF suffixes — strip and recurse
                String base = variant.replace("_CF_LOOSE", "").replace("_CF_TIGHT", "");
                yield score(base, ref, scene);
            }
        };
    }

    /** Vertex-count similarity: exponential decay on the absolute vertex difference. */
    private static double scoreVerts(ShapeDescriptor ref, ShapeDescriptor s) {
        double diff = Math.abs(ref.vertices - s.vertices);
        return Math.exp(-diff / Math.max(1, ref.vertices)) * 100.0;
    }

    /** Circularity similarity: 1 − |refCirc − sceneCirc|, each in [0,1]. */
    private static double scoreCirc(ShapeDescriptor ref, ShapeDescriptor s) {
        return Math.max(0, (1.0 - Math.abs(ref.circularity - s.circularity))) * 100.0;
    }

    /**
     * Combined score: equal-weighted blend of vertex, circularity,
     * convexity agreement, and aspect-ratio similarity.
     */
    private static double scoreCombined(ShapeDescriptor ref, ShapeDescriptor s) {
        double vScore    = scoreVerts(ref, s) / 100.0;
        double cScore    = scoreCirc(ref, s)  / 100.0;
        // Convexity: 1.0 if both agree, 0.5 if they disagree
        double convScore = (ref.convex == s.convex) ? 1.0 : 0.5;
        // Aspect ratio similarity: 1 − |log(refAR / sceneAR)| / log(4), clamped to [0,1]
        double arSim = 1.0 - Math.min(1.0,
                Math.abs(Math.log(ref.aspectRatio / Math.max(0.01, s.aspectRatio))) / Math.log(4.0));
        return (vScore + cScore + convScore + arSim) / 4.0 * 100.0;
    }

    // -------------------------------------------------------------------------
    // Shape descriptor
    // -------------------------------------------------------------------------

    /**
     * Per-contour shape descriptor extracted once and reused across all variant scores.
     */
    private record ShapeDescriptor(int vertices, double circularity, boolean convex, double aspectRatio) {}

    /**
     * Computes the shape descriptor for a single contour.
     *
     * @param c  the contour (MatOfPoint)
     * @return descriptor, or null if the contour is degenerate
     */
    private static ShapeDescriptor describe(MatOfPoint c) {
        double area      = Imgproc.contourArea(c);
        double perimeter = Imgproc.arcLength(new MatOfPoint2f(c.toArray()), true);
        if (area < MIN_CONTOUR_AREA || perimeter < 1.0) return null;

        // approxPolyDP vertex count
        MatOfPoint2f c2f  = new MatOfPoint2f(c.toArray());
        MatOfPoint2f poly = new MatOfPoint2f();
        Imgproc.approxPolyDP(c2f, poly, POLY_EPSILON_FACTOR * perimeter, true);
        int vertices = (int) poly.total();
        poly.release();
        c2f.release();

        // Circularity = 4π·area / perimeter²
        double circularity = Math.min(1.0, (4.0 * Math.PI * area) / (perimeter * perimeter));

        // Convexity
        boolean convex = Imgproc.isContourConvex(c);

        // Aspect ratio from bounding rect
        Rect br = Imgproc.boundingRect(c);
        double aspectRatio = (br.height > 0) ? (double) br.width / br.height : 1.0;

        return new ShapeDescriptor(vertices, circularity, convex, aspectRatio);
    }

    /**
     * Extracts a descriptor from the largest contour in a binary image.
     * Returns null if no suitable contour is found.
     */
    private static ShapeDescriptor dominantDescriptor(Mat bin) {
        List<MatOfPoint> contours = findExternalContours(bin);
        MatOfPoint best = null;
        double bestArea = 0;
        for (MatOfPoint c : contours) {
            double a = Imgproc.contourArea(c);
            if (a > bestArea) { bestArea = a; best = c; }
        }
        ShapeDescriptor desc = (best != null && bestArea >= MIN_CONTOUR_AREA)
                ? describe(best) : null;
        releaseContours(contours);
        return desc;
    }

    // -------------------------------------------------------------------------
    // Image helpers
    // -------------------------------------------------------------------------

    private static Mat binarise(Mat bgr, Mat mask) {
        Mat grey = new Mat();
        Imgproc.cvtColor(bgr, grey, Imgproc.COLOR_BGR2GRAY);
        if (mask != null && !mask.empty()) {
            Mat masked = new Mat(grey.size(), grey.type(), Scalar.all(0));
            grey.copyTo(masked, mask);
            grey.release();
            grey = masked;
        }
        Mat bin = new Mat();
        Imgproc.threshold(grey, bin, BINARISE_THRESH, 255, Imgproc.THRESH_BINARY);
        grey.release();
        return bin;
    }

    private static List<MatOfPoint> findExternalContours(Mat bin) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(bin.clone(), contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        hierarchy.release();
        return contours;
    }

    private static void releaseContours(List<MatOfPoint> contours) {
        for (MatOfPoint c : contours) c.release();
    }

    // -------------------------------------------------------------------------
    // Annotation writer
    // -------------------------------------------------------------------------

    private static Path writeAnnotated(Mat scene, Rect bbox, String variant, double score,
                                        ReferenceId refId, SceneEntry sceneEntry,
                                        Path outputDir) {
        try {
            Path dir = outputDir.resolve("annotated").resolve(sanitise(variant));
            Files.createDirectories(dir);
            String fname = sanitise(refId.name()) + "_" + sanitise(sceneEntry.variantLabel()) + ".png";
            Path dest = dir.resolve(fname);

            Mat    m      = scene.clone();
            Scalar colour = score >= 70 ? new Scalar(0, 200, 0)
                          : score >= 40 ? new Scalar(0, 200, 200)
                          :               new Scalar(0, 0, 200);
            if (bbox.width > 1 && bbox.height > 1) {
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

    private static int    scenePx(SceneEntry s)  { return s.sceneMat().cols() * s.sceneMat().rows(); }
    private static String sanitise(String s)      { return s.replaceAll("[^A-Za-z0-9_\\-]", "_"); }
    private static String shortName(String v)     {
        return v.replace("MORPH_COMBINED", "MA_CMB")
                .replace("MORPH_POLY",     "MA_PLY")
                .replace("MORPH_CIRC",     "MA_CIR")
                .replace("_CF_LOOSE",      "·CFL")
                .replace("_CF_TIGHT",      "·CFT");
    }
}

