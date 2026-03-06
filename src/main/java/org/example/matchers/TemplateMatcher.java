package org.example.matchers;

import org.example.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Template Matching technique — Milestone 7.
 *
 * <p>Runs all 6 OpenCV TM_* methods, plus {@code _CF_LOOSE} and {@code _CF_TIGHT}
 * colour-pre-filtered variants, giving <b>18 variants total</b>.
 *
 * <p>Annotated images are written straight to disk (never held in memory).
 * {@link AnalysisResult#annotatedPath()} is set only for variants listed in
 * {@code saveVariants}; pass an empty set to skip all image saving.
 */
public final class TemplateMatcher {

    public static final String[] BASE_METHODS = {
        "TM_SQDIFF", "TM_SQDIFF_NORMED",
        "TM_CCORR",  "TM_CCORR_NORMED",
        "TM_CCOEFF", "TM_CCOEFF_NORMED"
    };

    /** CF1 representative variant names (TM_CCOEFF_NORMED only). */
    public static final String CF1_LOOSE = "TM_CCOEFF_NORMED_CF1_LOOSE";
    public static final String CF1_TIGHT = "TM_CCOEFF_NORMED_CF1_TIGHT";

    private static final int[] TM_FLAGS = {
        Imgproc.TM_SQDIFF, Imgproc.TM_SQDIFF_NORMED,
        Imgproc.TM_CCORR,  Imgproc.TM_CCORR_NORMED,
        Imgproc.TM_CCOEFF, Imgproc.TM_CCOEFF_NORMED
    };

    private static final boolean[] LOWER_IS_BETTER = {
        true,  true,
        false, false,
        false, false
    };

    private TemplateMatcher() {}

    /**
     * Runs all 18 variants against one (reference, scene) pair.
     *
     * @param referenceId  reference ID
     * @param refMat       128×128 BGR reference Mat — caller retains ownership
     * @param scene        scene to search in
     * @param saveVariants variant names whose annotated PNG should be written to disk
     * @param outputDir    root output directory; sub-dirs per variant are created as needed
     * @return 18 {@link AnalysisResult} objects (one per variant)
     */
    public static List<AnalysisResult> match(ReferenceId referenceId,
                                              Mat refMat,
                                              SceneEntry scene,
                                              Set<String> saveVariants,
                                              Path outputDir) {
        List<AnalysisResult> out = new ArrayList<>(18);
        Mat sceneMat = scene.sceneMat();

        for (int i = 0; i < BASE_METHODS.length; i++) {
            String  baseName = BASE_METHODS[i];
            int     flag     = TM_FLAGS[i];
            boolean lower    = LOWER_IS_BETTER[i];

            // Base
            out.add(runVariant(baseName, flag, lower, sceneMat, refMat, 0L,
                    referenceId, scene, saveVariants, outputDir));

            // CF_LOOSE
            long t0 = System.currentTimeMillis();
            Mat sL = ColourPreFilter.applyToScene(sceneMat, referenceId, ColourPreFilter.LOOSE);
            Mat rL = ColourPreFilter.applyToReference(refMat,  referenceId, ColourPreFilter.LOOSE);
            long cfL = System.currentTimeMillis() - t0;
            out.add(runVariant(baseName + "_CF_LOOSE", flag, lower, sL, rL, cfL,
                    referenceId, scene, saveVariants, outputDir));
            sL.release(); rL.release();

            // CF_TIGHT
            t0 = System.currentTimeMillis();
            Mat sT = ColourPreFilter.applyToScene(sceneMat, referenceId, ColourPreFilter.TIGHT);
            Mat rT = ColourPreFilter.applyToReference(refMat,  referenceId, ColourPreFilter.TIGHT);
            long cfT = System.currentTimeMillis() - t0;
            out.add(runVariant(baseName + "_CF_TIGHT", flag, lower, sT, rT, cfT,
                    referenceId, scene, saveVariants, outputDir));
            sT.release(); rT.release();
        }

        // ---- CF1 variants (TM_CCOEFF_NORMED inside colour-first windows) ----
        for (String cf1Name : new String[]{CF1_LOOSE, CF1_TIGHT}) {
            double tol = cf1Name.endsWith("LOOSE") ? ColourPreFilter.LOOSE : ColourPreFilter.TIGHT;
            long t0 = System.currentTimeMillis();
            List<Rect> windows = ColourFirstLocator.propose(sceneMat, referenceId, tol);
            long cfMs = System.currentTimeMillis() - t0;

            double bestScore = -1;
            Rect   bestBbox  = windows.get(0);
            for (Rect w : windows) {
                Mat crop = sceneMat.submat(w);
                AnalysisResult r = runVariant(cf1Name, Imgproc.TM_CCOEFF_NORMED, false,
                        crop, refMat, cfMs, referenceId, scene,
                        saveVariants, outputDir);
                // Translate bbox from crop-local to scene-global coords
                if (r.matchScorePercent() > bestScore) {
                    bestScore = r.matchScorePercent();
                    Rect lb = r.boundingRect();
                    if (lb != null) {
                        bestBbox = new Rect(w.x + lb.x, w.y + lb.y, lb.width, lb.height);
                    }
                }
            }

            Path savedPath = null;
            if (saveVariants.contains(cf1Name)) {
                savedPath = writeAnnotated(sceneMat, bestBbox, cf1Name,
                        Math.max(0, bestScore), referenceId, scene, outputDir);
            }
            out.add(new AnalysisResult(cf1Name, referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    Math.max(0, bestScore), bestBbox,
                    System.currentTimeMillis() - t0, cfMs,
                    scenePx(scene), savedPath, false, null));
        }

        return out;
    }

    // =========================================================================
    // Single variant
    // =========================================================================

    private static AnalysisResult runVariant(String variantName,
                                              int tmFlag,
                                              boolean lowerIsBetter,
                                              Mat searchImage,
                                              Mat tmpl,
                                              long preFilterMs,
                                              ReferenceId referenceId,
                                              SceneEntry scene,
                                              Set<String> saveVariants,
                                              Path outputDir) {
        long t0 = System.currentTimeMillis();
        try {
            if (searchImage.rows() < tmpl.rows() || searchImage.cols() < tmpl.cols()) {
                return AnalysisResult.error(variantName, referenceId,
                        scene.variantLabel(), scene.category(), scene.backgroundId(),
                        0L, scenePx(scene), "Scene smaller than template");
            }

            Mat result = new Mat();
            Imgproc.matchTemplate(searchImage, tmpl, result, tmFlag);
            Core.MinMaxLocResult mmr = Core.minMaxLoc(result);
            result.release();

            long   elapsed = System.currentTimeMillis() - t0;
            double score;
            Point  bestLoc;

            if (lowerIsBetter) {
                bestLoc = mmr.minLoc;
                if (tmFlag == Imgproc.TM_SQDIFF_NORMED) {
                    score = (1.0 - mmr.minVal) * 100.0;
                } else {
                    double maxP = 255.0 * 255.0 * tmpl.rows() * tmpl.cols() * tmpl.channels();
                    score = Math.max(0, (1.0 - mmr.minVal / maxP)) * 100.0;
                }
            } else {
                bestLoc = mmr.maxLoc;
                if (tmFlag == Imgproc.TM_CCORR_NORMED || tmFlag == Imgproc.TM_CCOEFF_NORMED) {
                    score = mmr.maxVal * 100.0;
                } else {
                    double maxP = 255.0 * tmpl.rows() * tmpl.cols() * tmpl.channels();
                    score = Math.max(0, Math.min(100, (mmr.maxVal / maxP) * 100.0));
                }
            }
            score = Math.max(0, Math.min(100, score));

            Rect bbox = clampRect(
                    new Rect((int) bestLoc.x, (int) bestLoc.y, tmpl.cols(), tmpl.rows()),
                    searchImage.cols(), searchImage.rows());

            // Write annotated image to disk immediately — never hold in memory
            Path savedPath = null;
            if (saveVariants.contains(variantName)) {
                savedPath = writeAnnotated(scene.sceneMat(), bbox, variantName, score,
                        referenceId, scene, outputDir);
            }

            return new AnalysisResult(variantName, referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    score, bbox, elapsed, preFilterMs,
                    scenePx(scene), savedPath, false, null);

        } catch (Exception e) {
            return AnalysisResult.error(variantName, referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    System.currentTimeMillis() - t0, scenePx(scene), e.getMessage());
        }
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private static Path writeAnnotated(Mat scene, Rect bbox, String variant, double score,
                                        ReferenceId refId, SceneEntry sceneEntry,
                                        Path outputDir) {
        try {
            Path dir = outputDir.resolve("annotated").resolve(sanitise(variant));
            Files.createDirectories(dir);
            String fname = sanitise(refId.name()) + "_" + sanitise(sceneEntry.variantLabel()) + ".png";
            Path dest = dir.resolve(fname);

            Mat m = scene.clone();
            Scalar colour = score >= 70 ? new Scalar(0, 200, 0)
                          : score >= 40 ? new Scalar(0, 200, 200)
                          :               new Scalar(0, 0, 200);
            Imgproc.rectangle(m, new Point(bbox.x, bbox.y),
                    new Point(bbox.x + bbox.width, bbox.y + bbox.height), colour, 2);
            Imgproc.putText(m, shortName(variant),
                    new Point(4, 13), Imgproc.FONT_HERSHEY_SIMPLEX, 0.32,
                    new Scalar(200, 200, 200), 1);
            Imgproc.putText(m, String.format("%.1f%%", score),
                    new Point(4, 28), Imgproc.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1);
            Imgcodecs.imwrite(dest.toString(), m);
            m.release();
            // Return path relative to outputDir so report.html is portable
            return outputDir.toAbsolutePath().relativize(dest.toAbsolutePath());
        } catch (Exception e) {
            return null;
        }
    }

    private static Rect clampRect(Rect r, int maxW, int maxH) {
        int x = Math.max(0, Math.min(r.x, maxW - 1));
        int y = Math.max(0, Math.min(r.y, maxH - 1));
        int w = Math.min(r.width,  maxW - x);
        int h = Math.min(r.height, maxH - y);
        return new Rect(x, y, Math.max(1, w), Math.max(1, h));
    }

    private static int     scenePx(SceneEntry s) { return s.sceneMat().cols() * s.sceneMat().rows(); }
    private static String  sanitise(String s)    { return s.replaceAll("[^A-Za-z0-9_\\-]", "_"); }
    private static String  shortName(String v)   { return v.replace("TM_","").replace("_NORMED","N").replace("_CF_","·"); }
}

