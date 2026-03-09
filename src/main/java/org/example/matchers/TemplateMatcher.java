package org.example.matchers;

import org.example.analytics.AnalysisResult;
import org.example.colour.CfMode;
import org.example.colour.ColourFirstLocator;
import org.example.colour.ColourPreFilter;
import org.example.factories.ReferenceId;
import org.example.factories.ReferenceImageFactory;
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
 * Template Matching technique — Milestone 7.
 *
 * <p>Runs all 6 OpenCV TM_* methods, plus {@code _CF_LOOSE} and {@code _CF_TIGHT}
 * colour-pre-filtered variants, giving <b>18 variants total</b>.
 *
 * <p><b>Transparency / masking:</b> because reference images have a solid-black
 * background, a foreground mask is derived from the reference Mat using
 * {@link ReferenceImageFactory#buildMask(Mat)}.  For {@code TM_SQDIFF} and
 * {@code TM_CCORR_NORMED} (the only methods OpenCV supports masked matching for)
 * the mask is passed directly to {@link Imgproc#matchTemplate}.  For all other
 * methods, background pixels are zeroed in both the template and each scene crop
 * before comparison, achieving the same effect.
 *
 * <p>Annotated images are written straight to disk (never held in memory).
 * {@link AnalysisResult#annotatedPath()} is set only for variants listed in
 * {@code saveVariants}; pass an empty set to skip all image saving.
 */
public final class TemplateMatcher {

    /**
     * Base method names (no CF suffix) — kept for backward compatibility with
     * any code that iterates base method names as strings.
     * Prefer {@link TmVariant} for new code.
     */
    public static final String[] BASE_METHODS = {
        TmVariant.TM_SQDIFF.variantName(),        TmVariant.TM_SQDIFF_NORMED.variantName(),
        TmVariant.TM_CCORR.variantName(),         TmVariant.TM_CCORR_NORMED.variantName(),
        TmVariant.TM_CCOEFF.variantName(),        TmVariant.TM_CCOEFF_NORMED.variantName()
    };

    /** @deprecated Use {@link TmVariant#TM_CCOEFF_NORMED_CF1_LOOSE}. */
    @Deprecated public static final String CF1_LOOSE = TmVariant.TM_CCOEFF_NORMED_CF1_LOOSE.variantName();
    /** @deprecated Use {@link TmVariant#TM_CCOEFF_NORMED_CF1_TIGHT}. */
    @Deprecated public static final String CF1_TIGHT = TmVariant.TM_CCOEFF_NORMED_CF1_TIGHT.variantName();

    /** The 6 base (non-CF) variants in order. */
    private static final TmVariant[] BASE_VARIANTS = {
        TmVariant.TM_SQDIFF,        TmVariant.TM_SQDIFF_NORMED,
        TmVariant.TM_CCORR,         TmVariant.TM_CCORR_NORMED,
        TmVariant.TM_CCOEFF,        TmVariant.TM_CCOEFF_NORMED
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

        // Build foreground mask once per reference — reused across all 18 variants.
        // Black pixels in the reference (background) become 0 in the mask;
        // shape pixels become 255.  This lets matchTemplate ignore the black canvas.
        Mat refMask = ReferenceImageFactory.buildMask(refMat);

        for (TmVariant base : BASE_VARIANTS) {
            out.add(runVariant(base.variantName(), base.cvFlag(), base.lowerIsBetter(),
                    sceneMat, refMat, refMask, 0L,
                    referenceId, scene, saveVariants, outputDir));

            // CF_LOOSE
            long t0 = System.currentTimeMillis();
            TmVariant looseVar = variantWithCf(base, CfMode.LOOSE);
            Mat sL = ColourPreFilter.applyMaskedBgrToScene(sceneMat, referenceId, ColourPreFilter.LOOSE);
            Mat rL = ColourPreFilter.applyMaskedBgrToReference(refMat, referenceId, ColourPreFilter.LOOSE);
            long cfL = System.currentTimeMillis() - t0;
            out.add(runVariant(looseVar.variantName(), base.cvFlag(), base.lowerIsBetter(),
                    sL, rL, refMask, cfL, referenceId, scene, saveVariants, outputDir));
            sL.release(); rL.release();

            // CF_TIGHT
            t0 = System.currentTimeMillis();
            TmVariant tightVar = variantWithCf(base, CfMode.TIGHT);
            Mat sT = ColourPreFilter.applyMaskedBgrToScene(sceneMat, referenceId, ColourPreFilter.TIGHT);
            Mat rT = ColourPreFilter.applyMaskedBgrToReference(refMat, referenceId, ColourPreFilter.TIGHT);
            long cfT = System.currentTimeMillis() - t0;
            out.add(runVariant(tightVar.variantName(), base.cvFlag(), base.lowerIsBetter(),
                    sT, rT, refMask, cfT, referenceId, scene, saveVariants, outputDir));
            sT.release(); rT.release();
        }

        // ---- CF1 variants (TM_CCOEFF_NORMED inside colour-first windows) ----
        for (TmVariant cf1 : new TmVariant[]{
                TmVariant.TM_CCOEFF_NORMED_CF1_LOOSE,
                TmVariant.TM_CCOEFF_NORMED_CF1_TIGHT}) {

            String cf1Name = cf1.variantName();
            double tol = cf1.cfMode().hueTolerance();
            long t0 = System.currentTimeMillis();
            List<Rect> windows = ColourFirstLocator.propose(sceneMat, referenceId, tol);
            long cfMs = System.currentTimeMillis() - t0;

            double bestScore = -1;
            Rect   bestBbox  = windows.get(0);
            for (Rect w : windows) {
                Mat crop = sceneMat.submat(w);
                AnalysisResult r = runVariant(cf1Name, Imgproc.TM_CCOEFF_NORMED, false,
                        crop, refMat, refMask, cfMs, referenceId, scene,
                        saveVariants, outputDir);
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

        refMask.release();
        return out;
    }

    // =========================================================================
    // Single variant
    // =========================================================================

    /** Finds the TmVariant for a base variant with a given CF mode applied. */
    private static TmVariant variantWithCf(TmVariant base, CfMode cf) {
        String targetName = base.variantName() + cf.suffix();
        for (TmVariant v : TmVariant.values()) {
            if (v.variantName().equals(targetName)) return v;
        }
        throw new IllegalArgumentException("No TmVariant for " + targetName);
    }

    /**
     * Whether a given TM flag supports OpenCV's native masked matchTemplate.
     * Only TM_SQDIFF and TM_CCORR_NORMED are supported by OpenCV.
     */
    private static boolean supportsNativeMask(int tmFlag) {
        return tmFlag == Imgproc.TM_SQDIFF || tmFlag == Imgproc.TM_CCORR_NORMED;
    }

    private static AnalysisResult runVariant(String variantName,
                                              int tmFlag,
                                              boolean lowerIsBetter,
                                              Mat searchImage,
                                              Mat tmpl,
                                              Mat refMask,
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
            if (supportsNativeMask(tmFlag)) {
                // OpenCV native masked matching (TM_SQDIFF, TM_CCORR_NORMED only)
                Imgproc.matchTemplate(searchImage, tmpl, result, tmFlag, refMask);
            } else {
                // Software emulation: zero background pixels in the template so the
                // black canvas doesn't contribute to the score.
                // Mat.copyTo(dst, mask) copies only where mask=255, leaving dst zeros
                // elsewhere — correctly handles 3-channel template with 1-channel mask.
                Mat maskedTmpl = Mat.zeros(tmpl.size(), tmpl.type());
                try {
                    tmpl.copyTo(maskedTmpl, refMask);
                    Imgproc.matchTemplate(searchImage, maskedTmpl, result, tmFlag);
                } finally {
                    maskedTmpl.release();
                }
            }

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
            String sceneRef = sceneEntry.primaryReferenceId() != null
                    ? sanitise(sceneEntry.primaryReferenceId().name()) : "neg";
            String fname = sanitise(refId.name()) + "_vs_" + sceneRef
                    + "_" + sanitise(sceneEntry.variantLabel()) + ".png";
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

