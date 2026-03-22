package org.example.matchers;

import org.example.analytics.AnalysisResult;
import org.example.colour.ColourFirstLocator;
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
 * Pixel Difference baseline matcher — Milestone 16.
 *
 * <p>1 base variant × base / CF_LOOSE / CF_TIGHT = <b>3 variants total</b>:
 * <ul>
 *   <li>{@code PIXEL_DIFF} — sliding-window absolute pixel difference</li>
 *   <li>{@code PIXEL_DIFF_CF_LOOSE} — same, colour pre-filtered (±15° hue)</li>
 *   <li>{@code PIXEL_DIFF_CF_TIGHT} — same, colour pre-filtered (±8° hue)</li>
 * </ul>
 *
 * <h2>Pipeline</h2>
 * <ol>
 *   <li>Convert both reference (128×128) and scene to greyscale.</li>
 *   <li>For CF variants, zero out non-foreground pixels via the colour mask before
 *       computing differences — isolating only the target colour in both images.</li>
 *   <li>Slide a 128×128 window across the scene at stride {@value #STRIDE}.</li>
 *   <li>For each crop: {@code Core.absdiff(refGrey, cropGrey)} →
 *       {@code Core.sumElems(diff)} → sum all channels.</li>
 *   <li>Score = {@code (1 − diffSum / maxPossibleDiff) × 100}, where
 *       {@code maxPossibleDiff = 255 × width × height}.</li>
 *   <li>The crop with the <em>highest</em> score is the detection.
 *       Its position is the bounding box.</li>
 * </ol>
 *
 * <p>This is deliberately the simplest possible baseline — no shape awareness,
 * no feature extraction.  It serves as a lower-bound reference for comparing
 * all other techniques.
 *
 * <p><b>Expected behaviour:</b>
 * <ul>
 *   <li>A_CLEAN — very high score when the scene background matches the reference canvas</li>
 *   <li>B_TRANSFORMED — score drops sharply under scale/rotation (not invariant)</li>
 *   <li>C_DEGRADED — degrades with noise/blur exactly as expected for raw pixel comparison</li>
 *   <li>D_NEGATIVE — typically gives a non-zero "best window" score (false positive risk),
 *       since some background region will always partially resemble the reference pixels;
 *       CF variants reduce this by zeroing mismatched colours first</li>
 * </ul>
 */
public final class PixelDiffMatcher {

    /** @deprecated Use {@link PixelDiffVariant#PIXEL_DIFF}. */
    @Deprecated public static final String VAR_BASE  = PixelDiffVariant.PIXEL_DIFF.variantName();
    /** @deprecated Use {@link PixelDiffVariant#PIXEL_DIFF_CF_LOOSE}. */
    @Deprecated public static final String VAR_LOOSE = PixelDiffVariant.PIXEL_DIFF_CF_LOOSE.variantName();
    /** @deprecated Use {@link PixelDiffVariant#PIXEL_DIFF_CF_TIGHT}. */
    @Deprecated public static final String VAR_TIGHT = PixelDiffVariant.PIXEL_DIFF_CF_TIGHT.variantName();

    /** Reference tile size — matches all reference images. */
    private static final int TILE   = 128;
    /** Sliding window stride in pixels. */
    private static final int STRIDE = 8;

    private PixelDiffMatcher() {}

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

        // Reference grey — base and CF variants share the greyscale ref,
        // but CF additionally zeroes out non-foreground pixels in the ref.
        Mat refGrey = toGrey(refMat, null);

        long t0 = System.currentTimeMillis();
        Mat looseMask = ColourPreFilter.applyToScene(sceneMat, referenceId, ColourPreFilter.LOOSE);
        long cfLMs    = System.currentTimeMillis() - t0;

        t0 = System.currentTimeMillis();
        Mat tightMask = ColourPreFilter.applyToScene(sceneMat, referenceId, ColourPreFilter.TIGHT);
        long cfTMs    = System.currentTimeMillis() - t0;

        // Masked reference greys (colour-isolated)
        Mat refMaskL    = ColourPreFilter.applyToReference(refMat, referenceId, ColourPreFilter.LOOSE);
        Mat refGreyLoose = toGrey(refMat, refMaskL);
        refMaskL.release();

        Mat refMaskT    = ColourPreFilter.applyToReference(refMat, referenceId, ColourPreFilter.TIGHT);
        Mat refGreyTight = toGrey(refMat, refMaskT);
        refMaskT.release();

        // Scene grey variants
        Mat sceneGrey      = toGrey(sceneMat, null);
        Mat sceneGreyLoose = toGrey(sceneMat, looseMask);
        Mat sceneGreyTight = toGrey(sceneMat, tightMask);
        looseMask.release();
        tightMask.release();

        out.add(runVariant(VAR_BASE,  sceneMat, sceneGrey,      refGrey,      0L,    referenceId, scene, saveVariants, outputDir));
        out.add(runVariant(VAR_LOOSE, sceneMat, sceneGreyLoose, refGreyLoose, cfLMs, referenceId, scene, saveVariants, outputDir));
        out.add(runVariant(VAR_TIGHT, sceneMat, sceneGreyTight, refGreyTight, cfTMs, referenceId, scene, saveVariants, outputDir));

        refGrey.release();
        refGreyLoose.release();
        refGreyTight.release();
        sceneGrey.release();
        sceneGreyLoose.release();
        sceneGreyTight.release();

        // ---- CF1 variants (PIXEL_DIFF inside colour-first windows) ----
        for (PixelDiffVariant cf1 : new PixelDiffVariant[]{
                PixelDiffVariant.PIXEL_DIFF_CF1_LOOSE,
                PixelDiffVariant.PIXEL_DIFF_CF1_TIGHT}) {

            String cf1Name  = cf1.variantName();
            double tol      = cf1.cfMode().hueTolerance();
            long   cf1Start = System.currentTimeMillis();
            List<Rect> windows = ColourFirstLocator.propose(sceneMat, referenceId, tol);
            long cfMs = System.currentTimeMillis() - cf1Start;

            // Colour-isolated reference grey for this tolerance
            Mat rMask = ColourPreFilter.applyToReference(refMat, referenceId, tol);
            Mat rGrey = toGrey(refMat, rMask);
            rMask.release();

            double bestScore = -1;
            Rect   bestBbox  = windows.get(0);

            for (Rect w : windows) {
                Mat sMask  = ColourPreFilter.applyToScene(new Mat(sceneMat, w), referenceId, tol);
                Mat sGrey  = toGrey(new Mat(sceneMat, w), sMask);
                sMask.release();
                AnalysisResult r = runVariant(cf1Name, new Mat(sceneMat, w),
                        sGrey, rGrey, cfMs, referenceId, scene, saveVariants, outputDir);
                sGrey.release();
                if (r.matchScorePercent() > bestScore) {
                    bestScore = r.matchScorePercent();
                    Rect lb   = r.boundingRect();
                    if (lb != null)
                        bestBbox = new Rect(w.x + lb.x, w.y + lb.y, lb.width, lb.height);
                }
            }
            rGrey.release();

            Path savedPath = null;
            if (saveVariants.contains(cf1Name)) {
                savedPath = writeAnnotated(sceneMat, bestBbox, cf1Name,
                        Math.max(0, bestScore), referenceId, scene, outputDir);
            }
            out.add(new AnalysisResult(cf1Name, referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    Math.max(0, bestScore), bestBbox,
                    System.currentTimeMillis() - cf1Start, cfMs,
                    scenePx(scene), savedPath, false, null,
                    AnalysisResult.ScoringLayers.ZERO));
        }

        return out;
    }

    // -------------------------------------------------------------------------
    // Single variant — sliding window absdiff
    // -------------------------------------------------------------------------

    private static AnalysisResult runVariant(String variantName,
                                              Mat sceneMat,
                                              Mat sceneGrey,
                                              Mat refGrey,
                                              long preFilterMs,
                                              ReferenceId referenceId,
                                              SceneEntry scene,
                                              Set<String> saveVariants,
                                              Path outputDir) {
        long t0 = System.currentTimeMillis();
        try {
            int sceneW = sceneGrey.cols();
            int sceneH = sceneGrey.rows();

            if (sceneW < TILE || sceneH < TILE) {
                return AnalysisResult.error(variantName, referenceId,
                        scene.variantLabel(), scene.category(), scene.backgroundId(),
                        0L, scenePx(scene), "Scene smaller than tile");
            }

            // maxPossibleDiff: 255 per pixel × tile area (single channel)
            double maxDiff = 255.0 * TILE * TILE;

            double bestScore = -1;
            Rect   bestBbox  = new Rect(0, 0, TILE, TILE);

            for (int y = 0; y <= sceneH - TILE; y += STRIDE) {
                for (int x = 0; x <= sceneW - TILE; x += STRIDE) {
                    Mat crop = sceneGrey.submat(new Rect(x, y, TILE, TILE));
                    Mat diff = new Mat();
                    Core.absdiff(refGrey, crop, diff);
                    double diffSum = Core.sumElems(diff).val[0];
                    diff.release();

                    double score = (1.0 - diffSum / maxDiff) * 100.0;
                    if (score > bestScore) {
                        bestScore = score;
                        bestBbox  = new Rect(x, y, TILE, TILE);
                    }
                }
            }

            long   elapsed    = System.currentTimeMillis() - t0;
            double finalScore = Math.max(0, Math.min(100, bestScore));

            // Tighten the bbox to just the foreground pixels within the winning window
            Rect tightBbox = tightenBbox(sceneMat, bestBbox);

            Path savedPath = null;
            if (saveVariants.contains(variantName)) {
                savedPath = writeAnnotated(sceneMat, tightBbox, variantName, finalScore,
                        referenceId, scene, outputDir);
            }

            return new AnalysisResult(variantName, referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    finalScore, tightBbox, elapsed, preFilterMs,
                    scenePx(scene), savedPath, false, null, AnalysisResult.ScoringLayers.ZERO);

        } catch (Exception e) {
            return AnalysisResult.error(variantName, referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    System.currentTimeMillis() - t0, scenePx(scene), e.getMessage());
        }
    }

    // -------------------------------------------------------------------------
    // Bbox tightening
    // -------------------------------------------------------------------------

    /**
     * Tightens the winning window to the bounding rect of its non-zero (foreground)
     * pixels.  Falls back to the original window if the crop is all black.
     */
    private static Rect tightenBbox(Mat sceneBGR, Rect window) {
        try {
            int sceneW = sceneBGR.cols(), sceneH = sceneBGR.rows();
            int wx = Math.max(0, window.x);
            int wy = Math.max(0, window.y);
            int ww = Math.min(window.width,  sceneW - wx);
            int wh = Math.min(window.height, sceneH - wy);
            if (ww <= 0 || wh <= 0) return window;
            Rect safeWin = new Rect(wx, wy, ww, wh);

            Mat crop = new Mat(sceneBGR, safeWin);
            Mat grey = new Mat();
            Imgproc.cvtColor(crop, grey, Imgproc.COLOR_BGR2GRAY);
            Mat bin = new Mat();
            Imgproc.threshold(grey, bin, 20, 255, Imgproc.THRESH_BINARY);
            grey.release();

            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(bin.clone(), contours, hierarchy,
                    Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            hierarchy.release();
            bin.release();

            if (contours.isEmpty()) return window;

            int x1 = Integer.MAX_VALUE, y1 = Integer.MAX_VALUE, x2 = 0, y2 = 0;
            for (MatOfPoint c : contours) {
                Rect br = Imgproc.boundingRect(c);
                x1 = Math.min(x1, br.x);
                y1 = Math.min(y1, br.y);
                x2 = Math.max(x2, br.x + br.width);
                y2 = Math.max(y2, br.y + br.height);
                c.release();
            }
            return new Rect(safeWin.x + x1, safeWin.y + y1,
                            Math.max(1, x2 - x1), Math.max(1, y2 - y1));
        } catch (Exception e) {
            return window;
        }
    }

    // -------------------------------------------------------------------------
    // Image helpers
    // -------------------------------------------------------------------------

    /**
     * Converts a BGR image to CV_8UC1 greyscale, optionally masking out
     * non-foreground pixels (zeroing them) before returning.
     */
    private static Mat toGrey(Mat bgr, Mat mask) {
        Mat grey = new Mat();
        Imgproc.cvtColor(bgr, grey, Imgproc.COLOR_BGR2GRAY);
        if (mask != null && !mask.empty()) {
            Mat masked = new Mat(grey.size(), grey.type(), Scalar.all(0));
            grey.copyTo(masked, mask);
            grey.release();
            return masked;
        }
        return grey;
    }

    // -------------------------------------------------------------------------
    // Annotation writer
    // -------------------------------------------------------------------------

    private static Path writeAnnotated(Mat scene, Rect bbox, String variant, double score,
                                        ReferenceId refId, SceneEntry sceneEntry,
                                        Path outputDir) {
        try {
            Path dir  = outputDir.resolve("annotated").resolve(sanitise(variant));
            Files.createDirectories(dir);
            String sceneRef = sceneEntry.primaryReferenceId() != null
                    ? sanitise(sceneEntry.primaryReferenceId().name()) : "neg";
            String fname = sanitise(refId.name()) + "_vs_" + sceneRef
                    + "_" + sanitise(sceneEntry.variantLabel()) + ".png";
            Path   dest  = dir.resolve(fname);

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

    private static int    scenePx(SceneEntry s) { return s.sceneMat().cols() * s.sceneMat().rows(); }
    private static String sanitise(String s)    { return s.replaceAll("[^A-Za-z0-9_\\-]", "_"); }
    private static String shortName(String v)   {
        return v.replace("PIXEL_DIFF_CF_LOOSE", "PD·CFL")
                .replace("PIXEL_DIFF_CF_TIGHT", "PD·CFT")
                .replace("PIXEL_DIFF",          "PD");
    }
}



