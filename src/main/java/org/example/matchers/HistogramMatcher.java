package org.example.matchers;

import org.example.*;
import org.example.CfMode;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * Histogram Comparison technique — Milestone 12.
 *
 * <p>4 base variants × base / CF_LOOSE / CF_TIGHT = <b>12 variants total</b>:
 * <ul>
 *   <li>{@code HISTCMP_CORREL}        — correlation  (-1..1,  higher = better)</li>
 *   <li>{@code HISTCMP_CHISQR}        — chi-square   (0..∞,   lower  = better)</li>
 *   <li>{@code HISTCMP_INTERSECT}     — intersection (0..sum, higher = better)</li>
 *   <li>{@code HISTCMP_BHATTACHARYYA} — Bhattacharyya(0..1,   lower  = better)</li>
 * </ul>
 *
 * <h2>Pipeline</h2>
 * <ol>
 *   <li>Convert reference and scene to HSV colour space.</li>
 *   <li>Compute a 2-D H-S histogram for the reference (normalised to [0,1]).</li>
 *   <li>Slide a reference-sized window across the scene; compute the H-S histogram
 *       for each crop and compare with {@link Imgproc#compareHist}.</li>
 *   <li>The crop with the best score is the detection.  Score is normalised to 0–100%.</li>
 *   <li>CF variants apply a colour-pre-filter mask to the scene HSV before histogramming,
 *       isolating only pixels of the expected foreground colour.</li>
 * </ol>
 *
 * <p><b>Normalisation per method:</b>
 * <ul>
 *   <li>CORREL:        score = (bestVal + 1) / 2 × 100   (maps -1..1  → 0..100%)</li>
 *   <li>CHISQR:        score = 1 / (1 + bestVal) × 100   (maps 0..∞   → 100..0%)</li>
 *   <li>INTERSECT:     score = bestVal / refHistSum × 100 (normalised by ref magnitude)</li>
 *   <li>BHATTACHARYYA: score = (1 – bestVal) × 100       (maps 0..1   → 100..0%)</li>
 * </ul>
 *
 * <p><b>Spatial note:</b> pure histogram comparison is spatially invariant — two images
 * with different shapes but identical colour distributions score identically.
 * The sliding-window approach provides localisation but does not fully remove this
 * ambiguity; the report's Base vs CF tab illustrates where CF dramatically narrows
 * the colour distribution, making the comparison far more discriminative.
 */
public final class HistogramMatcher {

    // -------------------------------------------------------------------------
    // Variant name constants
    // -------------------------------------------------------------------------
    /** @deprecated Use {@link HistVariant#HISTCMP_CORREL}. */
    @Deprecated public static final String VAR_CORREL    = HistVariant.HISTCMP_CORREL.variantName();
    /** @deprecated Use {@link HistVariant#HISTCMP_CHISQR}. */
    @Deprecated public static final String VAR_CHISQR    = HistVariant.HISTCMP_CHISQR.variantName();
    /** @deprecated Use {@link HistVariant#HISTCMP_INTERSECT}. */
    @Deprecated public static final String VAR_INTERSECT = HistVariant.HISTCMP_INTERSECT.variantName();
    /** @deprecated Use {@link HistVariant#HISTCMP_BHATTACHARYYA}. */
    @Deprecated public static final String VAR_BHATTA    = HistVariant.HISTCMP_BHATTACHARYYA.variantName();

    // Sliding-window step size — stride of 8px keeps runtime reasonable
    private static final int STRIDE = 8;

    // HSV histogram bins: H=50, S=60 (ignoring V — colour shape only)
    private static final int   H_BINS  = 50;
    private static final int   S_BINS  = 60;
    private static final float H_RANGE_LO = 0f, H_RANGE_HI = 180f;
    private static final float S_RANGE_LO = 0f, S_RANGE_HI = 256f;

    private HistogramMatcher() {}

    // -------------------------------------------------------------------------
    // Public entry point
    // -------------------------------------------------------------------------

    public static List<AnalysisResult> match(ReferenceId referenceId,
                                              Mat refMat,
                                              SceneEntry scene,
                                              Set<String> saveVariants,
                                              Path outputDir) {
        List<AnalysisResult> out = new ArrayList<>(12);
        Mat sceneMat = scene.sceneMat();

        // Reference histogram (normalised) — masked to foreground only so that
        // black background pixels do not skew the colour distribution.
        Mat refHsv  = toHsv(refMat);
        Mat refFgMask = ReferenceImageFactory.buildMask(refMat);
        Mat refHist = calcHSHist(refHsv, refFgMask);
        double refHistSum = Core.sumElems(refHist).val[0]; // for INTERSECT normalisation
        refHsv.release();
        refFgMask.release();

        // Scene HSV: base, CF_LOOSE, CF_TIGHT
        Mat sceneHsv = toHsv(sceneMat);

        long t0 = System.currentTimeMillis();
        Mat looseMask     = ColourPreFilter.applyToScene(sceneMat, referenceId, ColourPreFilter.LOOSE);
        long cfLMs = System.currentTimeMillis() - t0;

        t0 = System.currentTimeMillis();
        Mat tightMask     = ColourPreFilter.applyToScene(sceneMat, referenceId, ColourPreFilter.TIGHT);
        long cfTMs = System.currentTimeMillis() - t0;

        // Run all 4 base methods, each in base / CF_LOOSE / CF_TIGHT mode
        HistVariant[] bases = {
            HistVariant.HISTCMP_CORREL,
            HistVariant.HISTCMP_CHISQR,
            HistVariant.HISTCMP_INTERSECT,
            HistVariant.HISTCMP_BHATTACHARYYA
        };
        for (HistVariant base : bases) {
            out.add(runVariant(base.variantName(), base.cvMethod(), sceneMat, sceneHsv, null,
                    refHist, refHistSum, 0L, referenceId, scene, saveVariants, outputDir));
            out.add(runVariant(base.variantName() + CfMode.LOOSE.suffix(), base.cvMethod(), sceneMat, sceneHsv, looseMask,
                    refHist, refHistSum, cfLMs, referenceId, scene, saveVariants, outputDir));
            out.add(runVariant(base.variantName() + CfMode.TIGHT.suffix(), base.cvMethod(), sceneMat, sceneHsv, tightMask,
                    refHist, refHistSum, cfTMs, referenceId, scene, saveVariants, outputDir));
        }

        refHist.release();
        sceneHsv.release();
        looseMask.release();
        tightMask.release();

        // ---- CF1 variants (HISTCMP_CORREL inside colour-first windows) ----
        for (HistVariant cf1 : new HistVariant[]{
                HistVariant.HISTCMP_CORREL_CF1_LOOSE,
                HistVariant.HISTCMP_CORREL_CF1_TIGHT}) {

            String cf1Name  = cf1.variantName();
            double tol      = cf1.cfMode().hueTolerance();
            long   cf1Start = System.currentTimeMillis();
            List<Rect> windows = ColourFirstLocator.propose(sceneMat, referenceId, tol);
            long cfMs = System.currentTimeMillis() - cf1Start;

            // Recompute ref histogram (already released above)
            Mat rHsv     = toHsv(refMat);
            Mat rFgMask  = ReferenceImageFactory.buildMask(refMat);
            Mat rHist    = calcHSHist(rHsv, rFgMask);
            double rHistSum = Core.sumElems(rHist).val[0];
            rHsv.release(); rFgMask.release();

            double bestScore = -1;
            Rect   bestBbox  = windows.get(0);

            for (Rect w : windows) {
                Mat cropBgr    = new Mat(sceneMat, w);
                Mat cropMask   = ColourPreFilter.applyToScene(cropBgr, referenceId, tol);
                Mat cropHsv    = toHsv(cropBgr);
                AnalysisResult r = runVariant(cf1Name, Imgproc.HISTCMP_CORREL,
                        cropBgr, cropHsv, cropMask,
                        rHist, rHistSum, cfMs, referenceId, scene, saveVariants, outputDir);
                cropMask.release(); cropHsv.release();
                if (r.matchScorePercent() > bestScore) {
                    bestScore = r.matchScorePercent();
                    Rect lb   = r.boundingRect();
                    if (lb != null)
                        bestBbox = new Rect(w.x + lb.x, w.y + lb.y, lb.width, lb.height);
                }
            }
            rHist.release();

            Path savedPath = null;
            if (saveVariants.contains(cf1Name)) {
                savedPath = writeAnnotated(sceneMat, bestBbox, cf1Name,
                        Math.max(0, bestScore), referenceId, scene, outputDir);
            }
            out.add(new AnalysisResult(cf1Name, referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    Math.max(0, bestScore), bestBbox,
                    System.currentTimeMillis() - cf1Start, cfMs,
                    scenePx(scene), savedPath, false, null));
        }

        return out;
    }

    // -------------------------------------------------------------------------
    // Single variant — sliding window histogram comparison
    // -------------------------------------------------------------------------

    private static AnalysisResult runVariant(String variantName,
                                              int cvMethod,
                                              Mat sceneMat,
                                              Mat sceneHsv,
                                              Mat mask,          // may be null (base)
                                              Mat refHist,
                                              double refHistSum,
                                              long preFilterMs,
                                              ReferenceId referenceId,
                                              SceneEntry scene,
                                              Set<String> saveVariants,
                                              Path outputDir) {
        long t0 = System.currentTimeMillis();
        try {
            int sceneW = sceneMat.cols();
            int sceneH = sceneMat.rows();
            int refW   = 128;  // reference is always 128×128
            int refH   = 128;

            // For CF variants, compute a masked scene HSV once
            Mat searchHsv;
            if (mask != null && !mask.empty()) {
                searchHsv = new Mat(sceneHsv.size(), sceneHsv.type(), Scalar.all(0));
                sceneHsv.copyTo(searchHsv, mask);
            } else {
                searchHsv = sceneHsv; // no copy needed — read-only
            }

            boolean higherIsBetter = (cvMethod == Imgproc.HISTCMP_CORREL
                    || cvMethod == Imgproc.HISTCMP_INTERSECT); // used for direction reference
            double bestScore = -Double.MAX_VALUE; // we track final 0-100 score
            Rect   bestBbox  = new Rect(0, 0, refW, refH);

            for (int y = 0; y <= sceneH - refH; y += STRIDE) {
                for (int x = 0; x <= sceneW - refW; x += STRIDE) {
                    Rect  crop     = new Rect(x, y, refW, refH);
                    Mat   cropHsv  = searchHsv.submat(crop);
                    Mat   cropHist = calcHSHist(cropHsv, null);

                    double raw   = Imgproc.compareHist(refHist, cropHist, cvMethod);
                    double score = toScore(raw, cvMethod, refHistSum);
                    cropHist.release();

                    if (score > bestScore) {
                        bestScore = score;
                        bestBbox  = crop;
                    }
                }
            }

            if (mask != null && !mask.empty()) searchHsv.release();

            long elapsed = System.currentTimeMillis() - t0;
            double finalScore = Math.max(0, Math.min(100, bestScore));

            // Tighten bestBbox to the actual matched pixels within the winning window.
            // For CF variants the colour mask already isolates the shape pixels;
            // for base variants we threshold the grey crop to find foreground pixels.
            Rect tightBbox = tightenBbox(sceneMat, mask, bestBbox);

            Path savedPath = null;
            if (saveVariants.contains(variantName)) {
                savedPath = writeAnnotated(sceneMat, tightBbox, variantName, finalScore,
                        referenceId, scene, outputDir);
            }

            return new AnalysisResult(variantName, referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    finalScore, tightBbox, elapsed, preFilterMs,
                    scenePx(scene), savedPath, false, null);

        } catch (Exception e) {
            return AnalysisResult.error(variantName, referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    System.currentTimeMillis() - t0, scenePx(scene), e.getMessage());
        }
    }

    // -------------------------------------------------------------------------
    // Score normalisation
    // -------------------------------------------------------------------------

    /**
     * Maps a raw {@code compareHist} value to 0–100%, with 100% = best match.
     */
    private static double toScore(double raw, int method, double refHistSum) {
        return switch (method) {
            // CORREL: -1..1, higher = better → (val+1)/2 × 100
            case Imgproc.HISTCMP_CORREL        -> (raw + 1.0) / 2.0 * 100.0;
            // CHISQR: 0..∞, lower = better → 1/(1+val) × 100
            case Imgproc.HISTCMP_CHISQR        -> 1.0 / (1.0 + raw) * 100.0;
            // INTERSECT: 0..sum, higher = better → val/refSum × 100
            case Imgproc.HISTCMP_INTERSECT     -> refHistSum > 0
                    ? (raw / refHistSum) * 100.0 : 0.0;
            // BHATTACHARYYA: 0..1, lower = better → (1-val) × 100
            case Imgproc.HISTCMP_BHATTACHARYYA -> (1.0 - raw) * 100.0;
            default                            -> 0.0;
        };
    }

    // -------------------------------------------------------------------------
    // Histogram helpers
    // -------------------------------------------------------------------------

    /**
     * Computes a normalised 2-D Hue–Saturation histogram from an HSV image.
     * The V (value/brightness) channel is ignored — pure colour shape comparison.
     *
     * @param hsv  source HSV image (CV_8UC3)
     * @param mask optional single-channel mask (may be null)
     * @return normalised histogram Mat
     */
    private static Mat calcHSHist(Mat hsv, Mat mask) {
        List<Mat>    images   = List.of(hsv);
        MatOfInt     channels = new MatOfInt(0, 1);          // H and S channels
        MatOfInt     histSize = new MatOfInt(H_BINS, S_BINS);
        MatOfFloat   ranges   = new MatOfFloat(
                H_RANGE_LO, H_RANGE_HI, S_RANGE_LO, S_RANGE_HI);
        Mat hist = new Mat();
        Imgproc.calcHist(images, channels, mask == null ? new Mat() : mask,
                hist, histSize, ranges, false);
        Core.normalize(hist, hist, 0, 1, Core.NORM_MINMAX);
        channels.release();
        histSize.release();
        ranges.release();
        return hist;
    }

    private static Mat toHsv(Mat bgr) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(bgr, hsv, Imgproc.COLOR_BGR2HSV);
        return hsv;
    }

    // -------------------------------------------------------------------------
    // Bbox tightening
    // -------------------------------------------------------------------------

    /**
     * Tightens a candidate window bbox to the actual foreground pixels within it.
     *
     * <p>For CF variants the colour mask already isolates the foreground pixels —
     * we crop the mask to the window and find the bounding rect of non-zero pixels.
     * For base variants we convert the window crop to grey and threshold it.
     * If no foreground pixels are found the original window is returned unchanged.
     */
    private static Rect tightenBbox(Mat sceneBGR, Mat colourMask, Rect window) {
        try {
            // Clamp window to scene bounds before taking submats
            int sceneW = sceneBGR.cols(), sceneH = sceneBGR.rows();
            int wx = Math.max(0, Math.min(window.x, sceneW - 1));
            int wy = Math.max(0, Math.min(window.y, sceneH - 1));
            int ww = Math.min(window.width,  sceneW - wx);
            int wh = Math.min(window.height, sceneH - wy);
            if (ww <= 0 || wh <= 0) return window;
            Rect safeWindow = new Rect(wx, wy, ww, wh);

            Mat fgMask;
            if (colourMask != null && !colourMask.empty()) {
                fgMask = new Mat(colourMask, safeWindow).clone();
            } else {
                Mat grey = new Mat();
                Mat crop = new Mat(sceneBGR, safeWindow);
                Imgproc.cvtColor(crop, grey, Imgproc.COLOR_BGR2GRAY);
                fgMask   = new Mat();
                Imgproc.threshold(grey, fgMask, 20, 255, Imgproc.THRESH_BINARY);
                grey.release();
            }

            // Find bounding rect of non-zero pixels in fgMask (local coords)
            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(fgMask.clone(), contours, hierarchy,
                    Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            hierarchy.release();
            fgMask.release();

            if (contours.isEmpty()) return window;

            // Union all contour bounding rects
            int x1 = Integer.MAX_VALUE, y1 = Integer.MAX_VALUE, x2 = 0, y2 = 0;
            for (MatOfPoint c : contours) {
                Rect br = Imgproc.boundingRect(c);
                x1 = Math.min(x1, br.x);
                y1 = Math.min(y1, br.y);
                x2 = Math.max(x2, br.x + br.width);
                y2 = Math.max(y2, br.y + br.height);
            }

            // Convert from window-local to scene-global coordinates
            int gx = safeWindow.x + x1;
            int gy = safeWindow.y + y1;
            int gw = Math.max(1, x2 - x1);
            int gh = Math.max(1, y2 - y1);
            return new Rect(gx, gy, gw, gh);

        } catch (Exception e) {
            return window; // fall back to the original window on any error
        }
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
            Scalar colour = scoreColour(score);

            // Draw bounding rect around the tightened match region
            if (bbox.width > 1 && bbox.height > 1) {
                Imgproc.rectangle(m,
                        new Point(bbox.x, bbox.y),
                        new Point(bbox.x + bbox.width, bbox.y + bbox.height),
                        colour, 2);
            }
            // Label: variant name + score at top-left corner of image (consistent with other matchers)
            Imgproc.putText(m, shortName(variant),
                    new Point(4, 13), Imgproc.FONT_HERSHEY_SIMPLEX, 0.32,
                    new Scalar(200, 200, 200), 1);
            Imgproc.putText(m, String.format("%.1f%%", score),
                    new Point(4, 28), Imgproc.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1);

            Imgcodecs.imwrite(dest.toString(), m);
            m.release();
            return outputDir.toAbsolutePath().relativize(dest.toAbsolutePath());
        } catch (Exception e) {
            System.err.printf("[HM] writeAnnotated failed for %s/%s: %s%n",
                    variant, sceneEntry.variantLabel(), e.getMessage());
            return null;
        }
    }

    // -------------------------------------------------------------------------
    // Tiny helpers
    // -------------------------------------------------------------------------

    private static Scalar scoreColour(double score) {
        return score >= 70 ? new Scalar(0, 200, 0)
             : score >= 40 ? new Scalar(0, 200, 200)
             :               new Scalar(0, 0, 200);
    }

    private static int    scenePx(SceneEntry s) { return s.sceneMat().cols() * s.sceneMat().rows(); }
    private static String sanitise(String s)    { return s.replaceAll("[^A-Za-z0-9_\\-]", "_"); }
    private static String shortName(String v)   {
        return v.replace("HISTCMP_CORREL",        "HC_COR")
                .replace("HISTCMP_CHISQR",        "HC_CHI")
                .replace("HISTCMP_INTERSECT",     "HC_INT")
                .replace("HISTCMP_BHATTACHARYYA", "HC_BHA")
                .replace("_CF_", "·CF·");
    }
}








