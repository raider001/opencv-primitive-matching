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
 * Structural Similarity Index (SSIM) sliding-window matcher — Milestone 17.
 *
 * <p>1 base variant × base / CF_LOOSE / CF_TIGHT = <b>3 variants total</b>.
 *
 * <h2>Algorithm</h2>
 * <p>SSIM decomposes similarity into three independent components:
 * <ul>
 *   <li><b>Luminance</b>  — {@code (2μxμy + C1) / (μx² + μy² + C1)}</li>
 *   <li><b>Contrast</b>   — {@code (2σxσy + C2) / (σx² + σy² + C2)}</li>
 *   <li><b>Structure</b>  — {@code (σxy + C3) / (σxσy + C3)}</li>
 * </ul>
 * The combined SSIM = luminance × contrast × structure ∈ [−1, 1], where 1 is a
 * perfect match.  {@code C1 = (K1·L)²}, {@code C2 = (K2·L)²}, {@code C3 = C2/2},
 * with {@code K1=0.01}, {@code K2=0.03}, {@code L=255}.
 *
 * <p>Each component is computed using local Gaussian-weighted statistics via a
 * {@code Imgproc.GaussianBlur} approximation, then the per-pixel SSIM map is
 * averaged over the whole window.
 *
 * <h2>Sliding window</h2>
 * A 128×128 window slides across the scene at stride {@value #STRIDE}.
 * The window with the highest mean SSIM score is the detection.
 *
 * <h2>CF variants</h2>
 * Before SSIM computation, the colour pre-filter zeroes all pixels outside the
 * reference foreground colour range.  SSIM then compares only the colour-matching
 * structure in both images.
 *
 * <h2>Expected behaviour</h2>
 * <ul>
 *   <li>A_CLEAN — very high score (SSIM is near 1.0 for exact matches)</li>
 *   <li>B_TRANSFORMED — score drops sharply under scale/rotation (SSIM is not invariant)</li>
 *   <li>C_DEGRADED — degrades gracefully with noise/blur — more slowly than raw absdiff</li>
 *   <li>D_NEGATIVE — non-zero but substantially lower than a true positive; CF reduces FP</li>
 * </ul>
 */
public final class SsimMatcher {

    // SSIM stability constants
    private static final double K1 = 0.01;
    private static final double K2 = 0.03;
    private static final double L  = 255.0;
    private static final double C1 = (K1 * L) * (K1 * L);   // 6.5025
    private static final double C2 = (K2 * L) * (K2 * L);   // 58.5225
    private static final double C3 = C2 / 2.0;               // 29.26125

    /** Gaussian blur kernel size for local statistics. */
    private static final int    SIGMA_PX = 11;
    private static final double SIGMA    = 1.5;

    /** Reference tile size — must match all reference images. */
    private static final int TILE   = 128;
    /** Sliding window stride in pixels. */
    private static final int STRIDE = 8;

    private SsimMatcher() {}

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

        // Base reference grey
        Mat refGrey = toGrey(refMat, null);

        // CF masks + timings
        long t0 = System.currentTimeMillis();
        Mat looseMask = ColourPreFilter.applyToScene(sceneMat, referenceId, ColourPreFilter.LOOSE);
        long cfLMs    = System.currentTimeMillis() - t0;

        t0 = System.currentTimeMillis();
        Mat tightMask = ColourPreFilter.applyToScene(sceneMat, referenceId, ColourPreFilter.TIGHT);
        long cfTMs    = System.currentTimeMillis() - t0;

        // Masked reference greys
        Mat refMaskL     = ColourPreFilter.applyToReference(refMat, referenceId, ColourPreFilter.LOOSE);
        Mat refGreyLoose = toGrey(refMat, refMaskL);
        refMaskL.release();

        Mat refMaskT     = ColourPreFilter.applyToReference(refMat, referenceId, ColourPreFilter.TIGHT);
        Mat refGreyTight = toGrey(refMat, refMaskT);
        refMaskT.release();

        // Scene grey variants
        Mat sceneGrey      = toGrey(sceneMat, null);
        Mat sceneGreyLoose = toGrey(sceneMat, looseMask);
        Mat sceneGreyTight = toGrey(sceneMat, tightMask);
        looseMask.release();
        tightMask.release();

        out.add(runVariant(SsimVariant.SSIM.variantName(),
                sceneMat, sceneGrey,      refGrey,       0L,    referenceId, scene, saveVariants, outputDir));
        out.add(runVariant(SsimVariant.SSIM_CF_LOOSE.variantName(),
                sceneMat, sceneGreyLoose, refGreyLoose,  cfLMs, referenceId, scene, saveVariants, outputDir));
        out.add(runVariant(SsimVariant.SSIM_CF_TIGHT.variantName(),
                sceneMat, sceneGreyTight, refGreyTight,  cfTMs, referenceId, scene, saveVariants, outputDir));

        refGrey.release();
        refGreyLoose.release();
        refGreyTight.release();
        sceneGrey.release();
        sceneGreyLoose.release();
        sceneGreyTight.release();

        return out;
    }

    // -------------------------------------------------------------------------
    // Single variant — sliding window SSIM
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

            // Pre-compute reference statistics needed for SSIM
            Mat refF = toFloat(refGrey);
            Mat muX  = gaussBlur(refF);
            Mat muX2 = sqr(muX);
            Mat refF2 = sqr(refF);
            Mat sigmaX2mat = new Mat();
            Core.subtract(gaussBlur(refF2), muX2, sigmaX2mat);
            refF2.release();
            // σx: mean std dev over the reference tile
            double sigmaX2 = Core.mean(sigmaX2mat).val[0];
            double muXVal   = Core.mean(muX).val[0];
            muX.release();
            muX2.release();
            sigmaX2mat.release();
            refF.release();

            double bestScore = -1;
            Rect   bestBbox  = new Rect(0, 0, TILE, TILE);

            for (int y = 0; y <= sceneH - TILE; y += STRIDE) {
                for (int x = 0; x <= sceneW - TILE; x += STRIDE) {
                    Mat crop = new Mat(sceneGrey, new Rect(x, y, TILE, TILE));
                    double score = computeSsim(refGrey, crop, muXVal, sigmaX2);
                    crop.release();
                    if (score > bestScore) {
                        bestScore = score;
                        bestBbox  = new Rect(x, y, TILE, TILE);
                    }
                }
            }

            long   elapsed    = System.currentTimeMillis() - t0;
            // Map SSIM [-1,1] → [0,100]%
            double finalScore = Math.max(0, Math.min(100, ((bestScore + 1.0) / 2.0) * 100.0));

            Rect tightBbox = tightenBbox(sceneMat, bestBbox);

            Path savedPath = null;
            if (saveVariants.contains(variantName)) {
                savedPath = writeAnnotated(sceneMat, tightBbox, variantName, finalScore,
                        referenceId, scene, outputDir);
            }

            return new AnalysisResult(variantName, referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    finalScore, tightBbox, elapsed, preFilterMs,
                    scenePx(scene), savedPath, false, null,
                    AnalysisResult.ScoringLayers.ZERO);

        } catch (Exception e) {
            return AnalysisResult.error(variantName, referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    System.currentTimeMillis() - t0, scenePx(scene), e.getMessage());
        }
    }

    // -------------------------------------------------------------------------
    // SSIM computation for one (reference tile, scene crop) pair
    // -------------------------------------------------------------------------

    /**
     * Computes mean SSIM between two same-size greyscale tiles.
     * Uses the reference's pre-computed μx and σx² to avoid redundant computation
     * across the many crop comparisons.
     *
     * @return mean SSIM ∈ [−1, 1]
     */
    private static double computeSsim(Mat ref, Mat crop,
                                       double muXVal, double sigmaX2) {
        Mat cropF = toFloat(crop);
        Mat muY   = gaussBlur(cropF);
        double muYVal = Core.mean(muY).val[0];

        Mat cropF2   = sqr(cropF);
        Mat muY2     = sqr(muY);
        Mat sigmaY2m = new Mat();
        Core.subtract(gaussBlur(cropF2), muY2, sigmaY2m);
        cropF2.release();
        muY2.release();
        double sigmaY2 = Core.mean(sigmaY2m).val[0];
        sigmaY2m.release();

        // Cross-correlation σxy
        Mat refF   = toFloat(ref);
        Mat xy     = new Mat();
        Core.multiply(refF, cropF, xy);
        refF.release();
        cropF.release();
        Mat muXY2  = new Mat();
        Core.multiply(gaussBlur(ref.clone()), muY, muXY2);   // approx μx·μy per pixel
        Mat sigmaXYm = new Mat();
        Core.subtract(gaussBlur(xy), muXY2, sigmaXYm);
        xy.release();
        muXY2.release();
        muY.release();
        double sigmaXY = Core.mean(sigmaXYm).val[0];
        sigmaXYm.release();

        double sigmaX = Math.sqrt(Math.max(0, sigmaX2));
        double sigmaY = Math.sqrt(Math.max(0, sigmaY2));

        double luminance = (2.0 * muXVal * muYVal + C1)
                         / (muXVal * muXVal + muYVal * muYVal + C1);
        double contrast  = (2.0 * sigmaX * sigmaY + C2)
                         / (sigmaX2 + sigmaY2 + C2);
        double structure = (sigmaXY + C3)
                         / (sigmaX * sigmaY + C3);

        return luminance * contrast * structure;
    }

    // -------------------------------------------------------------------------
    // Mat math helpers
    // -------------------------------------------------------------------------

    /** Converts an 8U single-channel Mat to 32F. */
    private static Mat toFloat(Mat src) {
        Mat dst = new Mat();
        src.convertTo(dst, CvType.CV_32F);
        return dst;
    }

    /** Returns element-wise square of a float Mat. */
    private static Mat sqr(Mat m) {
        Mat out = new Mat();
        Core.multiply(m, m, out);
        return out;
    }

    /** Gaussian blur approximating local statistics. */
    private static Mat gaussBlur(Mat m) {
        Mat out = new Mat();
        Imgproc.GaussianBlur(m, out, new Size(SIGMA_PX, SIGMA_PX), SIGMA);
        return out;
    }

    /** BGR → grey, optionally masked (non-mask pixels zeroed). */
    private static Mat toGrey(Mat bgr, Mat mask) {
        Mat grey = new Mat();
        if (bgr.channels() == 1) {
            bgr.copyTo(grey);
        } else {
            Imgproc.cvtColor(bgr, grey, Imgproc.COLOR_BGR2GRAY);
        }
        if (mask != null && !mask.empty()) {
            Mat masked = new Mat(grey.size(), grey.type(), Scalar.all(0));
            grey.copyTo(masked, mask);
            grey.release();
            return masked;
        }
        return grey;
    }

    // -------------------------------------------------------------------------
    // Bbox tightening
    // -------------------------------------------------------------------------

    private static Rect tightenBbox(Mat sceneBGR, Rect window) {
        try {
            int sceneW = sceneBGR.cols(), sceneH = sceneBGR.rows();
            int wx = Math.max(0, window.x);
            int wy = Math.max(0, window.y);
            int ww = Math.min(window.width,  sceneW - wx);
            int wh = Math.min(window.height, sceneH - wy);
            if (ww <= 0 || wh <= 0) return window;

            Mat crop = new Mat(sceneBGR, new Rect(wx, wy, ww, wh));
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
                x1 = Math.min(x1, br.x); y1 = Math.min(y1, br.y);
                x2 = Math.max(x2, br.x + br.width);
                y2 = Math.max(y2, br.y + br.height);
                c.release();
            }
            return new Rect(wx + x1, wy + y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1));
        } catch (Exception e) {
            return window;
        }
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
            String sceneRef = sceneEntry.primaryReferenceId() != null
                    ? sanitise(sceneEntry.primaryReferenceId().name()) : "neg";
            String fname = sanitise(refId.name()) + "_vs_" + sceneRef
                    + "_" + sanitise(sceneEntry.variantLabel()) + ".png";
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

    private static int    scenePx(SceneEntry s) { return s.sceneMat().cols() * s.sceneMat().rows(); }
    private static String sanitise(String s)    { return s.replaceAll("[^A-Za-z0-9_\\-]", "_"); }
    private static String shortName(String v)   {
        return v.replace("SSIM_CF_LOOSE", "SSIM·CFL")
                .replace("SSIM_CF_TIGHT", "SSIM·CFT");
    }
}


