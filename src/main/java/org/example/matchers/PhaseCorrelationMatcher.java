package org.example.matchers;

import org.example.analytics.AnalysisResult;
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
 * Phase Correlation technique — Milestone 13.
 *
 * <p>2 base variants × base / CF_LOOSE / CF_TIGHT = <b>6 variants total</b>:
 * <ul>
 *   <li>{@code PHASE_CORRELATE}         — FFT cross-correlation without windowing</li>
 *   <li>{@code PHASE_CORRELATE_HANNING} — same but with a Hanning window applied first,
 *       reducing spectral leakage from boundary discontinuities</li>
 * </ul>
 *
 * <h2>Pipeline</h2>
 * <ol>
 *   <li>Convert the 128×128 reference to greyscale float (CV_32F), normalised 0–1.</li>
 *   <li>Slide a reference-sized (128×128) window across the scene at stride 8.</li>
 *   <li>For each crop: compute the normalised cross-power spectrum via DFT, then IDFT
 *       to get the correlation surface; the peak value is the response confidence.</li>
 *   <li>Pick the crop with the highest peak response (0–1).</li>
 *   <li>Score = peak × 100.</li>
 *   <li>Bounding box = winning crop position. Sub-pixel shift from peak location refines it.</li>
 *   <li>CF variants zero out non-foreground pixels in the scene before correlation.</li>
 * </ol>
 *
 * <p><b>Note:</b> {@code Core.phaseCorrelate} is not exposed in the openpnp opencv-4.7.0-0
 * Java binding.  This class reimplements it manually using {@code Core.dft},
 * {@code Core.mulSpectrums}, {@code Core.idft}, and {@code Core.minMaxLoc}.
 */
public final class PhaseCorrelationMatcher {

    /** @deprecated Use {@link PhaseVariant#PHASE_CORRELATE}. */
    @Deprecated public static final String VAR_PLAIN   = PhaseVariant.PHASE_CORRELATE.variantName();
    /** @deprecated Use {@link PhaseVariant#PHASE_CORRELATE_HANNING}. */
    @Deprecated public static final String VAR_HANNING = PhaseVariant.PHASE_CORRELATE_HANNING.variantName();

    private static final int TILE   = 128;
    private static final int STRIDE = 8;

    private PhaseCorrelationMatcher() {}

    // -------------------------------------------------------------------------
    // Public entry point
    // -------------------------------------------------------------------------

    public static List<AnalysisResult> match(ReferenceId referenceId,
                                             Mat refMat,
                                             SceneEntry scene,
                                             Set<String> saveVariants,
                                             Path outputDir) {
        List<AnalysisResult> out = new ArrayList<>(6);
        Mat sceneMat = scene.sceneMat();

        // Reference as normalised greyscale float — masked to foreground only
        // so the black canvas does not contribute to the phase correlation peak.
        Mat refFgMask = ReferenceImageFactory.buildMask(refMat);
        Mat refFloat  = toNormFloat(refMat, refFgMask);
        refFgMask.release();

        // Hanning window — same size as tile, computed once
        Mat hanning = makeHanningWindow(TILE);

        // Pre-apply Hanning to the reference (windowed variant)
        Mat refWindowed = new Mat();
        Core.multiply(refFloat, hanning, refWindowed);

        // Scene float variants
        Mat sceneBase = toNormFloat(sceneMat, null);

        long t0 = System.currentTimeMillis();
        Mat looseMask  = ColourPreFilter.applyToScene(sceneMat, referenceId, ColourPreFilter.LOOSE);
        Mat sceneLoose = toNormFloat(sceneMat, looseMask);
        long cfLMs = System.currentTimeMillis() - t0;

        t0 = System.currentTimeMillis();
        Mat tightMask  = ColourPreFilter.applyToScene(sceneMat, referenceId, ColourPreFilter.TIGHT);
        Mat sceneTight = toNormFloat(sceneMat, tightMask);
        long cfTMs = System.currentTimeMillis() - t0;

        looseMask.release();
        tightMask.release();

        out.add(runVariant(VAR_PLAIN,              false, sceneMat, refFloat,    sceneBase,  hanning, 0L,    referenceId, scene, saveVariants, outputDir));
        out.add(runVariant(VAR_PLAIN + "_CF_LOOSE",false, sceneMat, refFloat,    sceneLoose, hanning, cfLMs, referenceId, scene, saveVariants, outputDir));
        out.add(runVariant(VAR_PLAIN + "_CF_TIGHT",false, sceneMat, refFloat,    sceneTight, hanning, cfTMs, referenceId, scene, saveVariants, outputDir));
        out.add(runVariant(VAR_HANNING,            true,  sceneMat, refWindowed, sceneBase,  hanning, 0L,    referenceId, scene, saveVariants, outputDir));
        out.add(runVariant(VAR_HANNING + "_CF_LOOSE",true,sceneMat, refWindowed, sceneLoose, hanning, cfLMs, referenceId, scene, saveVariants, outputDir));
        out.add(runVariant(VAR_HANNING + "_CF_TIGHT",true,sceneMat, refWindowed, sceneTight, hanning, cfTMs, referenceId, scene, saveVariants, outputDir));

        refFloat.release();
        refWindowed.release();
        hanning.release();
        sceneBase.release();
        sceneLoose.release();
        sceneTight.release();

        // ---- CF1 variants (PHASE_CORRELATE inside colour-first windows) ----
        // Recompute hanning for window-level correlation
        Mat hanningCf1 = makeHanningWindow(TILE);
        Mat refFloatCf1 = toNormFloat(refMat, ReferenceImageFactory.buildMask(refMat));

        for (PhaseVariant cf1 : new PhaseVariant[]{
                PhaseVariant.PHASE_CORRELATE_CF1_LOOSE,
                PhaseVariant.PHASE_CORRELATE_CF1_TIGHT}) {

            String cf1Name  = cf1.variantName();
            double tol      = cf1.cfMode().hueTolerance();
            long   cf1Start = System.currentTimeMillis();
            List<Rect> windows = ColourFirstLocator.propose(sceneMat, referenceId, tol);
            long cfMs = System.currentTimeMillis() - cf1Start;

            double bestScore = -1;
            Rect   bestBbox  = windows.get(0);
            Point  bestShift = new Point(0, 0);

            for (Rect w : windows) {
                Mat cropMask  = ColourPreFilter.applyToScene(new Mat(sceneMat, w), referenceId, tol);
                Mat cropFloat = toNormFloat(new Mat(sceneMat, w), cropMask);
                cropMask.release();
                AnalysisResult r = runVariant(cf1Name, false,
                        new Mat(sceneMat, w), refFloatCf1, cropFloat, hanningCf1,
                        cfMs, referenceId, scene, saveVariants, outputDir);
                cropFloat.release();
                if (r.matchScorePercent() > bestScore) {
                    bestScore = r.matchScorePercent();
                    Rect lb   = r.boundingRect();
                    if (lb != null)
                        bestBbox = new Rect(w.x + lb.x, w.y + lb.y, lb.width, lb.height);
                }
            }

            Path savedPath = null;
            if (saveVariants.contains(cf1Name)) {
                savedPath = writeAnnotated(sceneMat, bestBbox, bestShift,
                        cf1Name, Math.max(0, bestScore), referenceId, scene, outputDir);
            }
            out.add(new AnalysisResult(cf1Name, referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    Math.max(0, bestScore), bestBbox,
                    System.currentTimeMillis() - cf1Start, cfMs,
                    scenePx(scene), savedPath, false, null));
        }

        hanningCf1.release();
        refFloatCf1.release();

        return out;
    }

    // -------------------------------------------------------------------------
    // Single variant
    // -------------------------------------------------------------------------

    private static AnalysisResult runVariant(String variantName,
                                              boolean useHanning,
                                              Mat sceneMat,
                                              Mat refFloat,
                                              Mat sceneFloat,
                                              Mat hanning,
                                              long preFilterMs,
                                              ReferenceId referenceId,
                                              SceneEntry scene,
                                              Set<String> saveVariants,
                                              Path outputDir) {
        long t0 = System.currentTimeMillis();
        try {
            int sceneW = sceneFloat.cols();
            int sceneH = sceneFloat.rows();

            // Pre-compute DFT of reference (same for all crops)
            Mat refDft = dft(refFloat);

            double bestResponse = -1.0;
            Rect   bestBbox     = new Rect(0, 0, TILE, TILE);
            Point  bestShift    = new Point(0, 0);

            for (int y = 0; y <= sceneH - TILE; y += STRIDE) {
                for (int x = 0; x <= sceneW - TILE; x += STRIDE) {
                    Rect cropRect  = new Rect(x, y, TILE, TILE);
                    Mat  cropRaw   = sceneFloat.submat(cropRect);

                    // Apply Hanning window to crop if required
                    Mat cropFloat;
                    if (useHanning) {
                        cropFloat = new Mat();
                        Core.multiply(cropRaw, hanning, cropFloat);
                    } else {
                        cropFloat = cropRaw; // read-only submat — no copy needed
                    }

                    // Phase correlation: normalised cross-power spectrum → IDFT → peak
                    Mat cropDft = dft(cropFloat);
                    if (useHanning) cropFloat.release();

                    // Cross-power spectrum = conj(refDft) .* cropDft / |conj(refDft) .* cropDft|
                    Mat crossPow = crossPowerSpectrum(refDft, cropDft);
                    cropDft.release();

                    // IDFT of cross-power spectrum → correlation surface
                    Mat corrSurface = new Mat();
                    Core.idft(crossPow, corrSurface, Core.DFT_REAL_OUTPUT | Core.DFT_SCALE, 0);
                    crossPow.release();

                    // Find peak in correlation surface
                    Core.MinMaxLocResult mm = Core.minMaxLoc(corrSurface);
                    double response = mm.maxVal;   // ∈ [0,1] after normalisation
                    Point  peakLoc  = mm.maxLoc;
                    corrSurface.release();

                    if (response > bestResponse) {
                        bestResponse = response;
                        bestBbox     = cropRect;
                        // Convert peak location to sub-pixel shift (centred at TILE/2)
                        double dx = peakLoc.x > TILE / 2 ? peakLoc.x - TILE : peakLoc.x;
                        double dy = peakLoc.y > TILE / 2 ? peakLoc.y - TILE : peakLoc.y;
                        bestShift = new Point(dx, dy);
                    }
                }
            }
            refDft.release();

            long   elapsed = System.currentTimeMillis() - t0;
            double score   = Math.min(100.0, Math.max(0.0, bestResponse * 100.0));

            // Refine bbox using sub-pixel shift
            int cx = (int) Math.round(bestBbox.x + TILE / 2.0 + bestShift.x);
            int cy = (int) Math.round(bestBbox.y + TILE / 2.0 + bestShift.y);
            int bx = Math.max(0, cx - TILE / 2);
            int by = Math.max(0, cy - TILE / 2);
            int bw = Math.min(TILE, sceneMat.cols() - bx);
            int bh = Math.min(TILE, sceneMat.rows() - by);
            Rect refinedBbox = (bw > 1 && bh > 1) ? new Rect(bx, by, bw, bh) : bestBbox;

            Path savedPath = null;
            if (saveVariants.contains(variantName)) {
                savedPath = writeAnnotated(sceneMat, refinedBbox, bestShift, variantName,
                        score, referenceId, scene, outputDir);
            }

            return new AnalysisResult(variantName, referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    score, refinedBbox, elapsed, preFilterMs,
                    scenePx(scene), savedPath, false, null);

        } catch (Exception e) {
            return AnalysisResult.error(variantName, referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    System.currentTimeMillis() - t0, scenePx(scene), e.getMessage());
        }
    }

    // -------------------------------------------------------------------------
    // DFT helpers
    // -------------------------------------------------------------------------

    /** Forward DFT of a CV_32F single-channel mat → 2-channel complex result. */
    private static Mat dft(Mat src) {
        Mat result = new Mat();
        Core.dft(src, result, Core.DFT_COMPLEX_OUTPUT, 0);
        return result;
    }

    /**
     * Computes the normalised cross-power spectrum:
     *   G = conj(A) ⊙ B / |conj(A) ⊙ B|
     * Both inputs are complex (2-channel) DFT outputs.
     */
    private static Mat crossPowerSpectrum(Mat dftA, Mat dftB) {
        // mulSpectrums(conj(A), B) — conjA flag
        Mat product = new Mat();
        Core.mulSpectrums(dftA, dftB, product, 0, true);   // true = conjugate first arg

        // Compute magnitude of product
        List<Mat> channels = new ArrayList<>(2);
        Core.split(product, channels);
        Mat mag = new Mat();
        Core.magnitude(channels.get(0), channels.get(1), mag);
        // Add small epsilon to avoid div-by-zero
        Core.add(mag, new Scalar(1e-10), mag);

        // Normalise: divide real and imag by magnitude
        for (Mat ch : channels) {
            Core.divide(ch, mag, ch);
        }
        mag.release();

        Mat normalised = new Mat();
        Core.merge(channels, normalised);
        product.release();
        channels.forEach(Mat::release);
        return normalised;
    }

    /**
     * Creates a 2-D Hanning (von Hann) window of size {@code sz × sz}.
     * Uses the separable 1-D formula: w[i] = 0.5 × (1 − cos(2π·i/(N−1))).
     */
    private static Mat makeHanningWindow(int sz) {
        // Build as CV_32F via Imgproc if available, else manual
        Mat win = new Mat(sz, sz, CvType.CV_32F);
        for (int r = 0; r < sz; r++) {
            double wr = 0.5 * (1.0 - Math.cos(2.0 * Math.PI * r / (sz - 1)));
            for (int c = 0; c < sz; c++) {
                double wc = 0.5 * (1.0 - Math.cos(2.0 * Math.PI * c / (sz - 1)));
                win.put(r, c, (float)(wr * wc));
            }
        }
        return win;
    }

    // -------------------------------------------------------------------------
    // Image helpers
    // -------------------------------------------------------------------------

    private static Mat toNormFloat(Mat bgr, Mat mask) {
        Mat work = bgr;
        if (mask != null && !mask.empty()) {
            work = new Mat(bgr.size(), bgr.type(), Scalar.all(0));
            bgr.copyTo(work, mask);
        }
        Mat grey    = new Mat();
        Mat float32 = new Mat();
        Imgproc.cvtColor(work, grey, Imgproc.COLOR_BGR2GRAY);
        if (mask != null && !mask.empty()) work.release();
        grey.convertTo(float32, CvType.CV_32F, 1.0 / 255.0);
        grey.release();
        return float32;
    }

    // -------------------------------------------------------------------------
    // Annotation writer
    // -------------------------------------------------------------------------

    private static Path writeAnnotated(Mat scene, Rect bbox, Point shift,
                                        String variant, double score,
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

            if (bbox.width > 1 && bbox.height > 1) {
                Imgproc.rectangle(m, new Point(bbox.x, bbox.y),
                        new Point(bbox.x + bbox.width, bbox.y + bbox.height), colour, 2);
            }
            // Crosshair at detected centre
            int cx = bbox.x + bbox.width  / 2;
            int cy = bbox.y + bbox.height / 2;
            Imgproc.line(m, new Point(cx - 6, cy), new Point(cx + 6, cy), colour, 1);
            Imgproc.line(m, new Point(cx, cy - 6), new Point(cx, cy + 6), colour, 1);

            Imgproc.putText(m, shortName(variant),
                    new Point(4, 13), Imgproc.FONT_HERSHEY_SIMPLEX, 0.32,
                    new Scalar(200, 200, 200), 1);
            Imgproc.putText(m, String.format("%.1f%%", score),
                    new Point(4, 28), Imgproc.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1);
            Imgproc.putText(m, String.format("dx=%.1f dy=%.1f", shift.x, shift.y),
                    new Point(4, 43), Imgproc.FONT_HERSHEY_SIMPLEX, 0.30,
                    new Scalar(160, 160, 160), 1);

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

    private static Scalar scoreColour(double score) {
        return score >= 70 ? new Scalar(0, 200, 0)
             : score >= 40 ? new Scalar(0, 200, 200)
             :               new Scalar(0, 0, 200);
    }

    private static int    scenePx(SceneEntry s) { return s.sceneMat().cols() * s.sceneMat().rows(); }
    private static String sanitise(String s)    { return s.replaceAll("[^A-Za-z0-9_\\-]", "_"); }
    private static String shortName(String v)   {
        return v.replace("PHASE_CORRELATE_HANNING", "PC_HAN")
                .replace("PHASE_CORRELATE",         "PC")
                .replace("_CF_LOOSE", "·CF·L")
                .replace("_CF_TIGHT", "·CF·T");
    }
}
