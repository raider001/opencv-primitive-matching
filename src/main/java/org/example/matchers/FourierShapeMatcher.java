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
 * Fourier Shape Descriptor matcher — Milestone 19.
 *
 * <p>1 base variant x base / CF_LOOSE / CF_TIGHT = 3 variants total.
 *
 * <h2>Algorithm</h2>
 * <ol>
 *   <li>Extract the largest contour from a binarised image.</li>
 *   <li>Resample the contour uniformly to {@value #N_SAMPLES} points.</li>
 *   <li>Encode as a 1-D complex signal: centroid-normalised (x + jy).</li>
 *   <li>Apply a 1-D DFT via {@link Core#dft}.</li>
 *   <li>Compute magnitude spectrum; keep first {@value #N_DESCRIPTORS} coefficients
 *       (DC excluded), normalised by coefficient k=1 for scale invariance.</li>
 *   <li>Rotation invariance is automatic — rotation is a global phase shift that
 *       vanishes in the magnitude spectrum.</li>
 *   <li>Compare descriptor vectors with L2 distance; score = 1/(1+dist)*100%.</li>
 * </ol>
 */
public final class FourierShapeMatcher {

    /** Uniform sample count fed to the DFT. */
    private static final int    N_SAMPLES       = 128;
    /** Number of Fourier coefficients kept (excluding DC). */
    private static final int    N_DESCRIPTORS   = 32;
    /** Minimum contour area to consider. */
    private static final double MIN_AREA        = 64.0;
    /** Binary threshold before contour extraction. */
    private static final int    BINARISE_THRESH = 30;

    private FourierShapeMatcher() {}

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

        double[] refDescBase  = buildDescriptor(refMat, null);
        Mat refMaskL          = ColourPreFilter.applyToReference(refMat, referenceId, ColourPreFilter.LOOSE);
        double[] refDescLoose = buildDescriptor(refMat, refMaskL);
        refMaskL.release();
        Mat refMaskT          = ColourPreFilter.applyToReference(refMat, referenceId, ColourPreFilter.TIGHT);
        double[] refDescTight = buildDescriptor(refMat, refMaskT);
        refMaskT.release();

        long t0 = System.currentTimeMillis();
        Mat sceneMaskL = ColourPreFilter.applyToScene(sceneMat, referenceId, ColourPreFilter.LOOSE);
        long cfLMs     = System.currentTimeMillis() - t0;

        t0 = System.currentTimeMillis();
        Mat sceneMaskT = ColourPreFilter.applyToScene(sceneMat, referenceId, ColourPreFilter.TIGHT);
        long cfTMs     = System.currentTimeMillis() - t0;

        List<MatOfPoint> contoursBase  = findContours(sceneMat, null);
        List<MatOfPoint> contoursLoose = findContours(sceneMat, sceneMaskL);
        List<MatOfPoint> contoursTight = findContours(sceneMat, sceneMaskT);
        sceneMaskL.release();
        sceneMaskT.release();

        out.add(runVariant(FourierShapeVariant.FOURIER_SHAPE.variantName(),
                refDescBase,  contoursBase,  sceneMat, 0L,    referenceId, scene, saveVariants, outputDir));
        out.add(runVariant(FourierShapeVariant.FOURIER_SHAPE_CF_LOOSE.variantName(),
                refDescLoose, contoursLoose, sceneMat, cfLMs, referenceId, scene, saveVariants, outputDir));
        out.add(runVariant(FourierShapeVariant.FOURIER_SHAPE_CF_TIGHT.variantName(),
                refDescTight, contoursTight, sceneMat, cfTMs, referenceId, scene, saveVariants, outputDir));

        releaseContours(contoursBase);
        releaseContours(contoursLoose);
        releaseContours(contoursTight);
        return out;
    }

    // -------------------------------------------------------------------------
    // Single variant
    // -------------------------------------------------------------------------

    private static AnalysisResult runVariant(String variantName,
                                              double[] refDesc,
                                              List<MatOfPoint> sceneContours,
                                              Mat sceneMat,
                                              long preFilterMs,
                                              ReferenceId referenceId,
                                              SceneEntry scene,
                                              Set<String> saveVariants,
                                              Path outputDir) {
        long t0 = System.currentTimeMillis();
        try {
            if (refDesc == null || refDesc.length == 0) {
                return new AnalysisResult(variantName, referenceId,
                        scene.variantLabel(), scene.category(), scene.backgroundId(),
                        0.0, new Rect(0, 0, 1, 1),
                        System.currentTimeMillis() - t0, preFilterMs,
                        scenePx(scene), null, false, null);
            }

            double bestScore = -1;
            Rect   bestBbox  = null;

            for (MatOfPoint c : sceneContours) {
                if (Imgproc.contourArea(c) < MIN_AREA) continue;
                double[] desc = descriptorFromContour(c);
                if (desc == null) continue;
                double dist  = l2Distance(refDesc, desc);
                double score = 1.0 / (1.0 + dist) * 100.0;
                if (score > bestScore) {
                    bestScore = score;
                    bestBbox  = Imgproc.boundingRect(c);
                }
            }

            long elapsed = System.currentTimeMillis() - t0;
            if (bestBbox == null) { bestScore = 0.0; bestBbox = new Rect(0, 0, 1, 1); }
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
    // Fourier descriptor
    // -------------------------------------------------------------------------

    private static double[] buildDescriptor(Mat bgr, Mat mask) {
        List<MatOfPoint> contours = findContours(bgr, mask);
        MatOfPoint best = null;
        double bestArea = 0;
        for (MatOfPoint c : contours) {
            double a = Imgproc.contourArea(c);
            if (a > bestArea) { bestArea = a; best = c; }
        }
        if (best == null || bestArea < MIN_AREA) { releaseContours(contours); return null; }
        double[] desc = descriptorFromContour(best);
        releaseContours(contours);
        return desc;
    }

    private static double[] descriptorFromContour(MatOfPoint contour) {
        Point[] pts = contour.toArray();
        if (pts.length < 4) return null;
        Point[] resampled = resample(pts, N_SAMPLES);

        double cx = 0, cy = 0;
        for (Point p : resampled) { cx += p.x; cy += p.y; }
        cx /= N_SAMPLES; cy /= N_SAMPLES;

        float[] real = new float[N_SAMPLES];
        float[] imag = new float[N_SAMPLES];
        for (int i = 0; i < N_SAMPLES; i++) {
            real[i] = (float)(resampled[i].x - cx);
            imag[i] = (float)(resampled[i].y - cy);
        }

        Mat realMat = new Mat(1, N_SAMPLES, CvType.CV_32F);
        Mat imagMat = new Mat(1, N_SAMPLES, CvType.CV_32F);
        realMat.put(0, 0, real);
        imagMat.put(0, 0, imag);
        List<Mat> ch = new ArrayList<>();
        ch.add(realMat); ch.add(imagMat);
        Mat complex = new Mat();
        Core.merge(ch, complex);
        realMat.release(); imagMat.release();

        Mat dft = new Mat();
        Core.dft(complex, dft, Core.DFT_COMPLEX_OUTPUT | Core.DFT_ROWS);
        complex.release();

        List<Mat> dftCh = new ArrayList<>();
        Core.split(dft, dftCh);
        dft.release();
        float[] re = new float[N_SAMPLES];
        float[] im = new float[N_SAMPLES];
        dftCh.get(0).get(0, 0, re);
        dftCh.get(1).get(0, 0, im);
        dftCh.forEach(Mat::release);

        double[] mag = new double[N_SAMPLES];
        for (int k = 0; k < N_SAMPLES; k++) mag[k] = Math.sqrt((double)re[k]*re[k] + (double)im[k]*im[k]);

        double norm = mag[1] > 1e-6 ? mag[1] : 1.0;
        double[] desc = new double[N_DESCRIPTORS];
        for (int k = 0; k < N_DESCRIPTORS; k++) desc[k] = mag[k + 1] / norm;
        return desc;
    }

    private static Point[] resample(Point[] pts, int n) {
        double[] arc = new double[pts.length];
        for (int i = 1; i < pts.length; i++) {
            double dx = pts[i].x - pts[i-1].x, dy = pts[i].y - pts[i-1].y;
            arc[i] = arc[i-1] + Math.sqrt(dx*dx + dy*dy);
        }
        double total = arc[pts.length - 1];
        if (total < 1e-6) {
            Point[] out = new Point[n];
            for (int i = 0; i < n; i++) out[i] = new Point(pts[0].x, pts[0].y);
            return out;
        }
        Point[] out = new Point[n];
        int j = 0;
        for (int i = 0; i < n; i++) {
            double target = total * i / n;
            while (j < pts.length - 2 && arc[j+1] < target) j++;
            double seg = arc[j+1] - arc[j];
            double t   = seg > 1e-9 ? (target - arc[j]) / seg : 0.0;
            out[i] = new Point(pts[j].x + t*(pts[j+1].x - pts[j].x),
                               pts[j].y + t*(pts[j+1].y - pts[j].y));
        }
        return out;
    }

    // -------------------------------------------------------------------------
    // Contour helpers
    // -------------------------------------------------------------------------

    private static List<MatOfPoint> findContours(Mat bgr, Mat mask) {
        Mat grey = new Mat();
        Imgproc.cvtColor(bgr, grey, Imgproc.COLOR_BGR2GRAY);
        if (mask != null && !mask.empty()) {
            Mat masked = new Mat(grey.size(), grey.type(), Scalar.all(0));
            grey.copyTo(masked, mask);
            grey.release(); grey = masked;
        }
        Mat bin = new Mat();
        Imgproc.threshold(grey, bin, BINARISE_THRESH, 255, Imgproc.THRESH_BINARY);
        grey.release();
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(bin.clone(), contours, new Mat(),
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
        bin.release();
        return contours;
    }

    private static void releaseContours(List<MatOfPoint> c) { c.forEach(MatOfPoint::release); }
    private static double l2Distance(double[] a, double[] b) {
        int len = Math.min(a.length, b.length); double s = 0;
        for (int i = 0; i < len; i++) { double d = a[i]-b[i]; s += d*d; }
        return Math.sqrt(s);
    }

    // -------------------------------------------------------------------------
    // Annotation + helpers
    // -------------------------------------------------------------------------

    private static Path writeAnnotated(Mat scene, Rect bbox, String variant, double score,
                                        ReferenceId refId, SceneEntry sceneEntry, Path outputDir) {
        try {
            Path dir = outputDir.resolve("annotated").resolve(sanitise(variant));
            Files.createDirectories(dir);
            String sceneRef = sceneEntry.primaryReferenceId() != null
                    ? sanitise(sceneEntry.primaryReferenceId().name()) : "neg";
            String fname = sanitise(refId.name()) + "_vs_" + sceneRef
                    + "_" + sanitise(sceneEntry.variantLabel()) + ".png";
            Path dest = dir.resolve(fname);
            Mat    m      = scene.clone();
            Scalar colour = score >= 70 ? new Scalar(0,200,0) : score >= 40 ? new Scalar(0,200,200) : new Scalar(0,0,200);
            if (bbox.width > 1 && bbox.height > 1)
                Imgproc.rectangle(m, new Point(bbox.x, bbox.y),
                        new Point(bbox.x+bbox.width, bbox.y+bbox.height), colour, 2);
            Imgproc.putText(m, shortName(variant), new Point(4,13), Imgproc.FONT_HERSHEY_SIMPLEX, 0.32, new Scalar(200,200,200), 1);
            Imgproc.putText(m, String.format("%.1f%%", score), new Point(4,28), Imgproc.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1);
            Imgcodecs.imwrite(dest.toString(), m);
            m.release();
            return outputDir.toAbsolutePath().relativize(dest.toAbsolutePath());
        } catch (Exception e) { return null; }
    }

    private static int    scenePx(SceneEntry s) { return s.sceneMat().cols() * s.sceneMat().rows(); }
    private static String sanitise(String s)    { return s.replaceAll("[^A-Za-z0-9_\\-]", "_"); }
    private static String shortName(String v)   {
        return v.replace("FOURIER_SHAPE_CF_LOOSE","FS·CFL")
                .replace("FOURIER_SHAPE_CF_TIGHT","FS·CFT")
                .replace("FOURIER_SHAPE","FS");
    }
}

