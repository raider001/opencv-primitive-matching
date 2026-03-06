package org.example.matchers;

import org.example.*;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Feature Matching technique — Milestone 8.
 *
 * <p>Runs 4 standard feature detector/descriptor variants (ORB, AKAZE, BRISK, KAZE),
 * each in base, CF_LOOSE, and CF_TIGHT modes = <b>12 variants total</b>.
 *
 * <p>SIFT is attempted at runtime; if the class is unavailable in the current OpenCV
 * build it is silently skipped (openpnp 4.7.0 may not include xfeatures2d).
 *
 * <p>Matching pipeline per variant:
 * <ol>
 *   <li>Detect keypoints + compute descriptors on reference and scene.</li>
 *   <li>BFMatcher (Hamming for binary descriptors, L2 for float descriptors) with knnMatch(k=2).</li>
 *   <li>Lowe's ratio test (threshold 0.75) to filter good matches.</li>
 *   <li>If ≥ 4 good matches: compute homography (RANSAC) and count inliers.</li>
 *   <li>Score = inlier count / max(1, reference keypoint count) × 100, clamped to 0–100.</li>
 * </ol>
 *
 * <p>For CF variants, the colour pre-filter mask restricts keypoint detection to the
 * colour-isolated region in both the scene and the reference.
 */
public final class FeatureMatcher {

    // -------------------------------------------------------------------------
    // Variant descriptors
    // -------------------------------------------------------------------------

    public record VariantDef(String name, boolean binary) {}

    /** All variants attempted. SIFT is tried first and skipped if unavailable. */
    public static final VariantDef[] VARIANTS = {
        new VariantDef("SIFT",  false),
        new VariantDef("ORB",   true),
        new VariantDef("AKAZE", true),
        new VariantDef("BRISK", true),
        new VariantDef("KAZE",  false),
    };

    private static final double RATIO_THRESH  = 0.75;
    private static final int    MIN_MATCHES   = 4;

    /** CF1 representative variant names (SIFT only). */
    public static final String CF1_LOOSE = "SIFT_CF1_LOOSE";
    public static final String CF1_TIGHT = "SIFT_CF1_TIGHT";

    private FeatureMatcher() {}

    // -------------------------------------------------------------------------
    // Public entry point
    // -------------------------------------------------------------------------

    /**
     * Runs all available feature variants (base + CF_LOOSE + CF_TIGHT) against one pair.
     *
     * @param referenceId  reference ID
     * @param refMat       128×128 BGR reference — caller retains ownership
     * @param scene        scene entry
     * @param saveVariants variant names whose annotated PNG should be written to disk
     * @param outputDir    root output directory
     * @return list of {@link AnalysisResult} (up to 15: 5 detectors × 3 modes, fewer if SIFT absent)
     */
    public static List<AnalysisResult> match(ReferenceId referenceId,
                                              Mat refMat,
                                              SceneEntry scene,
                                              Set<String> saveVariants,
                                              Path outputDir) {
        List<AnalysisResult> out = new ArrayList<>();
        Mat sceneMat = scene.sceneMat();

        for (VariantDef vd : VARIANTS) {
            // Create a fresh detector instance per variant per call — avoids
            // thread-safety issues when match() is called from parallel streams.
            Feature2D detector = createDetector(vd.name());
            if (detector == null) continue;

            // Base
            out.add(runVariant(vd.name(), vd.binary(), detector,
                    sceneMat, refMat, null, null, 0L,
                    referenceId, scene, saveVariants, outputDir));

            // CF_LOOSE
            long t0 = System.currentTimeMillis();
            Mat maskScene = ColourPreFilter.applyToScene(sceneMat, referenceId, ColourPreFilter.LOOSE);
            Mat maskRef   = ColourPreFilter.applyToReference(refMat, referenceId, ColourPreFilter.LOOSE);
            long cfL = System.currentTimeMillis() - t0;
            out.add(runVariant(vd.name() + "_CF_LOOSE", vd.binary(), detector,
                    sceneMat, refMat, maskScene, maskRef, cfL,
                    referenceId, scene, saveVariants, outputDir));
            maskScene.release(); maskRef.release();

            // CF_TIGHT
            t0 = System.currentTimeMillis();
            maskScene = ColourPreFilter.applyToScene(sceneMat, referenceId, ColourPreFilter.TIGHT);
            maskRef   = ColourPreFilter.applyToReference(refMat, referenceId, ColourPreFilter.TIGHT);
            long cfT = System.currentTimeMillis() - t0;
            out.add(runVariant(vd.name() + "_CF_TIGHT", vd.binary(), detector,
                    sceneMat, refMat, maskScene, maskRef, cfT,
                    referenceId, scene, saveVariants, outputDir));
            maskScene.release(); maskRef.release();
        }

        // ---- CF1 variants (SIFT inside colour-first windows) ----
        Feature2D sift = createDetector("SIFT");
        if (sift != null) {
            for (String cf1Name : new String[]{CF1_LOOSE, CF1_TIGHT}) {
                double tol = cf1Name.endsWith("LOOSE") ? ColourPreFilter.LOOSE : ColourPreFilter.TIGHT;
                long t0 = System.currentTimeMillis();
                List<Rect> windows = ColourFirstLocator.propose(sceneMat, referenceId, tol);
                long cfMs = System.currentTimeMillis() - t0;

                double bestScore = -1;
                Rect   bestBbox  = windows.get(0);
                for (Rect w : windows) {
                    // Crop scene to candidate window
                    Mat crop = sceneMat.submat(w);
                    AnalysisResult r = runVariant(cf1Name, false, sift,
                            crop, refMat, null, null, cfMs,
                            referenceId, scene, saveVariants, outputDir);
                    if (r.matchScorePercent() > bestScore) {
                        bestScore = r.matchScorePercent();
                        Rect lb = r.boundingRect();
                        if (lb != null) {
                            bestBbox = new Rect(w.x + lb.x, w.y + lb.y, lb.width, lb.height);
                        }
                    }
                }
                Path savedPath = null;
                if (saveVariants.contains(cf1Name) && bestScore >= 0) {
                    savedPath = writeAnnotated(sceneMat, bestBbox,
                            cf1Name, Math.max(0, bestScore), referenceId, scene, outputDir);
                }
                out.add(new AnalysisResult(cf1Name, referenceId,
                        scene.variantLabel(), scene.category(), scene.backgroundId(),
                        Math.max(0, bestScore), bestBbox,
                        System.currentTimeMillis() - t0, cfMs,
                        scenePx(scene), savedPath, false, null));
            }
        }

        return out;
    }

    // -------------------------------------------------------------------------
    // Single variant
    // -------------------------------------------------------------------------

    private static AnalysisResult runVariant(String variantName,
                                              boolean binary,
                                              Feature2D detector,
                                              Mat sceneMat,
                                              Mat refMat,
                                              Mat sceneMask,
                                              Mat refMask,
                                              long preFilterMs,
                                              ReferenceId referenceId,
                                              SceneEntry scene,
                                              Set<String> saveVariants,
                                              Path outputDir) {
        long t0 = System.currentTimeMillis();
        try {
            // Convert to greyscale for detection
            Mat sceneGrey = toGrey(sceneMat);
            Mat refGrey   = toGrey(refMat);

            // Detect + compute
            MatOfKeyPoint kpScene = new MatOfKeyPoint();
            MatOfKeyPoint kpRef   = new MatOfKeyPoint();
            Mat descScene = new Mat();
            Mat descRef   = new Mat();

            detector.detectAndCompute(sceneGrey, sceneMask != null ? sceneMask : new Mat(), kpScene, descScene);
            detector.detectAndCompute(refGrey,   refMask   != null ? refMask   : new Mat(), kpRef,   descRef);

            sceneGrey.release();
            refGrey.release();

            int refKpCount = (int) kpRef.total();

            // No descriptors — score 0, still write annotated so HTML report shows something
            if (descRef.empty() || descScene.empty() || refKpCount == 0) {
                kpScene.release(); kpRef.release(); descScene.release(); descRef.release();
                long elapsed = System.currentTimeMillis() - t0;
                Path saved = writeAnnotated(sceneMat, null, variantName, 0.0,
                        referenceId, scene, outputDir);
                return new AnalysisResult(variantName, referenceId,
                        scene.variantLabel(), scene.category(), scene.backgroundId(),
                        0.0, null, elapsed, preFilterMs, scenePx(scene), saved, false, null);
            }

            // BFMatcher — Hamming for binary, L2 for float
            int normType = binary ? Core.NORM_HAMMING : Core.NORM_L2;
            DescriptorMatcher matcher = DescriptorMatcher.create(normType == Core.NORM_HAMMING
                    ? DescriptorMatcher.BRUTEFORCE_HAMMING
                    : DescriptorMatcher.BRUTEFORCE);

            List<MatOfDMatch> knnMatches = new ArrayList<>();
            matcher.knnMatch(descRef, descScene, knnMatches, 2);

            descRef.release(); descScene.release();

            // Lowe's ratio test
            List<DMatch> good = new ArrayList<>();
            for (MatOfDMatch pair : knnMatches) {
                DMatch[] arr = pair.toArray();
                if (arr.length >= 2 && arr[0].distance < RATIO_THRESH * arr[1].distance) {
                    good.add(arr[0]);
                }
                pair.release();
            }

            double score;
            Rect   bbox = null;

            if (good.size() >= MIN_MATCHES) {
                // Homography RANSAC to find inliers
                KeyPoint[] kpRefArr   = kpRef.toArray();
                KeyPoint[] kpSceneArr = kpScene.toArray();

                List<Point> refPts   = new ArrayList<>();
                List<Point> scenePts = new ArrayList<>();
                for (DMatch m : good) {
                    refPts.add(kpRefArr[m.queryIdx].pt);
                    scenePts.add(kpSceneArr[m.trainIdx].pt);
                }

                MatOfPoint2f refPtsMat   = new MatOfPoint2f(refPts.toArray(new Point[0]));
                MatOfPoint2f scenePtsMat = new MatOfPoint2f(scenePts.toArray(new Point[0]));
                Mat inlierMask = new Mat();

                Mat H = Calib3d.findHomography(refPtsMat, scenePtsMat, Calib3d.RANSAC, 3.0, inlierMask);
                refPtsMat.release(); scenePtsMat.release();

                int inliers = 0;
                if (!H.empty()) {
                    for (int i = 0; i < inlierMask.rows(); i++) {
                        if (inlierMask.get(i, 0)[0] != 0) inliers++;
                    }
                    bbox = projectBbox(H, refMat.cols(), refMat.rows(),
                            sceneMat.cols(), sceneMat.rows());
                    H.release();
                }
                inlierMask.release();

                score = Math.min(100.0, (inliers * 100.0) / refKpCount);
            } else {
                // Not enough good matches
                score = Math.min(100.0, (good.size() * 100.0) / Math.max(1, refKpCount));
            }

            kpRef.release(); kpScene.release();

            long elapsed = System.currentTimeMillis() - t0;

            // Write annotated image to disk — always saved (one per variant per scene)
            Path savedPath = writeAnnotated(sceneMat, bbox, variantName, score,
                    referenceId, scene, outputDir);

            return new AnalysisResult(variantName, referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    score, bbox, elapsed, preFilterMs, scenePx(scene), savedPath, false, null);

        } catch (Exception e) {
            return AnalysisResult.error(variantName, referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    System.currentTimeMillis() - t0, scenePx(scene), e.getMessage());
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /** Serialises OpenCV Feature2D constructor calls which are not thread-safe. */
    private static final Object DETECTOR_LOCK = new Object();

    /** Creates the Feature2D detector for the given name; returns null if unavailable. */
    private static Feature2D createDetector(String name) {
        synchronized (DETECTOR_LOCK) {
            try {
                if ("SIFT".equals(name)) {
                    try {
                        Class<?> cls = Class.forName("org.opencv.xfeatures2d.SIFT");
                        return (Feature2D) cls.getMethod("create").invoke(null);
                    } catch (Exception ex) {
                        return SIFT.create();
                    }
                } else if ("ORB".equals(name)) {
                    return ORB.create(500);
                } else if ("AKAZE".equals(name)) {
                    return AKAZE.create();
                } else if ("BRISK".equals(name)) {
                    return BRISK.create();
                } else if ("KAZE".equals(name)) {
                    return KAZE.create();
                }
                return null;
            } catch (Exception e) {
                return null;
            }
        }
    }

    private static Mat toGrey(Mat bgr) {
        Mat grey = new Mat();
        if (bgr.channels() == 1) {
            bgr.copyTo(grey);
        } else {
            Imgproc.cvtColor(bgr, grey, Imgproc.COLOR_BGR2GRAY);
        }
        return grey;
    }

    /** Projects the reference image corners through homography to get the bounding box in the scene. */
    private static Rect projectBbox(Mat H, int refW, int refH, int sceneW, int sceneH) {
        try {
            MatOfPoint2f corners = new MatOfPoint2f(
                    new Point(0, 0), new Point(refW, 0),
                    new Point(refW, refH), new Point(0, refH));
            MatOfPoint2f projected = new MatOfPoint2f();
            Core.perspectiveTransform(corners, projected, H);
            corners.release();

            Point[] pts = projected.toArray();
            projected.release();

            double minX = pts[0].x, maxX = pts[0].x;
            double minY = pts[0].y, maxY = pts[0].y;
            for (Point p : pts) {
                minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
                minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
            }
            int x = Math.max(0, (int) minX);
            int y = Math.max(0, (int) minY);
            int w = Math.min(sceneW - x, (int) (maxX - minX));
            int h = Math.min(sceneH - y, (int) (maxY - minY));
            return new Rect(x, y, Math.max(1, w), Math.max(1, h));
        } catch (Exception e) {
            return new Rect(0, 0, Math.min(refW, sceneW), Math.min(refH, sceneH));
        }
    }

    private static Path writeAnnotated(Mat scene, Rect bbox, String variant, double score,
                                        ReferenceId refId, SceneEntry sceneEntry, Path outputDir) {
        try {
            Path dir = outputDir.resolve("annotated").resolve(sanitise(variant));
            Files.createDirectories(dir);
            String fname = sanitise(refId.name()) + "_" + sanitise(sceneEntry.variantLabel()) + ".png";
            Path dest = dir.resolve(fname);

            Mat m = scene.clone();
            Scalar colour = score >= 70 ? new Scalar(0, 200, 0)
                          : score >= 40 ? new Scalar(0, 200, 200)
                          :               new Scalar(0, 0, 200);
            if (bbox != null) {
                Imgproc.rectangle(m, new Point(bbox.x, bbox.y),
                        new Point(bbox.x + bbox.width, bbox.y + bbox.height), colour, 2);
            }
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

    private static int    scenePx(SceneEntry s) { return s.sceneMat().cols() * s.sceneMat().rows(); }
    private static String sanitise(String s)    { return s.replaceAll("[^A-Za-z0-9_\\-]", "_"); }
    private static String shortName(String v)   { return v.replace("_CF_", "\u00B7"); }
}




























