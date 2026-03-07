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
 * Hough Transform technique — Milestone 10.
 *
 * <p>Two base detectors, each in base / CF_LOOSE / CF_TIGHT modes = <b>6 variants total</b>:
 * <ul>
 *   <li>{@code HoughLinesP}  — Probabilistic Hough line-segment detector</li>
 *   <li>{@code HoughCircles} — Hough circle detector (Hough Gradient method)</li>
 * </ul>
 *
 * <h2>Scoring strategy</h2>
 * <p>Hough detectors do not return a similarity score directly — they return a set of
 * detected primitives.  We score a scene by how well its detections match what the
 * reference image is expected to contain:
 *
 * <ul>
 *   <li><b>HoughLinesP:</b> Run on both the binarised reference and the binarised scene.
 *       The scene score is the line-count similarity:
 *       {@code score = min(sceneLines, refLines) / max(1, max(sceneLines, refLines)) × 100}.
 *       Bounding box = axis-aligned rect that spans all detected line segments in the scene.</li>
 *
 *   <li><b>HoughCircles:</b> Run on the grey reference to get an expected radius range.
 *       Run on the grey scene.  The best-matching circle (closest radius to the reference
 *       median radius) is the detection.  Score = radius-match ratio clamped to 0–100%,
 *       boosted by the scene circle count relative to the reference count.
 *       Bounding box = bounding rect of the best-matching circle.</li>
 * </ul>
 *
 * <p>CF variants apply the colour pre-filter mask to the grey/binary image before running
 * the detector — this removes background primitives of the wrong colour, giving a much
 * cleaner detection set.
 */
public final class HoughDetector {

    // -------------------------------------------------------------------------
    // Variant names
    // -------------------------------------------------------------------------

    /** @deprecated Use {@link HoughVariant#HOUGH_LINES_P}. */
    @Deprecated public static final String VAR_LINES   = HoughVariant.HOUGH_LINES_P.variantName();
    /** @deprecated Use {@link HoughVariant#HOUGH_CIRCLES}. */
    @Deprecated public static final String VAR_CIRCLES = HoughVariant.HOUGH_CIRCLES.variantName();

    // -------------------------------------------------------------------------
    // HoughLinesP parameters
    // -------------------------------------------------------------------------
    /** Accumulator resolution (pixels). */
    private static final double LINES_RHO         = 1.0;
    /** Accumulator resolution (radians). */
    private static final double LINES_THETA       = Math.PI / 180.0;
    /** Minimum vote count to accept a line. */
    private static final int    LINES_THRESHOLD   = 20;
    /** Minimum line length (px). */
    private static final double LINES_MIN_LENGTH  = 15.0;
    /** Maximum gap between collinear segments to merge (px). */
    private static final double LINES_MAX_GAP     = 8.0;
    /** Canny low/high thresholds used before HoughLinesP. */
    private static final double CANNY_LOW         = 50.0;
    private static final double CANNY_HIGH        = 150.0;

    // -------------------------------------------------------------------------
    // HoughCircles parameters
    // -------------------------------------------------------------------------
    /** Inverse accumulator resolution (1 = same as input). */
    private static final double CIRCLES_DP         = 1.2;
    /** Minimum distance between circle centres (px). */
    private static final double CIRCLES_MIN_DIST   = 20.0;
    /** Canny high threshold (internal). */
    private static final double CIRCLES_PARAM1     = 100.0;
    /** Accumulator threshold — lower = more circles detected. */
    private static final double CIRCLES_PARAM2     = 25.0;
    /** Min / max radius searched (0 = auto). */
    private static final int    CIRCLES_MIN_RADIUS = 5;
    private static final int    CIRCLES_MAX_RADIUS = 200;

    private HoughDetector() {}

    // -------------------------------------------------------------------------
    // Public entry point
    // -------------------------------------------------------------------------

    /**
     * Runs all 6 Hough variants against one (reference, scene) pair.
     *
     * @param referenceId  reference ID
     * @param refMat       128×128 BGR reference — caller retains ownership
     * @param scene        scene entry
     * @param saveVariants variant names whose annotated PNG should be written to disk
     * @param outputDir    root output directory
     * @return 6 {@link AnalysisResult} objects
     */
    public static List<AnalysisResult> match(ReferenceId referenceId,
                                              Mat refMat,
                                              SceneEntry scene,
                                              Set<String> saveVariants,
                                              Path outputDir) {
        List<AnalysisResult> out = new ArrayList<>(6);
        Mat sceneMat = scene.sceneMat();

        // --- Pre-compute grey + binary reference images ---
        Mat refGrey  = toGrey(refMat);
        Mat refCanny = canny(refGrey);

        // --- Pre-compute scene grey / canny for base ---
        Mat sceneGrey  = toGrey(sceneMat);
        Mat sceneCanny = canny(sceneGrey);

        // --- CF_LOOSE ---
        long t0 = System.currentTimeMillis();
        Mat looseMask       = ColourPreFilter.applyToScene(sceneMat, referenceId, ColourPreFilter.LOOSE);
        Mat sceneGreyLoose  = maskedGrey(sceneGrey, looseMask);
        Mat sceneCannyLoose = canny(sceneGreyLoose);
        long cfLMs = System.currentTimeMillis() - t0;
        looseMask.release();
        sceneGreyLoose.release();

        // --- CF_TIGHT ---
        t0 = System.currentTimeMillis();
        Mat tightMask       = ColourPreFilter.applyToScene(sceneMat, referenceId, ColourPreFilter.TIGHT);
        Mat sceneGreyTight  = maskedGrey(sceneGrey, tightMask);
        Mat sceneCannyTight = canny(sceneGreyTight);
        long cfTMs = System.currentTimeMillis() - t0;
        tightMask.release();
        sceneGreyTight.release();

        // =====================================================================
        // HoughLinesP
        // =====================================================================
        // Detect reference lines once — used to calibrate score
        List<double[]> refLines = detectLines(refCanny);

        out.add(runLinesVariant(VAR_LINES, sceneMat, sceneCanny,
                refLines, 0L, referenceId, scene, saveVariants, outputDir));

        out.add(runLinesVariant(VAR_LINES + "_CF_LOOSE", sceneMat, sceneCannyLoose,
                refLines, cfLMs, referenceId, scene, saveVariants, outputDir));

        out.add(runLinesVariant(VAR_LINES + "_CF_TIGHT", sceneMat, sceneCannyTight,
                refLines, cfTMs, referenceId, scene, saveVariants, outputDir));

        // =====================================================================
        // HoughCircles
        // =====================================================================
        // Detect reference circles — extract expected radius range
        Mat refGreyBlurred = blur(refGrey);
        List<double[]> refCircles = detectCircles(refGreyBlurred);
        refGreyBlurred.release();

        out.add(runCirclesVariant(VAR_CIRCLES, sceneMat, sceneGrey,
                refCircles, 0L, referenceId, scene, saveVariants, outputDir));

        Mat sceneGreyLoose2  = maskedGrey(sceneGrey, ColourPreFilter.applyToScene(sceneMat, referenceId, ColourPreFilter.LOOSE));
        out.add(runCirclesVariant(VAR_CIRCLES + "_CF_LOOSE", sceneMat, sceneGreyLoose2,
                refCircles, cfLMs, referenceId, scene, saveVariants, outputDir));
        sceneGreyLoose2.release();

        Mat sceneGreyTight2  = maskedGrey(sceneGrey, ColourPreFilter.applyToScene(sceneMat, referenceId, ColourPreFilter.TIGHT));
        out.add(runCirclesVariant(VAR_CIRCLES + "_CF_TIGHT", sceneMat, sceneGreyTight2,
                refCircles, cfTMs, referenceId, scene, saveVariants, outputDir));
        sceneGreyTight2.release();

        // Release shared resources
        refGrey.release();
        refCanny.release();
        sceneGrey.release();
        sceneCanny.release();
        sceneCannyLoose.release();
        sceneCannyTight.release();

        // =====================================================================
        // CF1 variants — HoughLinesP inside colour-first windows
        // =====================================================================
        for (HoughVariant cf1 : new HoughVariant[]{
                HoughVariant.HOUGH_LINES_P_CF1_LOOSE,
                HoughVariant.HOUGH_LINES_P_CF1_TIGHT}) {

            String cf1Name  = cf1.variantName();
            double tol      = cf1.cfMode().hueTolerance();
            long   cf1Start = System.currentTimeMillis();
            List<Rect> windows = ColourFirstLocator.propose(sceneMat, referenceId, tol);
            long cfMs = System.currentTimeMillis() - cf1Start;

            // Re-detect reference lines for scoring baseline
            Mat rfGrey  = toGrey(refMat);
            Mat rfCanny = canny(rfGrey);
            rfGrey.release();
            List<double[]> rfLines = detectLines(rfCanny);
            rfCanny.release();

            double        bestScore  = -1;
            Rect          bestBbox   = windows.get(0);
            List<double[]> bestLines = List.of();

            for (Rect w : windows) {
                Mat cropBgr   = new Mat(sceneMat, w);
                Mat cropGrey  = toGrey(cropBgr);
                Mat cropCanny = canny(cropGrey);
                cropGrey.release();
                AnalysisResult r = runLinesVariant(cf1Name, cropBgr, cropCanny,
                        rfLines, cfMs, referenceId, scene, saveVariants, outputDir);
                cropCanny.release();
                if (r.matchScorePercent() > bestScore) {
                    bestScore = r.matchScorePercent();
                    Rect lb   = r.boundingRect();
                    if (lb != null)
                        bestBbox = new Rect(w.x + lb.x, w.y + lb.y, lb.width, lb.height);
                    // Re-detect lines in the winning crop for annotation
                    Mat wCropGrey  = toGrey(cropBgr);
                    Mat wCropCanny = canny(wCropGrey);
                    wCropGrey.release();
                    bestLines = detectLines(wCropCanny);
                    wCropCanny.release();
                }
            }

            Path savedPath = null;
            if (saveVariants.contains(cf1Name)) {
                savedPath = writeLinesAnnotated(sceneMat, bestLines, bestBbox,
                        cf1Name, Math.max(0, bestScore), referenceId, scene, outputDir);
            }
            out.add(new AnalysisResult(cf1Name, referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    Math.max(0, bestScore), bestBbox,
                    System.currentTimeMillis() - cf1Start, cfMs,
                    scenePx(scene), savedPath, false, null));
        }

        return out;
    }

    // =========================================================================
    // HoughLinesP variant
    // =========================================================================

    private static AnalysisResult runLinesVariant(String variantName,
                                                    Mat sceneMat,
                                                    Mat sceneCanny,
                                                    List<double[]> refLines,
                                                    long preFilterMs,
                                                    ReferenceId referenceId,
                                                    SceneEntry scene,
                                                    Set<String> saveVariants,
                                                    Path outputDir) {
        long t0 = System.currentTimeMillis();
        try {
            List<double[]> sceneLines = detectLines(sceneCanny);

            long elapsed = System.currentTimeMillis() - t0;

            // Score: ratio of detected line counts (how many lines in scene vs reference)
            int nRef   = Math.max(1, refLines.size());
            int nScene = sceneLines.size();
            double score;
            if (nScene == 0) {
                score = 0.0;
            } else {
                // Reward scenes that have a similar line count to the reference.
                // If the reference has N lines we want the scene to also detect ~N.
                double ratio = (double) Math.min(nScene, nRef) / Math.max(nScene, nRef);
                score = ratio * 100.0;
            }
            score = Math.max(0, Math.min(100, score));

            // Bounding box = axis-aligned rect spanning all detected scene line segments
            Rect bbox = lineBoundingBox(sceneLines, sceneMat.cols(), sceneMat.rows());

            Path savedPath = null;
            if (saveVariants.contains(variantName)) {
                savedPath = writeLinesAnnotated(sceneMat, sceneLines, bbox,
                        variantName, score, referenceId, scene, outputDir);
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
    // HoughCircles variant
    // =========================================================================

    private static AnalysisResult runCirclesVariant(String variantName,
                                                      Mat sceneMat,
                                                      Mat sceneGrey,
                                                      List<double[]> refCircles,
                                                      long preFilterMs,
                                                      ReferenceId referenceId,
                                                      SceneEntry scene,
                                                      Set<String> saveVariants,
                                                      Path outputDir) {
        long t0 = System.currentTimeMillis();
        try {
            Mat blurred = blur(sceneGrey);
            List<double[]> sceneCircles = detectCircles(blurred);
            blurred.release();

            long elapsed = System.currentTimeMillis() - t0;

            double score;
            Rect   bbox;

            if (sceneCircles.isEmpty()) {
                score = 0.0;
                bbox  = new Rect(0, 0, 1, 1);
            } else if (refCircles.isEmpty()) {
                // Reference has no circles — any detection is a false positive
                score = 0.0;
                bbox  = circleBoundingRect(sceneCircles.get(0));
            } else {
                // Find the scene circle whose radius best matches the median reference radius
                double refMedianR = medianRadius(refCircles);
                double[] best     = bestMatchingCircle(sceneCircles, refMedianR);

                // Score: radius match ratio (how close detected radius is to reference)
                double radiusRatio = Math.min(best[2], refMedianR) / Math.max(best[2], refMedianR);
                // Also reward having a similar count of circles
                int    nRef        = Math.max(1, refCircles.size());
                int    nScene      = sceneCircles.size();
                double countRatio  = (double) Math.min(nScene, nRef) / Math.max(nScene, nRef);
                score = (radiusRatio * 0.7 + countRatio * 0.3) * 100.0;
                score = Math.max(0, Math.min(100, score));
                bbox  = circleBoundingRect(best);
            }

            Path savedPath = null;
            if (saveVariants.contains(variantName)) {
                savedPath = writeCirclesAnnotated(sceneMat, sceneCircles, bbox,
                        variantName, score, referenceId, scene, outputDir);
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
    // Detection helpers
    // =========================================================================

    /** Runs HoughLinesP on a Canny edge image; returns list of [x1,y1,x2,y2] segments. */
    private static List<double[]> detectLines(Mat canny) {
        Mat linesRaw = new Mat();
        Imgproc.HoughLinesP(canny, linesRaw,
                LINES_RHO, LINES_THETA, LINES_THRESHOLD,
                LINES_MIN_LENGTH, LINES_MAX_GAP);
        List<double[]> lines = new ArrayList<>();
        for (int i = 0; i < linesRaw.rows(); i++) {
            lines.add(linesRaw.get(i, 0));
        }
        linesRaw.release();
        return lines;
    }

    /** Runs HoughCircles on a blurred grey image; returns list of [cx,cy,r]. */
    private static List<double[]> detectCircles(Mat grey) {
        Mat circlesRaw = new Mat();
        Imgproc.HoughCircles(grey, circlesRaw, Imgproc.HOUGH_GRADIENT,
                CIRCLES_DP, CIRCLES_MIN_DIST,
                CIRCLES_PARAM1, CIRCLES_PARAM2,
                CIRCLES_MIN_RADIUS, CIRCLES_MAX_RADIUS);
        List<double[]> circles = new ArrayList<>();
        for (int i = 0; i < circlesRaw.cols(); i++) {
            circles.add(circlesRaw.get(0, i));
        }
        circlesRaw.release();
        return circles;
    }

    /** Returns the circle [cx,cy,r] from {@code circles} whose radius is closest to {@code targetR}. */
    private static double[] bestMatchingCircle(List<double[]> circles, double targetR) {
        double[] best  = circles.get(0);
        double   bestD = Math.abs(best[2] - targetR);
        for (double[] c : circles) {
            double d = Math.abs(c[2] - targetR);
            if (d < bestD) { best = c; bestD = d; }
        }
        return best;
    }

    /** Median radius of a list of [cx,cy,r] circles. */
    private static double medianRadius(List<double[]> circles) {
        List<Double> radii = new ArrayList<>();
        for (double[] c : circles) radii.add(c[2]);
        radii.sort(Double::compareTo);
        return radii.get(radii.size() / 2);
    }

    /** Bounding rect of a circle [cx,cy,r]. */
    private static Rect circleBoundingRect(double[] c) {
        int r = (int) Math.ceil(c[2]);
        int x = (int) Math.round(c[0]) - r;
        int y = (int) Math.round(c[1]) - r;
        return new Rect(Math.max(0, x), Math.max(0, y), r * 2, r * 2);
    }

    /**
     * Axis-aligned bounding rect that spans all detected line segments.
     * Falls back to a 1×1 rect if no lines are detected.
     */
    private static Rect lineBoundingBox(List<double[]> lines, int sceneW, int sceneH) {
        if (lines.isEmpty()) return new Rect(0, 0, 1, 1);
        int x1 = sceneW, y1 = sceneH, x2 = 0, y2 = 0;
        for (double[] seg : lines) {
            int lx1 = (int) seg[0], ly1 = (int) seg[1];
            int lx2 = (int) seg[2], ly2 = (int) seg[3];
            x1 = Math.min(x1, Math.min(lx1, lx2));
            y1 = Math.min(y1, Math.min(ly1, ly2));
            x2 = Math.max(x2, Math.max(lx1, lx2));
            y2 = Math.max(y2, Math.max(ly1, ly2));
        }
        return new Rect(x1, y1, Math.max(1, x2 - x1), Math.max(1, y2 - y1));
    }

    // =========================================================================
    // Image conversion helpers
    // =========================================================================

    private static Mat toGrey(Mat bgr) {
        Mat grey = new Mat();
        Imgproc.cvtColor(bgr, grey, Imgproc.COLOR_BGR2GRAY);
        return grey;
    }

    private static Mat canny(Mat grey) {
        Mat edges = new Mat();
        Imgproc.Canny(grey, edges, CANNY_LOW, CANNY_HIGH);
        return edges;
    }

    private static Mat blur(Mat grey) {
        Mat blurred = new Mat();
        Imgproc.GaussianBlur(grey, blurred, new Size(9, 9), 2.0, 2.0);
        return blurred;
    }

    /**
     * Returns a greyscale image where pixels outside {@code mask} are zeroed.
     * Used to restrict Hough detectors to the colour-filtered region.
     */
    private static Mat maskedGrey(Mat grey, Mat mask) {
        Mat out = new Mat(grey.size(), grey.type(), Scalar.all(0));
        grey.copyTo(out, mask);
        return out;
    }

    // =========================================================================
    // Annotation writers
    // =========================================================================

    private static Path writeLinesAnnotated(Mat scene,
                                              List<double[]> lines,
                                              Rect bbox,
                                              String variant, double score,
                                              ReferenceId refId,
                                              SceneEntry sceneEntry,
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

            // Draw each detected line segment in the score colour
            for (double[] seg : lines) {
                Imgproc.line(m,
                        new Point(seg[0], seg[1]),
                        new Point(seg[2], seg[3]),
                        colour, 2);
            }
            // Bounding rect of all lines
            if (bbox.width > 1 && bbox.height > 1) {
                Imgproc.rectangle(m,
                        new Point(bbox.x, bbox.y),
                        new Point(bbox.x + bbox.width, bbox.y + bbox.height),
                        colour, 1);
            }
            Imgproc.putText(m, String.format("%.1f%%  n=%d", score, lines.size()),
                    new Point(4, 13), Imgproc.FONT_HERSHEY_SIMPLEX, 0.38,
                    colour, 1);
            Imgproc.putText(m, shortName(variant),
                    new Point(4, 26), Imgproc.FONT_HERSHEY_SIMPLEX, 0.32,
                    new Scalar(200, 200, 200), 1);
            Imgcodecs.imwrite(dest.toString(), m);
            m.release();
            return outputDir.toAbsolutePath().relativize(dest.toAbsolutePath());
        } catch (Exception e) {
            return null;
        }
    }

    private static Path writeCirclesAnnotated(Mat scene,
                                               List<double[]> circles,
                                               Rect bbox,
                                               String variant, double score,
                                               ReferenceId refId,
                                               SceneEntry sceneEntry,
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

            // Draw all detected circles
            for (double[] c : circles) {
                Point centre = new Point(Math.round(c[0]), Math.round(c[1]));
                int   radius = (int) Math.round(c[2]);
                Imgproc.circle(m, centre, radius, colour, 2);
                Imgproc.circle(m, centre, 3,      colour, Core.FILLED);
            }
            // Highlight the best-match bounding rect
            if (bbox.width > 1 && bbox.height > 1) {
                Imgproc.rectangle(m,
                        new Point(bbox.x, bbox.y),
                        new Point(bbox.x + bbox.width, bbox.y + bbox.height),
                        colour, 1);
            }
            Imgproc.putText(m, String.format("%.1f%%  n=%d", score, circles.size()),
                    new Point(4, 13), Imgproc.FONT_HERSHEY_SIMPLEX, 0.38,
                    colour, 1);
            Imgproc.putText(m, shortName(variant),
                    new Point(4, 26), Imgproc.FONT_HERSHEY_SIMPLEX, 0.32,
                    new Scalar(200, 200, 200), 1);
            Imgcodecs.imwrite(dest.toString(), m);
            m.release();
            return outputDir.toAbsolutePath().relativize(dest.toAbsolutePath());
        } catch (Exception e) {
            return null;
        }
    }

    // =========================================================================
    // Tiny helpers
    // =========================================================================

    private static Scalar scoreColour(double score) {
        return score >= 70 ? new Scalar(0, 200, 0)
             : score >= 40 ? new Scalar(0, 200, 200)
             :               new Scalar(0, 0, 200);
    }

    private static int    scenePx(SceneEntry s) { return s.sceneMat().cols() * s.sceneMat().rows(); }
    private static String sanitise(String s)    { return s.replaceAll("[^A-Za-z0-9_\\-]", "_"); }
    private static String shortName(String v)   {
        return v.replace("HoughLinesP", "HLP")
                .replace("HoughCircles", "HC")
                .replace("_CF_", "·CF·");
    }
}

