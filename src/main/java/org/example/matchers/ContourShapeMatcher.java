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
 * Contour Shape Matching via Hu Moments — Milestone 9.
 *
 * <p>Pipeline per variant:
 * <ol>
 *   <li>Binarise the 128×128 reference and extract its largest contour.</li>
 *   <li>Binarise the full scene, then run {@link Imgproc#findContours} to get every
 *       external contour in the scene.</li>
 *   <li>Score each scene contour with
 *       {@link Imgproc#matchShapes(Mat, Mat, int, double)} against the reference contour.
 *       Lower value = more similar shape.</li>
 *   <li>The contour with the <em>lowest</em> matchShapes value is the best match.
 *       Its {@link Imgproc#boundingRect} is reported as the bounding box.</li>
 *   <li>Score = {@code 1 / (1 + bestValue) × 100%}.</li>
 * </ol>
 *
 * <p>Variants (11 total = 3 base × 3 modes + 2 CF1):
 * <ul>
 *   <li>{@code CONTOURS_MATCH_I1 / I2 / I3} — base</li>
 *   <li>{@code _CF_LOOSE} / {@code _CF_TIGHT} — colour pre-filter restricts which
 *       pixels are binarised, reducing false contours from background clutter.</li>
 *   <li>{@code CONTOURS_MATCH_I1_CF1_LOOSE} / {@code _CF1_TIGHT} — Colour-First Region
 *       Proposal (Milestone 15): {@link org.example.ColourFirstLocator} proposes candidate
 *       windows; contour matching runs only inside those windows.</li>
 * </ul>
 */
public final class ContourShapeMatcher {

    public static final String[] BASE_METHODS = {
        "CONTOURS_MATCH_I1",
        "CONTOURS_MATCH_I2",
        "CONTOURS_MATCH_I3"
    };

    private static final int[] CV_METHODS = {
        Imgproc.CONTOURS_MATCH_I1,
        Imgproc.CONTOURS_MATCH_I2,
        Imgproc.CONTOURS_MATCH_I3
    };

    /** CF1 variant names (Milestone 15 — Colour-First Region Proposal). */
    public static final String VAR_CF1_LOOSE = "CONTOURS_MATCH_I1_CF1_LOOSE";
    public static final String VAR_CF1_TIGHT = "CONTOURS_MATCH_I1_CF1_TIGHT";

    /** Binary threshold applied before contour extraction. */
    private static final int BINARISE_THRESH = 30;

    /**
     * Minimum contour area (px²) considered for matching.
     * Filters out single-pixel noise contours.
     */
    private static final double MIN_CONTOUR_AREA = 16.0;

    private ContourShapeMatcher() {}

    // -------------------------------------------------------------------------
    // Public entry point
    // -------------------------------------------------------------------------

    public static List<AnalysisResult> match(ReferenceId referenceId,
                                              Mat refMat,
                                              SceneEntry scene,
                                              Set<String> saveVariants,
                                              Path outputDir) {
        List<AnalysisResult> out = new ArrayList<>(11);

        // Pre-binarise reference for all 3 modes
        Mat refBin      = binarise(refMat, null);
        Mat refMaskL    = ColourPreFilter.applyToReference(refMat, referenceId, ColourPreFilter.LOOSE);
        Mat refMaskT    = ColourPreFilter.applyToReference(refMat, referenceId, ColourPreFilter.TIGHT);
        Mat refBinLoose = binarise(refMat, refMaskL);
        Mat refBinTight = binarise(refMat, refMaskT);
        refMaskL.release();
        refMaskT.release();

        // Extract reference contour once per mode — reused across all 3 I1/I2/I3 methods
        Mat sceneMat = scene.sceneMat();

        // Pre-binarise scene for all 3 modes
        Mat sceneBin = binarise(sceneMat, null);

        long t0 = System.currentTimeMillis();
        Mat sceneMaskL = ColourPreFilter.applyToScene(sceneMat, referenceId, ColourPreFilter.LOOSE);
        long cfLMs = System.currentTimeMillis() - t0;
        Mat sceneBinLoose = binarise(sceneMat, sceneMaskL);
        sceneMaskL.release();

        t0 = System.currentTimeMillis();
        Mat sceneMaskT = ColourPreFilter.applyToScene(sceneMat, referenceId, ColourPreFilter.TIGHT);
        long cfTMs = System.currentTimeMillis() - t0;
        Mat sceneBinTight = binarise(sceneMat, sceneMaskT);
        sceneMaskT.release();

        // Extract scene contours once per mode
        List<MatOfPoint> sceneContours      = findExternalContours(sceneBin);
        List<MatOfPoint> sceneContoursLoose = findExternalContours(sceneBinLoose);
        List<MatOfPoint> sceneContoursTight = findExternalContours(sceneBinTight);

        sceneBin.release();
        sceneBinLoose.release();
        sceneBinTight.release();

        for (int i = 0; i < BASE_METHODS.length; i++) {
            String baseName = BASE_METHODS[i];
            int    cvMethod = CV_METHODS[i];

            out.add(runVariant(baseName, cvMethod,
                    sceneMat, sceneContours, refBin,
                    0L, referenceId, scene, saveVariants, outputDir));

            out.add(runVariant(baseName + "_CF_LOOSE", cvMethod,
                    sceneMat, sceneContoursLoose, refBinLoose,
                    cfLMs, referenceId, scene, saveVariants, outputDir));

            out.add(runVariant(baseName + "_CF_TIGHT", cvMethod,
                    sceneMat, sceneContoursTight, refBinTight,
                    cfTMs, referenceId, scene, saveVariants, outputDir));
        }

        // CF1 variants — Colour-First Region Proposal (Milestone 15)
        // Only for I1 as representative (per plan: CF1_CSM_I1_LOOSE / CF1_CSM_I1_TIGHT)
        out.add(runCf1Variant(VAR_CF1_LOOSE, Imgproc.CONTOURS_MATCH_I1,
                sceneMat, refBin, referenceId, ColourPreFilter.LOOSE,
                scene, saveVariants, outputDir));
        out.add(runCf1Variant(VAR_CF1_TIGHT, Imgproc.CONTOURS_MATCH_I1,
                sceneMat, refBin, referenceId, ColourPreFilter.TIGHT,
                scene, saveVariants, outputDir));

        // Release everything
        releaseContours(sceneContours);
        releaseContours(sceneContoursLoose);
        releaseContours(sceneContoursTight);
        refBin.release();
        refBinLoose.release();
        refBinTight.release();

        return out;
    }

    // -------------------------------------------------------------------------
    // CF1 variant — Colour-First Region Proposal (Milestone 15)
    // -------------------------------------------------------------------------

    /**
     * Runs contour matching (I1) restricted to colour-proposed candidate windows.
     *
     * <p>Pipeline:
     * <ol>
     *   <li>{@link ColourFirstLocator#propose} returns candidate windows sorted by area.</li>
     *   <li>For each window: crop the scene, binarise the crop, find contours
     *       (coordinates local to the crop), run {@code matchShapes} vs. the reference binary.</li>
     *   <li>Translate the best contour's bbox back to scene-global coordinates.</li>
     *   <li>Return the window (and contour set) that produces the best score.</li>
     * </ol>
     *
     * <p>If no colour blobs are found {@link ColourFirstLocator} falls back to the full
     * scene rect, so behaviour degrades gracefully to a single full-scene search.
     */
    private static AnalysisResult runCf1Variant(String variantName,
                                                  int cvMethod,
                                                  Mat sceneMat,
                                                  Mat refBinMat,
                                                  ReferenceId referenceId,
                                                  double hueTolerance,
                                                  SceneEntry scene,
                                                  Set<String> saveVariants,
                                                  Path outputDir) {
        long t0 = System.currentTimeMillis();
        try {
            if (refBinMat == null || refBinMat.empty()) {
                return AnalysisResult.error(variantName, referenceId,
                        scene.variantLabel(), scene.category(), scene.backgroundId(),
                        0L, scenePx(scene), "No reference binary image");
            }

            // Propose candidate windows via colour threshold
            long tCf1 = System.currentTimeMillis();
            List<Rect> windows = ColourFirstLocator.propose(sceneMat, referenceId, hueTolerance);
            long preFilterMs = System.currentTimeMillis() - tCf1;

            double bestVal    = Double.MAX_VALUE;
            Rect   bestBbox   = null;
            List<MatOfPoint> bestWindowContours = null;

            for (Rect window : windows) {
                // Clamp window to scene bounds
                int sceneW = sceneMat.cols(), sceneH = sceneMat.rows();
                int wx = Math.max(0, window.x);
                int wy = Math.max(0, window.y);
                int ww = Math.min(window.width,  sceneW - wx);
                int wh = Math.min(window.height, sceneH - wy);
                if (ww <= 0 || wh <= 0) continue;
                Rect safeWin = new Rect(wx, wy, ww, wh);

                // Binarise the cropped region
                Mat crop    = new Mat(sceneMat, safeWin);
                Mat cropBin = binarise(crop, null);
                // Note: crop is a submat (no release needed); cropBin owns its data

                List<MatOfPoint> cropContours = findExternalContours(cropBin);
                cropBin.release();

                double winBestVal   = Double.MAX_VALUE;
                Rect   winBestLocal = null; // bbox in window-local coordinates

                for (MatOfPoint c : cropContours) {
                    if (Imgproc.contourArea(c) < MIN_CONTOUR_AREA) continue;
                    Rect   br       = Imgproc.boundingRect(c);
                    Mat    rendered = renderContour(c, br);
                    double val      = Imgproc.matchShapes(refBinMat, rendered, cvMethod, 0.0);
                    rendered.release();
                    if (!Double.isFinite(val)) continue;
                    if (val < winBestVal) {
                        winBestVal   = val;
                        winBestLocal = br;
                    }
                }

                if (winBestVal < bestVal) {
                    bestVal = winBestVal;
                    if (winBestLocal != null) {
                        // Translate from window-local to scene-global coordinates
                        Rect globalBr = new Rect(
                                safeWin.x + winBestLocal.x,
                                safeWin.y + winBestLocal.y,
                                winBestLocal.width,
                                winBestLocal.height);
                        // Translate all contours to scene-global for unionNearbyContours
                        for (MatOfPoint c : cropContours) {
                            org.opencv.core.Point[] pts = c.toArray();
                            for (org.opencv.core.Point p : pts) {
                                p.x += safeWin.x;
                                p.y += safeWin.y;
                            }
                            c.fromArray(pts);
                        }
                        if (bestWindowContours != null) releaseContours(bestWindowContours);
                        bestWindowContours = cropContours;
                        bestBbox = globalBr;
                    } else {
                        releaseContours(cropContours);
                    }
                } else {
                    releaseContours(cropContours);
                }
            }

            long   elapsed = System.currentTimeMillis() - t0;
            double score;
            if (bestBbox == null) {
                score    = 0.0;
                bestBbox = new Rect(0, 0, 1, 1);
            } else {
                score = (1.0 / (1.0 + bestVal)) * 100.0;
                score = Math.max(0, Math.min(100, score));
                // Expand bbox to union nearby contours within the winning window
                if (bestWindowContours != null) {
                    int refW = refBinMat.cols();
                    int refH = refBinMat.rows();
                    bestBbox = unionNearbyContours(bestWindowContours, bestBbox, refW, refH);
                }
            }

            if (bestWindowContours != null) releaseContours(bestWindowContours);

            Path savedPath = null;
            if (saveVariants.contains(variantName)) {
                savedPath = writeAnnotated(sceneMat, bestBbox, variantName, score,
                        referenceId, scene, outputDir);
            }

            return new AnalysisResult(variantName, referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    score, bestBbox, elapsed, preFilterMs,
                    scenePx(scene), savedPath, false, null);

        } catch (Exception e) {
            return AnalysisResult.error(variantName, referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    System.currentTimeMillis() - t0, scenePx(scene), e.getMessage());
        }
    }

    // -------------------------------------------------------------------------
    // Single variant — contour scan
    // -------------------------------------------------------------------------

    private static AnalysisResult runVariant(String variantName,
                                              int cvMethod,
                                              Mat sceneMat,
                                              List<MatOfPoint> sceneContours,
                                              Mat refBinMat,
                                              long preFilterMs,
                                              ReferenceId referenceId,
                                              SceneEntry scene,
                                              Set<String> saveVariants,
                                              Path outputDir) {
        long t0 = System.currentTimeMillis();
        try {
            if (refBinMat == null || refBinMat.empty()) {
                return AnalysisResult.error(variantName, referenceId,
                        scene.variantLabel(), scene.category(), scene.backgroundId(),
                        0L, scenePx(scene), "No reference binary image");
            }

            double bestVal  = Double.MAX_VALUE;
            Rect   bestBbox = null;

            // matchShapes on raw MatOfPoint can return Infinity for smooth contours (I1).
            // Render each contour to a small binary Mat and compare Mats instead —
            // moment computation from pixel areas is numerically stable for all methods.
            for (MatOfPoint c : sceneContours) {
                if (Imgproc.contourArea(c) < MIN_CONTOUR_AREA) continue;
                Rect   br       = Imgproc.boundingRect(c);
                Mat    rendered = renderContour(c, br);
                double val      = Imgproc.matchShapes(refBinMat, rendered, cvMethod, 0.0);
                rendered.release();
                if (!Double.isFinite(val)) continue;
                if (val < bestVal) {
                    bestVal  = val;
                    bestBbox = br;
                }
            }

            long   elapsed = System.currentTimeMillis() - t0;
            double score;
            if (bestBbox == null) {
                score    = 0.0;
                bestBbox = new Rect(0, 0, 1, 1);
            } else {
                score = (1.0 / (1.0 + bestVal)) * 100.0;
                score = Math.max(0, Math.min(100, score));
                // Expand the bounding box to union all contours in the neighbourhood.
                // The best contour may only be a fragment (arc) of the full shape, so we
                // collect every contour whose bounding rect overlaps an expected-size
                // search region centred on the best contour and union them all.
                int refW = refBinMat.cols();
                int refH = refBinMat.rows();
                bestBbox = unionNearbyContours(sceneContours, bestBbox, refW, refH);
            }

            Path savedPath = null;
            if (saveVariants.contains(variantName)) {
                savedPath = writeAnnotated(sceneMat, bestBbox, variantName, score,
                        referenceId, scene, outputDir);
            }

            return new AnalysisResult(variantName, referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    score, bestBbox, elapsed, preFilterMs,
                    scenePx(scene), savedPath, false, null);

        } catch (Exception e) {
            return AnalysisResult.error(variantName, referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    System.currentTimeMillis() - t0, scenePx(scene), e.getMessage());
        }
    }

    // -------------------------------------------------------------------------
    // Contour helpers
    // -------------------------------------------------------------------------

    /**
     * Renders a single contour onto a fresh binary Mat sized to its bounding rect,
     * shifting the contour to local (0,0)-based coordinates.
     * Used to compare contour shapes via the Mat overload of {@code matchShapes},
     * which is numerically more stable than the contour-point overload for I1.
     */
    private static Mat renderContour(MatOfPoint contour, Rect br) {
        Mat rendered = Mat.zeros(br.height, br.width, CvType.CV_8UC1);
        // Shift contour points to local coordinates
        MatOfPoint shifted = new MatOfPoint();
        org.opencv.core.Point[] pts = contour.toArray();
        for (org.opencv.core.Point p : pts) {
            p.x -= br.x;
            p.y -= br.y;
        }
        shifted.fromArray(pts);
        List<MatOfPoint> cList = new ArrayList<>();
        cList.add(shifted);
        Imgproc.drawContours(rendered, cList, 0, new Scalar(255), Core.FILLED);
        shifted.release();
        return rendered;
    }

    /**
     * Starting from the best-matching contour's bounding rect, unions all scene contours
     * whose bounding rects intersect an expanded search region.
     *
     * <p>The search region is centred on {@code seed} and has dimensions
     * {@code refW × refH} (the reference image size) — this is the expected footprint
     * of the target shape in the scene.  Any contour fragment that falls inside this
     * region is folded into the final bounding box, reconstructing the full shape bbox
     * even when the contour was broken into arcs by background overlap.
     *
     * @param contours  all external contours found in the scene
     * @param seed      bounding rect of the best-scoring individual contour
     * @param refW      reference image width  (expected shape footprint width)
     * @param refH      reference image height (expected shape footprint height)
     * @return union of all contour bounding rects within the neighbourhood
     */
    private static Rect unionNearbyContours(List<MatOfPoint> contours,
                                             Rect seed, int refW, int refH) {
        // Centre of the seed contour
        int cx = seed.x + seed.width  / 2;
        int cy = seed.y + seed.height / 2;

        // Search region: reference-sized window centred on the seed
        int sx = cx - refW / 2;
        int sy = cy - refH / 2;
        Rect searchRegion = new Rect(sx, sy, refW, refH);

        int x1 = seed.x, y1 = seed.y;
        int x2 = seed.x + seed.width, y2 = seed.y + seed.height;

        for (MatOfPoint c : contours) {
            if (Imgproc.contourArea(c) < MIN_CONTOUR_AREA) continue;
            Rect br = Imgproc.boundingRect(c);
            // Include this contour if its bbox intersects the search region
            if (rectsIntersect(br, searchRegion)) {
                x1 = Math.min(x1, br.x);
                y1 = Math.min(y1, br.y);
                x2 = Math.max(x2, br.x + br.width);
                y2 = Math.max(y2, br.y + br.height);
            }
        }
        return new Rect(x1, y1, x2 - x1, y2 - y1);
    }

    /** Returns true if two rects overlap (touching counts as overlap). */
    private static boolean rectsIntersect(Rect a, Rect b) {
        return a.x < b.x + b.width  && a.x + a.width  > b.x
            && a.y < b.y + b.height && a.y + a.height > b.y;
    }

    /** Finds all external contours in a binary image. */
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
    // Image helpers
    // -------------------------------------------------------------------------

    /**
     * Converts a BGR Mat to binary (CV_8UC1).
     * If {@code mask} is non-null the grey image is AND-masked before thresholding,
     * restricting visible pixels to the colour-isolated region.
     */
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
            // Only draw the bounding rect if it's a real detection (not the 1×1 fallback)
            if (bbox.width > 1 && bbox.height > 1) {
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
        return v.replace("CONTOURS_MATCH_", "CM_")
                .replace("_CF1_", "·CF1·")
                .replace("_CF_",  "·CF·");
    }
}





















