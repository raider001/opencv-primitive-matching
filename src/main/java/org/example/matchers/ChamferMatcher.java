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
 * Chamfer Distance Matching — Milestone 18.
 *
 * <p>2 distance types × base / CF_LOOSE / CF_TIGHT = <b>6 variants total</b>.
 *
 * <h2>Algorithm</h2>
 * <ol>
 *   <li>Extract the reference shape's edge contour by binarising and running Canny.</li>
 *   <li>Build a distance-transform map of the scene's edge map once per run using
 *       {@link Imgproc#distanceTransform} with the chosen metric (L1 or L2).</li>
 *   <li>Sample each reference edge pixel into the scene distance map.
 *       The average sampled distance is the Chamfer distance.</li>
 *   <li>Slide a 128×128 search window across the scene at stride {@value #STRIDE}.
 *       For each window, translate the reference edge points to window-local coordinates
 *       and accumulate the distance-map values under them.</li>
 *   <li>Score = {@code 1 / (1 + avgChamferDistance) × 100}. Lower distance = higher score.</li>
 * </ol>
 *
 * <h2>CF variants</h2>
 * The colour mask is applied to both the scene and reference before edge extraction.
 * This isolates only colour-matching edges in both images, reducing false matches
 * from background structure that differs in colour from the target.
 *
 * <h2>Expected behaviour</h2>
 * <ul>
 *   <li>A_CLEAN — very high score; edge map aligns well on clean placements</li>
 *   <li>B_TRANSFORMED — partial score under rotation/scale; distance field tolerates
 *       small offsets better than pixel diff</li>
 *   <li>C_DEGRADED/occlusion — more graceful than pixel diff: missing edge fragments
 *       contribute a bounded distance rather than failing entirely</li>
 *   <li>D_NEGATIVE — non-zero but lower; CF variants greatly reduce false positives from
 *       background edges that happen to be spatially close to the reference edge layout</li>
 * </ul>
 */
public final class ChamferMatcher {

    /** Distance type constant — city-block / L1. */
    public static final int DIST_L1 = Imgproc.DIST_L1;
    /** Distance type constant — Euclidean / L2. */
    public static final int DIST_L2 = Imgproc.DIST_L2;

    /** Canny edge detector lower threshold. */
    private static final double CANNY_LO = 50.0;
    /** Canny edge detector upper threshold. */
    private static final double CANNY_HI = 150.0;

    /** Reference tile size. */
    private static final int TILE   = 128;
    /** Sliding window stride. */
    private static final int STRIDE = 8;

    /** Minimum edge pixel count required in the reference; below this the match is skipped. */
    private static final int MIN_EDGE_PX = 20;

    private ChamferMatcher() {}

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

        // Build reference edge maps for each CF mode
        Mat refEdge      = extractEdges(refMat, null);
        Mat refMaskL     = ColourPreFilter.applyToReference(refMat, referenceId, ColourPreFilter.LOOSE);
        Mat refEdgeLoose = extractEdges(refMat, refMaskL);
        refMaskL.release();
        Mat refMaskT     = ColourPreFilter.applyToReference(refMat, referenceId, ColourPreFilter.TIGHT);
        Mat refEdgeTight = extractEdges(refMat, refMaskT);
        refMaskT.release();

        // Extract reference edge point lists once per mode
        List<Point> refPts      = edgePoints(refEdge);
        List<Point> refPtsLoose = edgePoints(refEdgeLoose);
        List<Point> refPtsTight = edgePoints(refEdgeTight);
        refEdge.release();
        refEdgeLoose.release();
        refEdgeTight.release();

        // CF mask timings for scene
        long t0 = System.currentTimeMillis();
        Mat sceneMaskL = ColourPreFilter.applyToScene(sceneMat, referenceId, ColourPreFilter.LOOSE);
        long cfLMs     = System.currentTimeMillis() - t0;

        t0 = System.currentTimeMillis();
        Mat sceneMaskT = ColourPreFilter.applyToScene(sceneMat, referenceId, ColourPreFilter.TIGHT);
        long cfTMs     = System.currentTimeMillis() - t0;

        // Run all 6 variants (2 dist types × 3 CF modes)
        for (ChamferVariant variant : ChamferVariant.values()) {
            List<Point> rPts;
            Mat         sEdgeMask;
            long        pfMs;

            switch (variant.cfMode()) {
                case LOOSE -> { rPts = refPtsLoose; sEdgeMask = sceneMaskL; pfMs = cfLMs; }
                case TIGHT -> { rPts = refPtsTight; sEdgeMask = sceneMaskT; pfMs = cfTMs; }
                default    -> { rPts = refPts;      sEdgeMask = null;       pfMs = 0L;    }
            }

            out.add(runVariant(variant.variantName(), variant.distType(), rPts,
                    sceneMat, sEdgeMask, pfMs, referenceId, scene, saveVariants, outputDir));
        }

        sceneMaskL.release();
        sceneMaskT.release();

        return out;
    }

    // -------------------------------------------------------------------------
    // Single variant
    // -------------------------------------------------------------------------

    private static AnalysisResult runVariant(String variantName,
                                              int distType,
                                              List<Point> refPts,
                                              Mat sceneMat,
                                              Mat sceneMask,
                                              long preFilterMs,
                                              ReferenceId referenceId,
                                              SceneEntry scene,
                                              Set<String> saveVariants,
                                              Path outputDir) {
        long t0 = System.currentTimeMillis();
        try {
            int sceneW = sceneMat.cols();
            int sceneH = sceneMat.rows();

            if (sceneW < TILE || sceneH < TILE) {
                return AnalysisResult.error(variantName, referenceId,
                        scene.variantLabel(), scene.category(), scene.backgroundId(),
                        0L, scenePx(scene), "Scene smaller than tile");
            }

            if (refPts.size() < MIN_EDGE_PX) {
                // Reference produced almost no edge pixels — probably a low-saturation ref
                // under CF. Return a zero score rather than crashing.
                return new AnalysisResult(variantName, referenceId,
                        scene.variantLabel(), scene.category(), scene.backgroundId(),
                        0.0, new Rect(0, 0, TILE, TILE),
                        System.currentTimeMillis() - t0, preFilterMs,
                        scenePx(scene), null, false, null);
            }

            // Build the full-scene distance transform map (computed once per variant)
            Mat sceneEdge  = extractEdges(sceneMat, sceneMask);
            Mat distMap    = new Mat();
            // distanceTransform requires non-zero pixels to be background (0) and
            // the foreground (edge) pixels to be 0 — so invert the edge map.
            Mat invEdge    = new Mat();
            Core.bitwise_not(sceneEdge, invEdge);
            sceneEdge.release();
            Imgproc.distanceTransform(invEdge, distMap, distType, 3);
            invEdge.release();
            // distMap is CV_32F; values = distance to nearest edge pixel

            double bestScore = -1;
            Rect   bestBbox  = new Rect(0, 0, TILE, TILE);

            for (int wy = 0; wy <= sceneH - TILE; wy += STRIDE) {
                for (int wx = 0; wx <= sceneW - TILE; wx += STRIDE) {
                    double distSum = 0;
                    for (Point rp : refPts) {
                        int sx = (int) (wx + rp.x);
                        int sy = (int) (wy + rp.y);
                        if (sx < 0 || sy < 0 || sx >= sceneW || sy >= sceneH) {
                            distSum += TILE; // penalty for out-of-bounds
                        } else {
                            distSum += distMap.get(sy, sx)[0];
                        }
                    }
                    double avgDist = distSum / refPts.size();
                    double score   = 1.0 / (1.0 + avgDist) * 100.0;
                    if (score > bestScore) {
                        bestScore = score;
                        bestBbox  = new Rect(wx, wy, TILE, TILE);
                    }
                }
            }
            distMap.release();

            long   elapsed    = System.currentTimeMillis() - t0;
            double finalScore = Math.max(0, Math.min(100, bestScore));
            Rect   tightBbox  = tightenBbox(sceneMat, bestBbox);

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
    // Edge extraction helpers
    // -------------------------------------------------------------------------

    /**
     * Converts a BGR image to greyscale (optionally masked), applies Canny edge
     * detection, and returns the binary edge map (CV_8UC1).
     */
    private static Mat extractEdges(Mat bgr, Mat mask) {
        Mat grey = new Mat();
        Imgproc.cvtColor(bgr, grey, Imgproc.COLOR_BGR2GRAY);
        if (mask != null && !mask.empty()) {
            Mat masked = new Mat(grey.size(), grey.type(), Scalar.all(0));
            grey.copyTo(masked, mask);
            grey.release();
            grey = masked;
        }
        Mat edges = new Mat();
        Imgproc.Canny(grey, edges, CANNY_LO, CANNY_HI);
        grey.release();
        return edges;
    }

    /**
     * Collects all non-zero pixel coordinates in a binary edge mat as a list of
     * {@link Point} objects with (x, y) in image coordinates.
     */
    private static List<Point> edgePoints(Mat edges) {
        List<Point> pts = new ArrayList<>();
        for (int r = 0; r < edges.rows(); r++) {
            for (int c = 0; c < edges.cols(); c++) {
                if (edges.get(r, c)[0] > 0) {
                    pts.add(new Point(c, r));
                }
            }
        }
        return pts;
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
            Imgproc.findContours(bin.clone(), contours, new Mat(),
                    Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
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
        return v.replace("CHAMFER_L1_CF_LOOSE", "CH·L1·CFL")
                .replace("CHAMFER_L1_CF_TIGHT", "CH·L1·CFT")
                .replace("CHAMFER_L2_CF_LOOSE", "CH·L2·CFL")
                .replace("CHAMFER_L2_CF_TIGHT", "CH·L2·CFT")
                .replace("CHAMFER_L1",          "CH·L1")
                .replace("CHAMFER_L2",          "CH·L2");
    }
}

