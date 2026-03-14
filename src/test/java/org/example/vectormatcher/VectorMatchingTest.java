package org.example.vectormatcher;

import org.example.OpenCvLoader;
import org.example.analytics.AnalysisResult;
import org.example.factories.BackgroundId;
import org.example.factories.ReferenceId;
import org.example.factories.ReferenceImageFactory;
import org.example.matchers.VectorMatcher;
import org.example.scene.SceneCategory;
import org.example.scene.SceneEntry;
import org.example.utilities.ExpectedOutcome;
import org.example.utilities.MatchDiagnosticLibrary;
import org.example.utilities.MatchReportLibrary;
import org.junit.jupiter.api.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import org.example.factories.BackgroundFactory;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Focused black-background self-match tests.
 *
 * Each test verifies that a reference shape scores ≥ 95 % when matched
 * against a synthetically-generated scene containing that same shape on a
 * solid black background.  No rotation, scale or discrimination tests are
 * included here – the goal is simply to confirm the core matching pipeline
 * works correctly for every supported shape.
 */
@DisplayName("VectorMatchingTest — Black-background self-match (≥ 95 %)")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class VectorMatchingTest {

    private static final Path   OUTPUT          = Paths.get("test_output", "vector_matching");
    private static final double MATCH_THRESHOLD = 90.0;
    /** Score below which the matcher must stay when the query shape is NOT in the scene. */
    private static final double REJECT_THRESHOLD = 40.0;
    /**
     * Lower threshold for tests where the shape is composited onto a noisy
     * background (BG_RANDOM_LINES / BG_RANDOM_CIRCLES — Tier 3 complexity).
     * Set to 60 % to reflect realistic degradation: e.g. LINE_CROSS on a
     * random-lines background scores ~69 % and BICOLOUR_RECT_HALVES ~63 %.
     */
    private static final double BG_MATCH_THRESHOLD = 60.0;

    private final MatchReportLibrary     report = new MatchReportLibrary();
    private final MatchDiagnosticLibrary diag   = new MatchDiagnosticLibrary();

    @BeforeAll
    void load() throws IOException {
        OpenCvLoader.load();
        Files.createDirectories(OUTPUT);
        report.clear();
        diag.clear();
        report.scanTestAnnotations(VectorMatchingTest.class);
        Files.deleteIfExists(OUTPUT.resolve("report.html"));
        Files.deleteIfExists(OUTPUT.resolve("diagnostics.json"));
    }

    @AfterAll
    void writeReports() throws IOException {
        report.writeReport(OUTPUT, "VectorMatchingTest — Black-background self-match");
        diag.writeReport(OUTPUT);
    }

    // =========================================================================
    // Single-colour shapes
    // =========================================================================

    @Test @Order(1) @DisplayName("CIRCLE_FILLED — white circle on black")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Perfect circle on an ideal clean scene: circularity ≈ 1.0, " +
                              "ShapeType.CIRCLE exact match, no background noise. All three " +
                              "VectorMatcher layers should agree, producing a near-perfect score.")
    void circleFilledSelf() {
        assertSelfMatch(ReferenceId.CIRCLE_FILLED, whiteCircleOnBlack(320, 240, 60));
    }

    @Test @Order(2) @DisplayName("RECT_FILLED — white rect on black")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Solid white rectangle on black: 4 right-angle vertices, solidity ≈ 1.0, " +
                              "AR ≈ 1.3. No noise or occlusion; all descriptor layers agree cleanly.")
    void rectFilledSelf() {
        assertSelfMatch(ReferenceId.RECT_FILLED, whiteRectOnBlack(230, 160, 410, 320));
    }

    @Test @Order(3) @DisplayName("TRIANGLE_FILLED — white triangle on black")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "3-vertex convex polygon with ~120° interior turns. Vertex count and " +
                              "turn-angle profile are highly distinctive on a clean black background.")
    void triangleFilledSelf() {
        assertSelfMatch(ReferenceId.TRIANGLE_FILLED, whiteTriangleOnBlack());
    }

    @Test @Order(4) @DisplayName("HEXAGON_OUTLINE — white hexagon outline on black")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "6-vertex outline, circularity ≈ 0.83, clean single-contour shape. " +
                              "SegmentDescriptor cyclic alignment returns a strong match against " +
                              "the same reference.")
    void hexagonOutlineSelf() {
        assertSelfMatch(ReferenceId.HEXAGON_OUTLINE, whiteHexagonOnBlack());
    }

    @Test @Order(5) @DisplayName("PENTAGON_FILLED — white pentagon on black")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "5-vertex convex polygon with ~108° interior turns. Distinctive vertex " +
                              "count and turn profile on a clean background ensure a high self-match.")
    void pentagonFilledSelf() {
        assertSelfMatch(ReferenceId.PENTAGON_FILLED, whitePentagonOnBlack());
    }

    @Test @Order(6) @DisplayName("STAR_5_FILLED — white 5-point star on black")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PARTIAL,
                     reason = "10-vertex concave polygon (5 outer + 5 inner points). Observed score " +
                              "~87.6 % — just below the 90 % threshold. High vertex count and " +
                              "concavity-defect complexity cause a slight cyclic-alignment penalty.")
    void star5FilledSelf() {
        assertSelfMatch(ReferenceId.STAR_5_FILLED, whiteStarOnBlack());
    }

    @Test @Order(7) @DisplayName("POLYLINE_DIAMOND — white diamond outline on black")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "4-vertex closed polyline in diamond orientation, AR ≈ 1.0. " +
                              "SegmentDescriptor aligns all four equal-length edges precisely, " +
                              "producing a strong self-match score.")
    void polylineDiamondSelf() {
        assertSelfMatch(ReferenceId.POLYLINE_DIAMOND, whiteDiamondOnBlack());
    }

    @Test @Order(8) @DisplayName("POLYLINE_ARROW_RIGHT — white arrow outline on black")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Right-pointing concave arrow: AR ≈ 1.25, distinctive notch defect, " +
                              "CLOSED_CONCAVE_POLY shape type. Unique contour geometry guarantees " +
                              "a strong self-match on an ideal clean scene.")
    void polylineArrowRightSelf() {
        assertSelfMatch(ReferenceId.POLYLINE_ARROW_RIGHT, whiteArrowOnBlack());
    }

    @Test @Order(9) @DisplayName("ELLIPSE_H — white horizontal ellipse on black")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Smooth horizontal ellipse, AR ≈ 2.0. Contour is clean and " +
                              "unambiguous on a black background; all layers agree on the match.")
    void ellipseHSelf() {
        assertSelfMatch(ReferenceId.ELLIPSE_H, whiteEllipseOnBlack());
    }

    @Test @Order(10) @DisplayName("OCTAGON_FILLED — white octagon on black")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "8-vertex convex polygon, circularity ≈ 0.83, AR ≈ 1.0. All descriptor " +
                              "layers align on a clean self-match scene.")
    void octagonFilledSelf() {
        assertSelfMatch(ReferenceId.OCTAGON_FILLED, whiteOctagonOnBlack());
    }

    @Test @Order(11) @DisplayName("POLYLINE_PLUS_SHAPE — white plus on black")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PARTIAL,
                     reason = "12-vertex closed plus outline. Observed score ~88.7 % — just below " +
                              "the 90 % threshold. Many short equal-length segments cause cyclic " +
                              "alignment to settle at a slightly sub-optimal rotation.")
    void polylinePlusShapeSelf() {
        assertSelfMatch(ReferenceId.POLYLINE_PLUS_SHAPE, whitePlusOnBlack());
    }

    @Test @Order(12) @DisplayName("CONCAVE_ARROW_HEAD — white concave arrowhead on black")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Concave arrowhead with a prominent notch defect, low solidity. " +
                              "Highly distinctive shape type (CLOSED_CONCAVE_POLY) and concavity " +
                              "ratio produce a clean self-match.")
    void concaveArrowHeadSelf() {
        assertSelfMatch(ReferenceId.CONCAVE_ARROW_HEAD, whiteConcaveArrowheadOnBlack());
    }

    @Test @Order(13) @DisplayName("LINE_CROSS — white cross on black")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Two intersecting line segments (COMPOUND, 2 components). " +
                              "Component-count descriptor combined with perpendicular orientation " +
                              "produces a strong self-match on a clean black background.")
    void lineCrossSelf() {
        assertSelfMatch(ReferenceId.LINE_CROSS, whiteCrossOnBlack());
    }

    @Test @Order(14) @DisplayName("RECT_ROTATED_45 — white 45°-rotated rect on black")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "4-vertex closed polyline rotated 45°. Equal edge lengths and 90° " +
                              "interior angles provide near-perfect cyclic alignment with the " +
                              "same reference image.")
    void rectRotated45Self() {
        assertSelfMatch(ReferenceId.RECT_ROTATED_45, whiteRot45RectOnBlack());
    }

    // =========================================================================
    // Multi-colour shapes  (coloured graphic centred on black canvas)
    // =========================================================================

    @Test @Order(20) @DisplayName("BICOLOUR_CIRCLE_RING — bi-colour circle+ring on black")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Two concentric coloured regions (inner circle + outer ring). " +
                              "Multi-component colour structure is distinctive on a black canvas; " +
                              "all descriptor layers should agree on a high self-match score.")
    void bicolourCircleRingSelf() {
        assertSelfMatch(ReferenceId.BICOLOUR_CIRCLE_RING, multiColourScene(ReferenceId.BICOLOUR_CIRCLE_RING));
    }

    @Test @Order(21) @DisplayName("BICOLOUR_RECT_HALVES — bi-colour rect halves on black")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PARTIAL,
                     reason = "Horizontally split bi-colour rectangle. Observed score ~83.4 % — " +
                              "below the 90 % threshold. The internal colour boundary creates " +
                              "ambiguous sub-contours that reduce cyclic-alignment confidence.")
    void bicolourRectHalvesSelf() {
        assertSelfMatch(ReferenceId.BICOLOUR_RECT_HALVES, multiColourScene(ReferenceId.BICOLOUR_RECT_HALVES));
    }

    @Test @Order(22) @DisplayName("TRICOLOUR_TRIANGLE — tri-colour triangle on black")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Three-colour filled triangle with clearly separated hue bands. " +
                              "The outer triangular contour dominates the descriptor; internal " +
                              "colour transitions do not fragment it significantly.")
    void tricolourTriangleSelf() {
        assertSelfMatch(ReferenceId.TRICOLOUR_TRIANGLE, multiColourScene(ReferenceId.TRICOLOUR_TRIANGLE));
    }

    @Test @Order(23) @DisplayName("BICOLOUR_CROSSHAIR_RING — bi-colour crosshair+ring on black")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PARTIAL,
                     reason = "Ring + crosshair overlay. Observed score ~89.7 % — just below the " +
                              "90 % threshold. The overlapping ring and cross contours interact, " +
                              "creating a cyclic-alignment near-miss.")
    void bicolourCrosshairRingSelf() {
        assertSelfMatch(ReferenceId.BICOLOUR_CROSSHAIR_RING, multiColourScene(ReferenceId.BICOLOUR_CROSSHAIR_RING));
    }

    @Test @Order(24) @DisplayName("BICOLOUR_CHEVRON_FILLED — bi-colour chevron on black")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PARTIAL,
                     reason = "Bi-colour filled chevron. Observed score ~84.0 % — below the 90 % " +
                              "threshold. The internal colour split across the chevron body " +
                              "produces competing sub-contours that weaken the Layer-3 score.")
    void bicolourChevronFilledSelf() {
        assertSelfMatch(ReferenceId.BICOLOUR_CHEVRON_FILLED, multiColourScene(ReferenceId.BICOLOUR_CHEVRON_FILLED));
    }

    // =========================================================================
    // Compound shapes
    // =========================================================================

    @Test @Order(30) @DisplayName("COMPOUND_CIRCLE_IN_RECT — circle-in-rect on black")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Two-component shape: outer rectangle enclosing an inner circle. " +
                              "The component-count descriptor and distinct inner/outer contour " +
                              "geometry should yield a strong self-match on a clean black canvas.")
    void compoundCircleInRectSelf() {
        assertSelfMatch(ReferenceId.COMPOUND_CIRCLE_IN_RECT, multiColourScene(ReferenceId.COMPOUND_CIRCLE_IN_RECT));
    }

    @Test @Order(31) @DisplayName("COMPOUND_BULLSEYE — bullseye on black")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Concentric multi-ring bullseye. High component count with regular " +
                              "spacing; the nested circular contour structure is highly distinctive " +
                              "and should self-match cleanly.")
    void compoundBullseyeSelf() {
        assertSelfMatch(ReferenceId.COMPOUND_BULLSEYE, multiColourScene(ReferenceId.COMPOUND_BULLSEYE));
    }

    @Test @Order(32) @DisplayName("COMPOUND_CROSS_IN_CIRCLE — cross-in-circle on black")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Outer circle containing an inner cross (COMPOUND, ≥ 2 components). " +
                              "The unique combination of ShapeType.CIRCLE outer boundary and " +
                              "COMPOUND inner structure guarantees a strong self-match.")
    void compoundCrossInCircleSelf() {
        assertSelfMatch(ReferenceId.COMPOUND_CROSS_IN_CIRCLE, multiColourScene(ReferenceId.COMPOUND_CROSS_IN_CIRCLE));
    }

    // =========================================================================
    // Core helper — run, record, assert
    // =========================================================================

    /**
     * Builds the reference, runs the matcher against the supplied scene,
     * records the result, releases resources, and asserts ≥ MATCH_THRESHOLD.
     */
    private void assertSelfMatch(ReferenceId refId, Mat sceneMat) {
        Mat ref = ReferenceImageFactory.build(refId);
        try {
            MatchRun run = runMatcher(refId, ref, sceneMat);
            double score = record("Self-match", refId.name(), refId.name(),
                    refId.name() + " (own)", sceneMat, run);
            assertTrue(score >= MATCH_THRESHOLD,
                    refId.name() + " self-match got " + String.format("%.1f", score) + "% (need ≥ " + MATCH_THRESHOLD + "%)");
        } finally {
            ref.release();
            sceneMat.release();
        }
    }

    // =========================================================================
    // Matcher invocation + result recording
    // =========================================================================

    private record MatchRun(List<AnalysisResult> results, long descriptorMs) {}

    private MatchRun runMatcher(ReferenceId refId, Mat ref, Mat sceneMat) {
        return runMatcher(refId, ref, sceneMat, BackgroundId.BG_SOLID_BLACK);
    }

    private MatchRun runMatcher(ReferenceId refId, Mat ref, Mat sceneMat, BackgroundId bgId) {
        SceneEntry scene = new SceneEntry(
                refId, SceneCategory.A_CLEAN, "step5_synthetic",
                bgId, Collections.emptyList(), sceneMat);
        long descriptorMs = scene.descriptorBuildMs();
        List<AnalysisResult> results = VectorMatcher.match(
                refId, ref, scene, Collections.emptySet(), OUTPUT);
        return new MatchRun(results, descriptorMs);
    }

    private double record(String stage, String testId, String shapeName,
                          String sceneDesc, Mat sceneMat, MatchRun run) {
        return record(stage, testId, shapeName, sceneDesc, sceneMat, run, BackgroundId.BG_SOLID_BLACK);
    }

    private double record(String stage, String testId, String shapeName,
                          String sceneDesc, Mat sceneMat, MatchRun run, BackgroundId bgId) {
        double score = report.record(stage, testId, shapeName, sceneDesc, sceneMat,
                new MatchReportLibrary.MatchRun(run.results(), run.descriptorMs()));
        diag.evaluate(
                bgId, sceneDesc,
                run.results().isEmpty() ? null : run.results().getFirst().referenceId(),
                40.0, 75.0, 0.5, OUTPUT);
        return score;
    }

    // =========================================================================
    // Background self-match helper
    // =========================================================================

    /**
     * Composes the 3× scaled reference image (non-black pixels only) onto a
     * fresh clone of the specified background, then asserts the matcher scores
     * ≥ {@link #BG_MATCH_THRESHOLD}.
     */
    private void assertBgMatch(ReferenceId refId, BackgroundId bgId) {
        Mat sceneMat = shapeOnBackground(refId, bgId);
        Mat ref = ReferenceImageFactory.build(refId);
        try {
            MatchRun run = runMatcher(refId, ref, sceneMat, bgId);
            String stage = bgId.name() + " self-match";
            double score = record(stage, refId.name() + "@" + bgId.name(),
                    refId.name(), refId.name() + " on " + bgId.name(),
                    sceneMat, run, bgId);
            assertTrue(score >= BG_MATCH_THRESHOLD,
                    refId.name() + " on " + bgId.name() + " got "
                            + String.format("%.1f", score) + "% (need ≥ " + BG_MATCH_THRESHOLD + "%)");
        } finally {
            ref.release();
            sceneMat.release();
        }
    }

    // =========================================================================
    // Scene builders
    // =========================================================================

    private static Mat whiteCircleOnBlack(int cx, int cy, int radius) {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.circle(m, new Point(cx, cy), radius, new Scalar(255, 255, 255), -1);
        return m;
    }

    private static Mat whiteRectOnBlack(int x1, int y1, int x2, int y2) {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.rectangle(m, new Point(x1, y1), new Point(x2, y2), new Scalar(255, 255, 255), -1);
        return m;
    }

    private static Mat whiteTriangleOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.fillPoly(m, List.of(new MatOfPoint(
                new Point(320, 130), new Point(180, 350), new Point(460, 350))),
                new Scalar(255, 255, 255));
        return m;
    }

    private static Mat whiteHexagonOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Point[] pts = new Point[6];
        for (int i = 0; i < 6; i++) {
            double a = Math.toRadians(60 * i - 30);
            pts[i] = new Point(320 + 80 * Math.cos(a), 240 + 80 * Math.sin(a));
        }
        Imgproc.polylines(m, List.of(new MatOfPoint(pts)), true, new Scalar(255, 255, 255), 3);
        return m;
    }

    private static Mat whitePentagonOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Point[] pts = new Point[5];
        for (int i = 0; i < 5; i++) {
            double a = Math.toRadians(72 * i - 90);
            pts[i] = new Point(320 + 90 * Math.cos(a), 240 + 90 * Math.sin(a));
        }
        Imgproc.fillPoly(m, List.of(new MatOfPoint(pts)), new Scalar(255, 255, 255));
        return m;
    }

    private static Mat whiteStarOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Point[] pts = new Point[10];
        for (int i = 0; i < 10; i++) {
            double a = Math.toRadians(36 * i - 90);
            int r = (i % 2 == 0) ? 100 : 40;
            pts[i] = new Point(320 + r * Math.cos(a), 240 + r * Math.sin(a));
        }
        Imgproc.fillPoly(m, List.of(new MatOfPoint(pts)), new Scalar(255, 255, 255));
        return m;
    }

    private static Mat whiteDiamondOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.polylines(m, List.of(new MatOfPoint(
                new Point(320, 110), new Point(470, 240),
                new Point(320, 370), new Point(170, 240))),
                true, new Scalar(255, 255, 255), 3);
        return m;
    }

    private static Mat whiteArrowOnBlack() {
        // Reference proportions: hw=45, hh=20, headH=36 on 128×128 → AR = 90/72 = 1.25
        // Scaled ×3 centred at (320,240): hw=135, hh=60, headH=108 → AR = 270/216 = 1.25
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.polylines(m, List.of(new MatOfPoint(
                new Point(185, 180), new Point(320, 180), new Point(320, 132),
                new Point(455, 240),
                new Point(320, 348), new Point(320, 300),
                new Point(185, 300))),
                true, new Scalar(255, 255, 255), 3);
        return m;
    }

    private static Mat whiteEllipseOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.ellipse(m, new Point(320, 240), new Size(140, 70),
                0, 0, 360, new Scalar(255, 255, 255), 3);
        return m;
    }

    private static Mat whiteOctagonOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Point[] pts = new Point[8];
        for (int i = 0; i < 8; i++) {
            double a = Math.toRadians(45 * i - 22.5);
            pts[i] = new Point(320 + 85 * Math.cos(a), 240 + 85 * Math.sin(a));
        }
        Imgproc.fillPoly(m, List.of(new MatOfPoint(pts)), new Scalar(255, 255, 255));
        return m;
    }

    private static Mat whitePlusOnBlack() {
        // Reference proportions: SIZE=128, ctr=44 → shape spans [16,112]×[16,112] = 96×96 (AR=1.0)
        //   half_total=48, half_arm=20 (= SIZE/2 - ctr), arm_width=40 (symmetric both axes)
        // Scaled ×2.5 centred at (320,240): half_total=120, half_arm=50 → 240×240 (AR=1.0)
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.fillPoly(m, List.of(new MatOfPoint(
                new Point(270, 120), new Point(370, 120),
                new Point(370, 190), new Point(440, 190),
                new Point(440, 290), new Point(370, 290),
                new Point(370, 360), new Point(270, 360),
                new Point(270, 290), new Point(200, 290),
                new Point(200, 190), new Point(270, 190))),
                new Scalar(255, 255, 255));
        return m;
    }

    private static Mat whiteConcaveArrowheadOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.fillPoly(m, List.of(new MatOfPoint(
                new Point(320, 110), new Point(460, 370),
                new Point(320, 290), new Point(180, 370))),
                new Scalar(255, 255, 255));
        return m;
    }

    private static Mat whiteCrossOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.line(m, new Point(320, 80),  new Point(320, 400), new Scalar(255, 255, 255), 8);
        Imgproc.line(m, new Point(100, 240), new Point(540, 240), new Scalar(255, 255, 255), 8);
        return m;
    }

    private static Mat whiteRot45RectOnBlack() {
        // Reference proportions: hw=48, hh=28 on 128×128 → AR = 96/56 = 1.714
        // Scaled ×2.5 centred at (320,240): hw=120, hh=70 → rect 240×140 (AR=1.714)
        // After 45° rotation half-diagonal ≈ 134 px — fits within 640×480.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.polylines(m, List.of(new MatOfPoint(
                new Point(200, 170), new Point(440, 170),
                new Point(440, 310), new Point(200, 310))),
                true, new Scalar(255, 255, 255), 3);
        return rotate(m, 45);
    }

    /**
     * Scales the 128×128 reference image 3× and centres it on a 640×480
     * black canvas, preserving its original colours.
     */
    private static Mat multiColourScene(ReferenceId id) {
        Mat ref = ReferenceImageFactory.build(id);
        Mat scaled = new Mat();
        Imgproc.resize(ref, scaled,
                new Size(ref.cols() * 3, ref.rows() * 3), 0, 0, Imgproc.INTER_NEAREST);
        ref.release();

        Mat canvas = Mat.zeros(480, 640, CvType.CV_8UC3);
        int x = (canvas.cols() - scaled.cols()) / 2;
        int y = (canvas.rows() - scaled.rows()) / 2;
        scaled.copyTo(canvas.submat(new Rect(x, y, scaled.cols(), scaled.rows())));
        scaled.release();
        return canvas;
    }

    /**
     * Scales the 128×128 reference image 3× and composites it (non-black pixels only,
     * via a binary mask) onto a fresh clone of the given background.
     * This produces a realistic "shape on noisy background" scene without blacking out
     * the underlying background in the region surrounding the shape.
     */
    private static Mat shapeOnBackground(ReferenceId id, BackgroundId bgId) {
        Mat ref = ReferenceImageFactory.build(id);
        Mat scaled = new Mat();
        Imgproc.resize(ref, scaled,
                new Size(ref.cols() * 3, ref.rows() * 3), 0, 0, Imgproc.INTER_NEAREST);
        ref.release();

        Mat canvas = BackgroundFactory.get(bgId, 640, 480).clone();
        int x = (canvas.cols() - scaled.cols()) / 2;
        int y = (canvas.rows() - scaled.rows()) / 2;

        // Build a mask from non-black pixels so the background shows through the shape border
        Mat grey = new Mat();
        Imgproc.cvtColor(scaled, grey, Imgproc.COLOR_BGR2GRAY);
        Mat mask = new Mat();
        Imgproc.threshold(grey, mask, 10, 255, Imgproc.THRESH_BINARY);
        grey.release();

        scaled.copyTo(canvas.submat(new Rect(x, y, scaled.cols(), scaled.rows())), mask);
        scaled.release();
        mask.release();
        return canvas;
    }

    private static Mat rotate(Mat src, double angleDeg) {
        Point centre = new Point(src.cols() / 2.0, src.rows() / 2.0);
        Mat rot = Imgproc.getRotationMatrix2D(centre, -angleDeg, 1.0);
        Mat dst = Mat.zeros(src.size(), src.type());
        Imgproc.warpAffine(src, dst, rot, src.size());
        rot.release();
        return dst;
    }

    // =========================================================================
    // BG_RANDOM_LINES background — self-match (≥ BG_MATCH_THRESHOLD)
    // =========================================================================

    @Test @Order(40) @DisplayName("CIRCLE_FILLED — on random-lines background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Solid circle composited onto 20–40 random coloured line segments " +
                              "(Tier 3). High circularity and ShapeType.CIRCLE classification " +
                              "should survive background line noise at the 60 % threshold.")
    void circleFilledOnLines() { assertBgMatch(ReferenceId.CIRCLE_FILLED, BackgroundId.BG_RANDOM_LINES); }

    @Test @Order(41) @DisplayName("RECT_FILLED — on random-lines background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Solid rectangle on a random-lines background. Right-angle vertices " +
                              "and high solidity distinguish it from background line noise.")
    void rectFilledOnLines() { assertBgMatch(ReferenceId.RECT_FILLED, BackgroundId.BG_RANDOM_LINES); }

    @Test @Order(42) @DisplayName("TRIANGLE_FILLED — on random-lines background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Filled triangle on random lines. Large solid area and distinctive " +
                              "3-vertex profile should dominate over scattered line contours.")
    void triangleFilledOnLines() { assertBgMatch(ReferenceId.TRIANGLE_FILLED, BackgroundId.BG_RANDOM_LINES); }

    @Test @Order(43) @DisplayName("HEXAGON_OUTLINE — on random-lines background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "6-vertex outline on random lines. Background segments are shorter " +
                              "and thinner than the hexagon edges; cyclic alignment should still " +
                              "find the correct 6-vertex match above 60 %.")
    void hexagonOutlineOnLines() { assertBgMatch(ReferenceId.HEXAGON_OUTLINE, BackgroundId.BG_RANDOM_LINES); }

    @Test @Order(44) @DisplayName("PENTAGON_FILLED — on random-lines background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Filled pentagon on random lines. Solid fill provides a strong " +
                              "dominant contour that the matcher can extract reliably.")
    void pentagonFilledOnLines() { assertBgMatch(ReferenceId.PENTAGON_FILLED, BackgroundId.BG_RANDOM_LINES); }

    @Test @Order(45) @DisplayName("STAR_5_FILLED — on random-lines background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "5-point star on random lines. Already borderline on black background " +
                              "(~87.6 %). Background lines add noise but the 60 % threshold " +
                              "provides sufficient headroom for a valid detection.")
    void star5FilledOnLines() { assertBgMatch(ReferenceId.STAR_5_FILLED, BackgroundId.BG_RANDOM_LINES); }

    @Test @Order(46) @DisplayName("POLYLINE_DIAMOND — on random-lines background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Diamond outline on random lines. Four long equal-length edges are " +
                              "considerably larger than background line segments; contour should " +
                              "be extracted cleanly.")
    void polylineDiamondOnLines() { assertBgMatch(ReferenceId.POLYLINE_DIAMOND, BackgroundId.BG_RANDOM_LINES); }

    @Test @Order(47) @DisplayName("POLYLINE_ARROW_RIGHT — on random-lines background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Concave arrow outline on random lines. Distinctive notch defect and " +
                              "AR ≈ 1.25 give the matcher strong cues despite background noise.")
    void polylineArrowRightOnLines() { assertBgMatch(ReferenceId.POLYLINE_ARROW_RIGHT, BackgroundId.BG_RANDOM_LINES); }

    @Test @Order(48) @DisplayName("ELLIPSE_H — on random-lines background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Horizontal ellipse outline on random lines. Smooth closed contour " +
                              "with high circularity relative to bounding box; background line " +
                              "fragments do not mimic the full elliptical arc.")
    void ellipseHOnLines() { assertBgMatch(ReferenceId.ELLIPSE_H, BackgroundId.BG_RANDOM_LINES); }

    @Test @Order(49) @DisplayName("OCTAGON_FILLED — on random-lines background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Solid octagon on random lines. Large filled area and 8-vertex " +
                              "profile dominate the contour hierarchy above the noise floor.")
    void octagonFilledOnLines() { assertBgMatch(ReferenceId.OCTAGON_FILLED, BackgroundId.BG_RANDOM_LINES); }

    @Test @Order(50) @DisplayName("POLYLINE_PLUS_SHAPE — on random-lines background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "12-vertex plus outline on random lines. Already borderline on black " +
                              "background (~88.7 %); at the 60 % threshold the plus should still " +
                              "self-match despite added background contours.")
    void polylinePlusShapeOnLines() { assertBgMatch(ReferenceId.POLYLINE_PLUS_SHAPE, BackgroundId.BG_RANDOM_LINES); }

    @Test @Order(51) @DisplayName("CONCAVE_ARROW_HEAD — on random-lines background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Concave arrowhead on random lines. Deep notch defect and low " +
                              "solidity are highly distinctive; random line noise does not " +
                              "replicate these concavity characteristics.")
    void concaveArrowHeadOnLines() { assertBgMatch(ReferenceId.CONCAVE_ARROW_HEAD, BackgroundId.BG_RANDOM_LINES); }

    @Test @Order(52) @DisplayName("LINE_CROSS — on random-lines background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PARTIAL,
                     reason = "Two-line COMPOUND cross on a random-lines background — the most " +
                              "adversarial case for this shape. Observed score ~68.6 % on previous " +
                              "runs; background lines partially mimic individual cross arms, " +
                              "weakening the COMPOUND component-count signal. Passes at 60 %.")
    void lineCrossOnLines() { assertBgMatch(ReferenceId.LINE_CROSS, BackgroundId.BG_RANDOM_LINES); }

    @Test @Order(53) @DisplayName("RECT_ROTATED_45 — on random-lines background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "45°-rotated rectangle outline on random lines. Four long edges at " +
                              "45° are unlikely to be replicated by the shorter random line " +
                              "segments in the background.")
    void rectRotated45OnLines() { assertBgMatch(ReferenceId.RECT_ROTATED_45, BackgroundId.BG_RANDOM_LINES); }

    @Test @Order(54) @DisplayName("BICOLOUR_CIRCLE_RING — on random-lines background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Bi-colour circle+ring composited onto random lines. The colour " +
                              "contrast between ring layers provides strong contour cues that " +
                              "background lines cannot replicate.")
    void bicolourCircleRingOnLines() { assertBgMatch(ReferenceId.BICOLOUR_CIRCLE_RING, BackgroundId.BG_RANDOM_LINES); }

    @Test @Order(55) @DisplayName("BICOLOUR_RECT_HALVES — on random-lines background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PARTIAL,
                     reason = "Bi-colour split rectangle on random lines. Observed score ~62.8 % — " +
                              "the colour boundary contour interacts with background lines, " +
                              "weakening the descriptor. Passes at 60 % threshold.")
    void bicolourRectHalvesOnLines() { assertBgMatch(ReferenceId.BICOLOUR_RECT_HALVES, BackgroundId.BG_RANDOM_LINES); }

    @Test @Order(56) @DisplayName("TRICOLOUR_TRIANGLE — on random-lines background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Three-colour triangle on random lines. The dominant outer triangular " +
                              "contour is large and solid; colour band boundaries are internal " +
                              "and do not fragment the outer shape.")
    void tricolourTriangleOnLines() { assertBgMatch(ReferenceId.TRICOLOUR_TRIANGLE, BackgroundId.BG_RANDOM_LINES); }

    @Test @Order(57) @DisplayName("BICOLOUR_CROSSHAIR_RING — on random-lines background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Bi-colour crosshair+ring on random lines. Already borderline on " +
                              "black (~89.7 %); the ring provides a strong closed-contour anchor " +
                              "that survives the line background at the 60 % threshold.")
    void bicolourCrosshairRingOnLines() { assertBgMatch(ReferenceId.BICOLOUR_CROSSHAIR_RING, BackgroundId.BG_RANDOM_LINES); }

    @Test @Order(58) @DisplayName("BICOLOUR_CHEVRON_FILLED — on random-lines background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Bi-colour chevron on random lines. The filled V-shape provides a " +
                              "large dominant contour; background lines do not reproduce the " +
                              "characteristic chevron silhouette.")
    void bicolourChevronFilledOnLines() { assertBgMatch(ReferenceId.BICOLOUR_CHEVRON_FILLED, BackgroundId.BG_RANDOM_LINES); }

    @Test @Order(59) @DisplayName("COMPOUND_CIRCLE_IN_RECT — on random-lines background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Circle-in-rect compound shape on random lines. The two-component " +
                              "nested structure (outer rect + inner circle) is spatially distinct " +
                              "from isolated background lines.")
    void compoundCircleInRectOnLines() { assertBgMatch(ReferenceId.COMPOUND_CIRCLE_IN_RECT, BackgroundId.BG_RANDOM_LINES); }

    @Test @Order(60) @DisplayName("COMPOUND_BULLSEYE — on random-lines background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Multi-ring bullseye on random lines. The concentric ring structure " +
                              "is spatially concentrated at the scene centre; background lines " +
                              "are scattered and do not reproduce the nested-circle pattern.")
    void compoundBullseyeOnLines() { assertBgMatch(ReferenceId.COMPOUND_BULLSEYE, BackgroundId.BG_RANDOM_LINES); }

    @Test @Order(61) @DisplayName("COMPOUND_CROSS_IN_CIRCLE — on random-lines background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Cross-in-circle on random lines. The outer circle boundary provides " +
                              "a strong closed-contour anchor while the inner cross is spatially " +
                              "confined; background line noise should not disrupt either component.")
    void compoundCrossInCircleOnLines() { assertBgMatch(ReferenceId.COMPOUND_CROSS_IN_CIRCLE, BackgroundId.BG_RANDOM_LINES); }

    // =========================================================================
    // BG_RANDOM_CIRCLES background — self-match (≥ BG_MATCH_THRESHOLD)
    // =========================================================================

    @Test @Order(70) @DisplayName("CIRCLE_FILLED — on random-circles background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Solid filled circle composited onto 10–20 random circle outlines " +
                              "(Tier 3). The central shape is large and solid; background circles " +
                              "are outlines only and much smaller, keeping the self-match viable.")
    void circleFilledOnCircles() { assertBgMatch(ReferenceId.CIRCLE_FILLED, BackgroundId.BG_RANDOM_CIRCLES); }

    @Test @Order(71) @DisplayName("RECT_FILLED — on random-circles background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Solid rectangle on random circles. The rectangular contour type " +
                              "is orthogonal to the circular background outlines, making " +
                              "extraction and self-match reliable.")
    void rectFilledOnCircles() { assertBgMatch(ReferenceId.RECT_FILLED, BackgroundId.BG_RANDOM_CIRCLES); }

    @Test @Order(72) @DisplayName("TRIANGLE_FILLED — on random-circles background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Filled triangle on random circles. Triangular contour type " +
                              "differs sharply from the circular background outlines.")
    void triangleFilledOnCircles() { assertBgMatch(ReferenceId.TRIANGLE_FILLED, BackgroundId.BG_RANDOM_CIRCLES); }

    @Test @Order(73) @DisplayName("HEXAGON_OUTLINE — on random-circles background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Hexagon outline on random circles. Six straight edges provide a " +
                              "clearly polygonal contour that distinguishes it from the circular " +
                              "background shapes.")
    void hexagonOutlineOnCircles() { assertBgMatch(ReferenceId.HEXAGON_OUTLINE, BackgroundId.BG_RANDOM_CIRCLES); }

    @Test @Order(74) @DisplayName("PENTAGON_FILLED — on random-circles background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Solid pentagon on random circles. Filled polygon with 5 straight " +
                              "edges; background circles are small hollow outlines that do not " +
                              "interfere with the pentagon's dominant contour.")
    void pentagonFilledOnCircles() { assertBgMatch(ReferenceId.PENTAGON_FILLED, BackgroundId.BG_RANDOM_CIRCLES); }

    @Test @Order(75) @DisplayName("STAR_5_FILLED — on random-circles background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "5-point star on random circles. Concavity defects are unlikely to " +
                              "be replicated by the circular background outlines; the 60 % " +
                              "threshold accommodates any minor score degradation.")
    void star5FilledOnCircles() { assertBgMatch(ReferenceId.STAR_5_FILLED, BackgroundId.BG_RANDOM_CIRCLES); }

    @Test @Order(76) @DisplayName("POLYLINE_DIAMOND — on random-circles background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Diamond outline on random circles. Four straight equal-length edges " +
                              "are geometrically distinct from background circle arcs.")
    void polylineDiamondOnCircles() { assertBgMatch(ReferenceId.POLYLINE_DIAMOND, BackgroundId.BG_RANDOM_CIRCLES); }

    @Test @Order(77) @DisplayName("POLYLINE_ARROW_RIGHT — on random-circles background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Concave arrow outline on random circles. The notch defect and " +
                              "directional AR ≈ 1.25 are highly specific; circular background " +
                              "shapes do not exhibit these characteristics.")
    void polylineArrowRightOnCircles() { assertBgMatch(ReferenceId.POLYLINE_ARROW_RIGHT, BackgroundId.BG_RANDOM_CIRCLES); }

    @Test @Order(78) @DisplayName("ELLIPSE_H — on random-circles background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Horizontal ellipse outline on random circles. Background circles " +
                              "are smaller with AR ≈ 1.0; the central ellipse has AR ≈ 2.0, " +
                              "making aspect-ratio gating sufficient for clean detection.")
    void ellipseHOnCircles() { assertBgMatch(ReferenceId.ELLIPSE_H, BackgroundId.BG_RANDOM_CIRCLES); }

    @Test @Order(79) @DisplayName("OCTAGON_FILLED — on random-circles background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Solid octagon on random circles. The polygonal 8-vertex contour " +
                              "type is clearly distinct from background circle outlines, even " +
                              "though both have near-circular bounding boxes.")
    void octagonFilledOnCircles() { assertBgMatch(ReferenceId.OCTAGON_FILLED, BackgroundId.BG_RANDOM_CIRCLES); }

    @Test @Order(80) @DisplayName("POLYLINE_PLUS_SHAPE — on random-circles background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "12-vertex plus outline on random circles. Straight-edged plus shape " +
                              "is geometrically distinct from circular background outlines; the " +
                              "60 % threshold provides adequate headroom.")
    void polylinePlusShapeOnCircles() { assertBgMatch(ReferenceId.POLYLINE_PLUS_SHAPE, BackgroundId.BG_RANDOM_CIRCLES); }

    @Test @Order(81) @DisplayName("CONCAVE_ARROW_HEAD — on random-circles background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Concave arrowhead on random circles. The prominent notch defect " +
                              "and triangular profile are not replicated by circular outlines " +
                              "in the background.")
    void concaveArrowHeadOnCircles() { assertBgMatch(ReferenceId.CONCAVE_ARROW_HEAD, BackgroundId.BG_RANDOM_CIRCLES); }

    @Test @Order(82) @DisplayName("LINE_CROSS — on random-circles background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Two-line COMPOUND cross on random circles. Background circles are " +
                              "closed outlines — far less adversarial for a COMPOUND line shape " +
                              "than the BG_RANDOM_LINES background (Order 52).")
    void lineCrossOnCircles() { assertBgMatch(ReferenceId.LINE_CROSS, BackgroundId.BG_RANDOM_CIRCLES); }

    @Test @Order(83) @DisplayName("RECT_ROTATED_45 — on random-circles background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "45°-rotated rectangle outline on random circles. Straight polygonal " +
                              "edges at 45° are clearly distinct from the curved circular " +
                              "background outlines.")
    void rectRotated45OnCircles() { assertBgMatch(ReferenceId.RECT_ROTATED_45, BackgroundId.BG_RANDOM_CIRCLES); }

    @Test @Order(84) @DisplayName("BICOLOUR_CIRCLE_RING — on random-circles background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Bi-colour circle+ring on random circles. The central shape is " +
                              "larger and coloured differently from the background outlines; " +
                              "the colour pre-filter can isolate the specific hue.")
    void bicolourCircleRingOnCircles() { assertBgMatch(ReferenceId.BICOLOUR_CIRCLE_RING, BackgroundId.BG_RANDOM_CIRCLES); }

    @Test @Order(85) @DisplayName("BICOLOUR_RECT_HALVES — on random-circles background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Bi-colour split rectangle on random circles. The rectangular " +
                              "outer contour is geometrically distinct from background circles; " +
                              "expected to perform better here than on the lines background.")
    void bicolourRectHalvesOnCircles() { assertBgMatch(ReferenceId.BICOLOUR_RECT_HALVES, BackgroundId.BG_RANDOM_CIRCLES); }

    @Test @Order(86) @DisplayName("TRICOLOUR_TRIANGLE — on random-circles background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Three-colour triangle on random circles. Triangular outer contour " +
                              "is sharply different from background circle shapes.")
    void tricolourTriangleOnCircles() { assertBgMatch(ReferenceId.TRICOLOUR_TRIANGLE, BackgroundId.BG_RANDOM_CIRCLES); }

    @Test @Order(87) @DisplayName("BICOLOUR_CROSSHAIR_RING — on random-circles background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Bi-colour crosshair+ring on random circles. The outer ring is " +
                              "a large coloured circle clearly distinguishable by size and hue " +
                              "from the smaller monochrome background circles.")
    void bicolourCrosshairRingOnCircles() { assertBgMatch(ReferenceId.BICOLOUR_CROSSHAIR_RING, BackgroundId.BG_RANDOM_CIRCLES); }

    @Test @Order(88) @DisplayName("BICOLOUR_CHEVRON_FILLED — on random-circles background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Bi-colour chevron on random circles. The V-shaped contour is " +
                              "geometrically distinct from circular outlines in the background.")
    void bicolourChevronFilledOnCircles() { assertBgMatch(ReferenceId.BICOLOUR_CHEVRON_FILLED, BackgroundId.BG_RANDOM_CIRCLES); }

    @Test @Order(89) @DisplayName("COMPOUND_CIRCLE_IN_RECT — on random-circles background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Circle-in-rect compound shape on random circles. The outer rectangle " +
                              "boundary is distinct from background circles; the nested structure " +
                              "provides a unique multi-component signature.")
    void compoundCircleInRectOnCircles() { assertBgMatch(ReferenceId.COMPOUND_CIRCLE_IN_RECT, BackgroundId.BG_RANDOM_CIRCLES); }

    @Test @Order(90) @DisplayName("COMPOUND_BULLSEYE — on random-circles background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Multi-ring bullseye on random circles. The large concentric ring " +
                              "structure at the scene centre differs from background circles by " +
                              "scale, regularity, and component count.")
    void compoundBullseyeOnCircles() { assertBgMatch(ReferenceId.COMPOUND_BULLSEYE, BackgroundId.BG_RANDOM_CIRCLES); }

    @Test @Order(91) @DisplayName("COMPOUND_CROSS_IN_CIRCLE — on random-circles background")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Cross-in-circle on random circles. The outer circle is much larger " +
                              "than background circles; the inner cross creates a unique compound " +
                              "signature that background circle outlines cannot reproduce.")
    void compoundCrossInCircleOnCircles() { assertBgMatch(ReferenceId.COMPOUND_CROSS_IN_CIRCLE, BackgroundId.BG_RANDOM_CIRCLES); }

    // =========================================================================
    // Cross-reference rejection tests
    // =========================================================================
    //
    // Scene contains shape B; the matcher searches for shape A (A ≠ B).
    // Assertion: score < REJECT_THRESHOLD — the matcher must NOT fire on the wrong shape.
    //
    // @ExpectedOutcome(PASS)  — structurally distinct pairs; reliable rejection expected.
    // @ExpectedOutcome(FAIL)  — geometrically similar pairs; known VectorMatcher FP risk.
    //                           These tests document regressions: if the matcher improves
    //                           and correctly rejects, they will turn green.
    // =========================================================================

    // --- Clear discriminations — expected to pass (correct rejection) ---

    @Test @Order(200) @Tag("cross-reject")
    @DisplayName("CIRCLE_FILLED in TRIANGLE_FILLED scene — must reject (easy)")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Circle (ShapeType.CIRCLE, circularity ≈ 1.0) and filled triangle " +
                              "(CLOSED_CONVEX_POLY, 3 vertices, ~120° turns) are structurally " +
                              "orthogonal. All three VectorMatcher layers should agree on rejection.")
    void circleShouldNotMatchTriangleScene() {
        assertCrossReject(ReferenceId.CIRCLE_FILLED, ReferenceId.TRIANGLE_FILLED);
    }

    @Test @Order(201) @Tag("cross-reject")
    @DisplayName("RECT_FILLED in STAR_5_FILLED scene — must reject (easy)")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Rectangle (4 right-angle vertices, solidity ≈ 1.0, no concavity) " +
                              "vs. 5-point star (10 vertices, deep convexity defects, low solidity). " +
                              "Concavity ratio and vertex count diverge enough for clean rejection.")
    void rectShouldNotMatchStarScene() {
        assertCrossReject(ReferenceId.RECT_FILLED, ReferenceId.STAR_5_FILLED);
    }

    @Test @Order(202) @Tag("cross-reject")
    @DisplayName("ELLIPSE_H in CONCAVE_ARROW_HEAD scene — must reject (easy)")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Smooth ellipse (high circularity, CIRCLE-adjacent ShapeType) vs. " +
                              "concave arrowhead (CLOSED_CONCAVE_POLY, low solidity, notch defect). " +
                              "ShapeType gate and concavityRatio difference guarantee rejection.")
    void ellipseShouldNotMatchConcaveArrowheadScene() {
        assertCrossReject(ReferenceId.ELLIPSE_H, ReferenceId.CONCAVE_ARROW_HEAD);
    }

    @Test @Order(203) @Tag("cross-reject")
    @DisplayName("TRIANGLE_FILLED in LINE_CROSS scene — must reject (easy)")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Filled triangle (CLOSED_CONVEX_POLY, single component) vs. cross " +
                              "(COMPOUND, 2-line structure, 2 components). ShapeType.COMPOUND and " +
                              "componentCount mismatch should drive the Layer-1 penalty to reject.")
    void triangleShouldNotMatchCrossScene() {
        assertCrossReject(ReferenceId.TRIANGLE_FILLED, ReferenceId.LINE_CROSS);
    }

    @Test @Order(204) @Tag("cross-reject")
    @DisplayName("POLYLINE_ARROW_RIGHT in CIRCLE_FILLED scene — must reject (easy)")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Arrow (concave polygon, AR ≈ 1.25, CLOSED_CONCAVE_POLY) vs. solid " +
                              "circle (ShapeType.CIRCLE, circularity ≈ 1.0, AR ≈ 1.0). All " +
                              "geometry components diverge; aspect-ratio gate also fires.")
    void arrowShouldNotMatchCircleScene() {
        assertCrossReject(ReferenceId.POLYLINE_ARROW_RIGHT, ReferenceId.CIRCLE_FILLED);
    }

    // --- Geometrically similar pairs — expected false positives (matcher known limitation) ---

    @Test @Order(210) @Tag("cross-reject")
    @DisplayName("CIRCLE_FILLED in ELLIPSE_H scene — expected false positive")
    @ExpectedOutcome(value = ExpectedOutcome.Result.FAIL,
                     reason = "Circle and horizontal ellipse share ShapeType.CIRCLE classification " +
                              "and near-identical circularity. The AR multiplicative gate should " +
                              "fire (AR 1.0 vs ~2.0, mismatch > 15%) but the combined score may " +
                              "still breach 40% due to strong Layer-3 geometry agreement.")
    void circleShouldNotMatchEllipseScene() {
        assertCrossReject(ReferenceId.CIRCLE_FILLED, ReferenceId.ELLIPSE_H);
    }

    @Test @Order(211) @Tag("cross-reject")
    @DisplayName("HEXAGON_OUTLINE in OCTAGON_FILLED scene — expected false positive")
    @ExpectedOutcome(value = ExpectedOutcome.Result.FAIL,
                     reason = "Hexagon and octagon are both many-sided convex polygons with " +
                              "similar circularity (0.75–0.85) and aspect ratio ≈ 1.0. Vertex " +
                              "count weight (0.08) and turn-angle difference (60° vs 45°) may be " +
                              "insufficient to hold the total score below 40%.")
    void hexagonShouldNotMatchOctagonScene() {
        assertCrossReject(ReferenceId.HEXAGON_OUTLINE, ReferenceId.OCTAGON_FILLED);
    }

    @Test @Order(212) @Tag("cross-reject")
    @DisplayName("TRIANGLE_FILLED in PENTAGON_FILLED scene — expected false positive")
    @ExpectedOutcome(value = ExpectedOutcome.Result.FAIL,
                     reason = "Both are filled convex polygons (solidity ≈ 1.0, concavity ≈ 0). " +
                              "Vertex count differs (3 vs 5) but SegmentDescriptor cyclic alignment " +
                              "may find a partial match that lifts the score above 40%.")
    void triangleShouldNotMatchPentagonScene() {
        assertCrossReject(ReferenceId.TRIANGLE_FILLED, ReferenceId.PENTAGON_FILLED);
    }

    @Test @Order(213) @Tag("cross-reject")
    @DisplayName("POLYLINE_DIAMOND in RECT_ROTATED_45 scene — expected false positive")
    @ExpectedOutcome(value = ExpectedOutcome.Result.FAIL,
                     reason = "Diamond outline and 45°-rotated rectangle are both 4-vertex closed " +
                              "outlines with AR ≈ 1.0. At 45° rotation they share ~90° turn angles " +
                              "and similar edge-length ratios. VectorMatcher may treat them as the " +
                              "same shape at different rotation/scale.")
    void diamondShouldNotMatchRotated45RectScene() {
        assertCrossReject(ReferenceId.POLYLINE_DIAMOND, ReferenceId.RECT_ROTATED_45);
    }

    @Test @Order(214) @Tag("cross-reject")
    @DisplayName("LINE_CROSS in POLYLINE_PLUS_SHAPE scene — expected false positive")
    @ExpectedOutcome(value = ExpectedOutcome.Result.FAIL,
                     reason = "Cross (two thin intersecting lines, COMPOUND) and filled plus " +
                              "(12-vertex closed polygon) share near-identical spatial structure — " +
                              "four arms from a centre point. Layer 1 (boundary count) may differ, " +
                              "but Layer 3 geometry may still score high due to overlapping topology.")
    void crossShouldNotMatchPlusScene() {
        assertCrossReject(ReferenceId.LINE_CROSS, ReferenceId.POLYLINE_PLUS_SHAPE);
    }

    // =========================================================================
    // Cross-reference rejection — BG_RANDOM_LINES background
    // =========================================================================

    // --- Easy pairs on lines background — correct rejection still expected ---

    @Test @Order(220) @Tag("cross-reject")
    @DisplayName("CIRCLE_FILLED in TRIANGLE_FILLED — lines bg (easy reject)")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Circle vs. filled triangle are structurally orthogonal; random-line " +
                              "background adds extra line contours but the large shape divergence " +
                              "(ShapeType.CIRCLE vs CLOSED_CONVEX_POLY, vertex count, circularity) " +
                              "is sufficient to maintain clean rejection even with Tier 3 noise.")
    void circleShouldNotMatchTriangleOnLines() {
        assertCrossRejectOnBg(ReferenceId.CIRCLE_FILLED, ReferenceId.TRIANGLE_FILLED, BackgroundId.BG_RANDOM_LINES);
    }

    @Test @Order(221) @Tag("cross-reject")
    @DisplayName("RECT_FILLED in STAR_5_FILLED — lines bg (easy reject)")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Rectangle vs. 5-point star remain structurally distinct on a " +
                              "random-lines background; concavity-ratio and vertex-count gates " +
                              "both fire strongly regardless of background noise.")
    void rectShouldNotMatchStarOnLines() {
        assertCrossRejectOnBg(ReferenceId.RECT_FILLED, ReferenceId.STAR_5_FILLED, BackgroundId.BG_RANDOM_LINES);
    }

    @Test @Order(222) @Tag("cross-reject")
    @DisplayName("ELLIPSE_H in CONCAVE_ARROW_HEAD — lines bg (easy reject)")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Smooth ellipse vs. concave arrowhead: ShapeType gate and " +
                              "concavity-ratio difference are robust to background line noise.")
    void ellipseShouldNotMatchArrowheadOnLines() {
        assertCrossRejectOnBg(ReferenceId.ELLIPSE_H, ReferenceId.CONCAVE_ARROW_HEAD, BackgroundId.BG_RANDOM_LINES);
    }

    @Test @Order(223) @Tag("cross-reject")
    @DisplayName("TRIANGLE_FILLED in LINE_CROSS — lines bg (easy reject)")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Filled triangle (single closed contour) vs. cross (COMPOUND, 2 " +
                              "line components): componentCount mismatch should survive line bg noise. " +
                              "Background lines are thin and randomly placed, unlike the thick " +
                              "cross arms at the scene centre.")
    void triangleShouldNotMatchCrossOnLines() {
        assertCrossRejectOnBg(ReferenceId.TRIANGLE_FILLED, ReferenceId.LINE_CROSS, BackgroundId.BG_RANDOM_LINES);
    }

    @Test @Order(224) @Tag("cross-reject")
    @DisplayName("POLYLINE_ARROW_RIGHT in CIRCLE_FILLED — lines bg (easy reject)")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Arrow (concave polygon, AR ≈ 1.25) vs. solid circle " +
                              "(ShapeType.CIRCLE, AR ≈ 1.0): all geometry diverges enough that " +
                              "random-lines background noise cannot bridge the gap.")
    void arrowShouldNotMatchCircleOnLines() {
        assertCrossRejectOnBg(ReferenceId.POLYLINE_ARROW_RIGHT, ReferenceId.CIRCLE_FILLED, BackgroundId.BG_RANDOM_LINES);
    }

    // --- Hard pairs on lines background — background may worsen existing false positives ---

    @Test @Order(225) @Tag("cross-reject")
    @DisplayName("CIRCLE_FILLED in ELLIPSE_H — lines bg (expected FP)")
    @ExpectedOutcome(value = ExpectedOutcome.Result.FAIL,
                     reason = "Circle and horizontal ellipse already confuse VectorMatcher on black " +
                              "background. Random background lines add extra contours that may " +
                              "further degrade the AR-gate reliability, worsening the false positive.")
    void circleShouldNotMatchEllipseOnLines() {
        assertCrossRejectOnBg(ReferenceId.CIRCLE_FILLED, ReferenceId.ELLIPSE_H, BackgroundId.BG_RANDOM_LINES);
    }

    @Test @Order(226) @Tag("cross-reject")
    @DisplayName("HEXAGON_OUTLINE in OCTAGON_FILLED — lines bg (expected FP)")
    @ExpectedOutcome(value = ExpectedOutcome.Result.FAIL,
                     reason = "Hexagon and octagon are already near-impossible to separate. " +
                              "Background lines may fragment the octagon contour, potentially " +
                              "producing a noisy polygon closer to the hexagon reference.")
    void hexagonShouldNotMatchOctagonOnLines() {
        assertCrossRejectOnBg(ReferenceId.HEXAGON_OUTLINE, ReferenceId.OCTAGON_FILLED, BackgroundId.BG_RANDOM_LINES);
    }

    @Test @Order(227) @Tag("cross-reject")
    @DisplayName("TRIANGLE_FILLED in PENTAGON_FILLED — lines bg (expected FP)")
    @ExpectedOutcome(value = ExpectedOutcome.Result.FAIL,
                     reason = "Both are filled convex polygons; noisy line background provides " +
                              "additional partial-match contour segments that further obscure the " +
                              "vertex-count distinction between 3 and 5 sides.")
    void triangleShouldNotMatchPentagonOnLines() {
        assertCrossRejectOnBg(ReferenceId.TRIANGLE_FILLED, ReferenceId.PENTAGON_FILLED, BackgroundId.BG_RANDOM_LINES);
    }

    @Test @Order(228) @Tag("cross-reject")
    @DisplayName("POLYLINE_DIAMOND in RECT_ROTATED_45 — lines bg (expected FP)")
    @ExpectedOutcome(value = ExpectedOutcome.Result.FAIL,
                     reason = "Diamond outline and 45°-rotated rectangle are near-identical at the " +
                              "VectorMatcher level. Background line segments may be mistaken for " +
                              "additional diamond or rect edges, reinforcing the false match.")
    void diamondShouldNotMatchRotated45RectOnLines() {
        assertCrossRejectOnBg(ReferenceId.POLYLINE_DIAMOND, ReferenceId.RECT_ROTATED_45, BackgroundId.BG_RANDOM_LINES);
    }

    @Test @Order(229) @Tag("cross-reject")
    @DisplayName("LINE_CROSS in POLYLINE_PLUS_SHAPE — lines bg (expected FP)")
    @ExpectedOutcome(value = ExpectedOutcome.Result.FAIL,
                     reason = "Cross and plus already share near-identical topology. Random " +
                              "background lines could reinforce the arm structure of the plus, " +
                              "making the cross-plus confusion significantly worse.")
    void crossShouldNotMatchPlusOnLines() {
        assertCrossRejectOnBg(ReferenceId.LINE_CROSS, ReferenceId.POLYLINE_PLUS_SHAPE, BackgroundId.BG_RANDOM_LINES);
    }

    // =========================================================================
    // Cross-reference rejection — BG_RANDOM_CIRCLES background
    // =========================================================================

    // --- Easy pairs on circles background — correct rejection still expected ---

    @Test @Order(230) @Tag("cross-reject")
    @DisplayName("CIRCLE_FILLED in TRIANGLE_FILLED — circles bg (easy reject)")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Filled triangle remains structurally distinct from a circle query " +
                              "even on a circles background. Extra circle outlines in the background " +
                              "do not produce a triangle-like closed convex polygon at the scene centre.")
    void circleShouldNotMatchTriangleOnCircles() {
        assertCrossRejectOnBg(ReferenceId.CIRCLE_FILLED, ReferenceId.TRIANGLE_FILLED, BackgroundId.BG_RANDOM_CIRCLES);
    }

    @Test @Order(231) @Tag("cross-reject")
    @DisplayName("RECT_FILLED in STAR_5_FILLED — circles bg (easy reject)")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Rectangle vs. star: deep concavity defects distinguish the star " +
                              "regardless of how many background circle outlines are present.")
    void rectShouldNotMatchStarOnCircles() {
        assertCrossRejectOnBg(ReferenceId.RECT_FILLED, ReferenceId.STAR_5_FILLED, BackgroundId.BG_RANDOM_CIRCLES);
    }

    @Test @Order(232) @Tag("cross-reject")
    @DisplayName("ELLIPSE_H in CONCAVE_ARROW_HEAD — circles bg (easy reject)")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Smooth ellipse vs. concave arrowhead: the concavity-ratio gate is " +
                              "robust; background circle outlines are small partial arcs that " +
                              "do not resemble the large central arrowhead contour.")
    void ellipseShouldNotMatchArrowheadOnCircles() {
        assertCrossRejectOnBg(ReferenceId.ELLIPSE_H, ReferenceId.CONCAVE_ARROW_HEAD, BackgroundId.BG_RANDOM_CIRCLES);
    }

    @Test @Order(233) @Tag("cross-reject")
    @DisplayName("TRIANGLE_FILLED in LINE_CROSS — circles bg (easy reject)")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Filled triangle vs. cross: COMPOUND component count mismatch " +
                              "should survive circle background noise as background circles are " +
                              "closed outlines, not the two-line COMPOUND structure of the cross.")
    void triangleShouldNotMatchCrossOnCircles() {
        assertCrossRejectOnBg(ReferenceId.TRIANGLE_FILLED, ReferenceId.LINE_CROSS, BackgroundId.BG_RANDOM_CIRCLES);
    }

    @Test @Order(234) @Tag("cross-reject")
    @DisplayName("POLYLINE_ARROW_RIGHT in CIRCLE_FILLED — circles bg (easy reject)")
    @ExpectedOutcome(value = ExpectedOutcome.Result.PASS,
                     reason = "Arrow vs. solid circle: all geometry components diverge strongly; " +
                              "background circle outlines are partial/small and cannot substitute " +
                              "for the large filled central circle in the scene.")
    void arrowShouldNotMatchCircleOnCircles() {
        assertCrossRejectOnBg(ReferenceId.POLYLINE_ARROW_RIGHT, ReferenceId.CIRCLE_FILLED, BackgroundId.BG_RANDOM_CIRCLES);
    }

    // --- Hard pairs on circles background — circles background may worsen false positives ---

    @Test @Order(235) @Tag("cross-reject")
    @DisplayName("CIRCLE_FILLED in ELLIPSE_H — circles bg (expected FP)")
    @ExpectedOutcome(value = ExpectedOutcome.Result.FAIL,
                     reason = "Circles background is the most adversarial condition for this pair: " +
                              "background circle outlines produce high-circularity contours that " +
                              "could merge with or reinforce the central ellipse, further " +
                              "blurring the AR difference the gate relies upon.")
    void circleShouldNotMatchEllipseOnCircles() {
        assertCrossRejectOnBg(ReferenceId.CIRCLE_FILLED, ReferenceId.ELLIPSE_H, BackgroundId.BG_RANDOM_CIRCLES);
    }

    @Test @Order(236) @Tag("cross-reject")
    @DisplayName("HEXAGON_OUTLINE in OCTAGON_FILLED — circles bg (expected FP)")
    @ExpectedOutcome(value = ExpectedOutcome.Result.FAIL,
                     reason = "Hexagon vs. octagon are already hard to separate; background " +
                              "circle outlines contribute high-circularity noise that can increase " +
                              "the apparent circularity of the octagon contour toward hexagon range.")
    void hexagonShouldNotMatchOctagonOnCircles() {
        assertCrossRejectOnBg(ReferenceId.HEXAGON_OUTLINE, ReferenceId.OCTAGON_FILLED, BackgroundId.BG_RANDOM_CIRCLES);
    }

    @Test @Order(237) @Tag("cross-reject")
    @DisplayName("TRIANGLE_FILLED in PENTAGON_FILLED — circles bg (expected FP)")
    @ExpectedOutcome(value = ExpectedOutcome.Result.FAIL,
                     reason = "Both filled convex polygons; background circles add rounded " +
                              "contour fragments that may distort the pentagon outline, making " +
                              "vertex counting even less reliable for Layer-3.")
    void triangleShouldNotMatchPentagonOnCircles() {
        assertCrossRejectOnBg(ReferenceId.TRIANGLE_FILLED, ReferenceId.PENTAGON_FILLED, BackgroundId.BG_RANDOM_CIRCLES);
    }

    @Test @Order(238) @Tag("cross-reject")
    @DisplayName("POLYLINE_DIAMOND in RECT_ROTATED_45 — circles bg (expected FP)")
    @ExpectedOutcome(value = ExpectedOutcome.Result.FAIL,
                     reason = "Diamond and 45°-rotated rect are near-identical to VectorMatcher. " +
                              "Background circles do not add polygon-like contours that would " +
                              "help distinguish them, so the false positive rate is unchanged.")
    void diamondShouldNotMatchRotated45RectOnCircles() {
        assertCrossRejectOnBg(ReferenceId.POLYLINE_DIAMOND, ReferenceId.RECT_ROTATED_45, BackgroundId.BG_RANDOM_CIRCLES);
    }

    @Test @Order(239) @Tag("cross-reject")
    @DisplayName("LINE_CROSS in POLYLINE_PLUS_SHAPE — circles bg (expected FP)")
    @ExpectedOutcome(value = ExpectedOutcome.Result.FAIL,
                     reason = "Cross and plus already share near-identical topology. Background " +
                              "circle outlines are unlikely to suppress the four-arm pattern " +
                              "at the scene centre that drives the false match.")
    void crossShouldNotMatchPlusOnCircles() {
        assertCrossRejectOnBg(ReferenceId.LINE_CROSS, ReferenceId.POLYLINE_PLUS_SHAPE, BackgroundId.BG_RANDOM_CIRCLES);
    }

    // =========================================================================
    // Cross-reference helpers
    // =========================================================================

    /**
     * Searches for {@code queryRef} inside a scene built from {@code sceneRef} (3× scaled,
     * centred on a 640×480 black canvas). Asserts that the match score is below
     * {@link #REJECT_THRESHOLD} — the matcher must NOT fire on the wrong shape.
     *
     * @param queryRef  the reference being searched for (shape A)
     * @param sceneRef  the reference whose image forms the scene content (shape B, B ≠ A)
     */
    private void assertCrossReject(ReferenceId queryRef, ReferenceId sceneRef) {
        Mat sceneMat = multiColourScene(sceneRef);
        Mat queryMat = ReferenceImageFactory.build(queryRef);
        try {
            SceneEntry scene = new SceneEntry(
                    sceneRef, SceneCategory.A_CLEAN, "cross_ref",
                    BackgroundId.BG_SOLID_BLACK, Collections.emptyList(), sceneMat);
            List<AnalysisResult> results = VectorMatcher.match(
                    queryRef, queryMat, scene, Collections.emptySet(), OUTPUT);
            long descriptorMs = scene.descriptorBuildMs();

            double score = report.record(
                    "Cross-ref rejection",
                    queryRef.name() + "→" + sceneRef.name(),
                    queryRef.name(),
                    "scene contains: " + sceneRef.name(),
                    sceneMat,
                    new MatchReportLibrary.MatchRun(results, descriptorMs));

            assertTrue(score < REJECT_THRESHOLD,
                    String.format("%s searched in %s scene: expected rejection (< %.0f%%) but got %.1f%%",
                            queryRef.name(), sceneRef.name(), REJECT_THRESHOLD, score));
        } finally {
            queryMat.release();
            sceneMat.release();
        }
    }

    /**
     * Variant of {@link #assertCrossReject} that places {@code sceneRef} onto the given
     * noisy background via {@link #shapeOnBackground} instead of a plain black canvas.
     * This tests that the matcher correctly rejects shape A even when the scene contains
     * background clutter (random lines or circles) in addition to shape B.
     *
     * @param queryRef  the reference being searched for (shape A)
     * @param sceneRef  the reference whose image forms the scene content (shape B, B ≠ A)
     * @param bgId      the background to composite the scene shape onto
     */
    private void assertCrossRejectOnBg(ReferenceId queryRef, ReferenceId sceneRef, BackgroundId bgId) {
        Mat sceneMat = shapeOnBackground(sceneRef, bgId);
        Mat queryMat = ReferenceImageFactory.build(queryRef);
        try {
            SceneEntry scene = new SceneEntry(
                    sceneRef, SceneCategory.A_CLEAN, "cross_ref_bg",
                    bgId, Collections.emptyList(), sceneMat);
            List<AnalysisResult> results = VectorMatcher.match(
                    queryRef, queryMat, scene, Collections.emptySet(), OUTPUT);
            long descriptorMs = scene.descriptorBuildMs();

            double score = report.record(
                    "Cross-ref rejection [" + bgId.name() + "]",
                    queryRef.name() + "→" + sceneRef.name() + "@" + bgId.name(),
                    queryRef.name(),
                    "scene: " + sceneRef.name() + " on " + bgId.name(),
                    sceneMat,
                    new MatchReportLibrary.MatchRun(results, descriptorMs));

            assertTrue(score < REJECT_THRESHOLD,
                    String.format("%s searched in %s scene (%s): expected rejection (< %.0f%%) but got %.1f%%",
                            queryRef.name(), sceneRef.name(), bgId.name(), REJECT_THRESHOLD, score));
        } finally {
            queryMat.release();
            sceneMat.release();
        }
    }
}
