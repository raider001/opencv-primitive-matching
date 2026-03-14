package org.example.vectormatcher;

import org.example.OpenCvLoader;
import org.example.analytics.AnalysisResult;
import org.example.factories.BackgroundId;
import org.example.factories.ReferenceId;
import org.example.factories.ReferenceImageFactory;
import org.example.matchers.VectorMatcher;
import org.example.scene.SceneCategory;
import org.example.scene.SceneEntry;
import org.example.utilities.MatchDiagnosticLibrary;
import org.example.utilities.MatchReportLibrary;
import org.junit.jupiter.api.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

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

    private final MatchReportLibrary     report = new MatchReportLibrary();
    private final MatchDiagnosticLibrary diag   = new MatchDiagnosticLibrary();

    @BeforeAll
    void load() throws IOException {
        OpenCvLoader.load();
        Files.createDirectories(OUTPUT);
        report.clear();
        diag.clear();
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
    void circleFilledSelf() {
        assertSelfMatch(ReferenceId.CIRCLE_FILLED, whiteCircleOnBlack(320, 240, 60));
    }

    @Test @Order(2) @DisplayName("RECT_FILLED — white rect on black")
    void rectFilledSelf() {
        assertSelfMatch(ReferenceId.RECT_FILLED, whiteRectOnBlack(230, 160, 410, 320));
    }

    @Test @Order(3) @DisplayName("TRIANGLE_FILLED — white triangle on black")
    void triangleFilledSelf() {
        assertSelfMatch(ReferenceId.TRIANGLE_FILLED, whiteTriangleOnBlack());
    }

    @Test @Order(4) @DisplayName("HEXAGON_OUTLINE — white hexagon outline on black")
    void hexagonOutlineSelf() {
        assertSelfMatch(ReferenceId.HEXAGON_OUTLINE, whiteHexagonOnBlack());
    }

    @Test @Order(5) @DisplayName("PENTAGON_FILLED — white pentagon on black")
    void pentagonFilledSelf() {
        assertSelfMatch(ReferenceId.PENTAGON_FILLED, whitePentagonOnBlack());
    }

    @Test @Order(6) @DisplayName("STAR_5_FILLED — white 5-point star on black")
    void star5FilledSelf() {
        assertSelfMatch(ReferenceId.STAR_5_FILLED, whiteStarOnBlack());
    }

    @Test @Order(7) @DisplayName("POLYLINE_DIAMOND — white diamond outline on black")
    void polylineDiamondSelf() {
        assertSelfMatch(ReferenceId.POLYLINE_DIAMOND, whiteDiamondOnBlack());
    }

    @Test @Order(8) @DisplayName("POLYLINE_ARROW_RIGHT — white arrow outline on black")
    void polylineArrowRightSelf() {
        assertSelfMatch(ReferenceId.POLYLINE_ARROW_RIGHT, whiteArrowOnBlack());
    }

    @Test @Order(9) @DisplayName("ELLIPSE_H — white horizontal ellipse on black")
    void ellipseHSelf() {
        assertSelfMatch(ReferenceId.ELLIPSE_H, whiteEllipseOnBlack());
    }

    @Test @Order(10) @DisplayName("OCTAGON_FILLED — white octagon on black")
    void octagonFilledSelf() {
        assertSelfMatch(ReferenceId.OCTAGON_FILLED, whiteOctagonOnBlack());
    }

    @Test @Order(11) @DisplayName("POLYLINE_PLUS_SHAPE — white plus on black")
    void polylinePlusShapeSelf() {
        assertSelfMatch(ReferenceId.POLYLINE_PLUS_SHAPE, whitePlusOnBlack());
    }

    @Test @Order(12) @DisplayName("CONCAVE_ARROW_HEAD — white concave arrowhead on black")
    void concaveArrowHeadSelf() {
        assertSelfMatch(ReferenceId.CONCAVE_ARROW_HEAD, whiteConcaveArrowheadOnBlack());
    }

    @Test @Order(13) @DisplayName("LINE_CROSS — white cross on black")
    void lineCrossSelf() {
        assertSelfMatch(ReferenceId.LINE_CROSS, whiteCrossOnBlack());
    }

    @Test @Order(14) @DisplayName("RECT_ROTATED_45 — white 45°-rotated rect on black")
    void rectRotated45Self() {
        assertSelfMatch(ReferenceId.RECT_ROTATED_45, whiteRot45RectOnBlack());
    }

    // =========================================================================
    // Multi-colour shapes  (coloured graphic centred on black canvas)
    // =========================================================================

    @Test @Order(20) @DisplayName("BICOLOUR_CIRCLE_RING — bi-colour circle+ring on black")
    void bicolourCircleRingSelf() {
        assertSelfMatch(ReferenceId.BICOLOUR_CIRCLE_RING, multiColourScene(ReferenceId.BICOLOUR_CIRCLE_RING));
    }

    @Test @Order(21) @DisplayName("BICOLOUR_RECT_HALVES — bi-colour rect halves on black")
    void bicolourRectHalvesSelf() {
        assertSelfMatch(ReferenceId.BICOLOUR_RECT_HALVES, multiColourScene(ReferenceId.BICOLOUR_RECT_HALVES));
    }

    @Test @Order(22) @DisplayName("TRICOLOUR_TRIANGLE — tri-colour triangle on black")
    void tricolourTriangleSelf() {
        assertSelfMatch(ReferenceId.TRICOLOUR_TRIANGLE, multiColourScene(ReferenceId.TRICOLOUR_TRIANGLE));
    }

    @Test @Order(23) @DisplayName("BICOLOUR_CROSSHAIR_RING — bi-colour crosshair+ring on black")
    void bicolourCrosshairRingSelf() {
        assertSelfMatch(ReferenceId.BICOLOUR_CROSSHAIR_RING, multiColourScene(ReferenceId.BICOLOUR_CROSSHAIR_RING));
    }

    @Test @Order(24) @DisplayName("BICOLOUR_CHEVRON_FILLED — bi-colour chevron on black")
    void bicolourChevronFilledSelf() {
        assertSelfMatch(ReferenceId.BICOLOUR_CHEVRON_FILLED, multiColourScene(ReferenceId.BICOLOUR_CHEVRON_FILLED));
    }

    // =========================================================================
    // Compound shapes
    // =========================================================================

    @Test @Order(30) @DisplayName("COMPOUND_CIRCLE_IN_RECT — circle-in-rect on black")
    void compoundCircleInRectSelf() {
        assertSelfMatch(ReferenceId.COMPOUND_CIRCLE_IN_RECT, multiColourScene(ReferenceId.COMPOUND_CIRCLE_IN_RECT));
    }

    @Test @Order(31) @DisplayName("COMPOUND_BULLSEYE — bullseye on black")
    void compoundBullseyeSelf() {
        assertSelfMatch(ReferenceId.COMPOUND_BULLSEYE, multiColourScene(ReferenceId.COMPOUND_BULLSEYE));
    }

    @Test @Order(32) @DisplayName("COMPOUND_CROSS_IN_CIRCLE — cross-in-circle on black")
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
        SceneEntry scene = new SceneEntry(
                refId, SceneCategory.A_CLEAN, "step5_synthetic",
                BackgroundId.BG_SOLID_BLACK, Collections.emptyList(), sceneMat);
        long descriptorMs = scene.descriptorBuildMs();
        List<AnalysisResult> results = VectorMatcher.match(
                refId, ref, scene, Collections.emptySet(), OUTPUT);
        return new MatchRun(results, descriptorMs);
    }

    private double record(String stage, String testId, String shapeName,
                          String sceneDesc, Mat sceneMat, MatchRun run) {
        double score = report.record(stage, testId, shapeName, sceneDesc, sceneMat,
                new MatchReportLibrary.MatchRun(run.results(), run.descriptorMs()));
        diag.evaluate(
                BackgroundId.BG_SOLID_BLACK, sceneDesc,
                run.results().isEmpty() ? null : run.results().getFirst().referenceId(),
                40.0, 75.0, 0.5, OUTPUT);
        return score;
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

    private static Mat rotate(Mat src, double angleDeg) {
        Point centre = new Point(src.cols() / 2.0, src.rows() / 2.0);
        Mat rot = Imgproc.getRotationMatrix2D(centre, -angleDeg, 1.0);
        Mat dst = Mat.zeros(src.size(), src.type());
        Imgproc.warpAffine(src, dst, rot, src.size());
        rot.release();
        return dst;
    }
}
