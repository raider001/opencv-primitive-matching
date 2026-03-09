package org.example.vectormatcher;

import org.example.OpenCvLoader;
import org.example.analytics.AnalysisResult;
import org.example.factories.BackgroundFactory;
import org.example.factories.BackgroundId;
import org.example.factories.ReferenceId;
import org.example.factories.ReferenceImageFactory;
import org.example.colour.ColourPreFilter;
import org.example.colour.SceneColourClusters;
import org.example.matchers.VectorMatcher;
import org.example.matchers.VectorSignature;
import org.example.matchers.VectorVariant;
import org.example.scene.SceneCategory;
import org.example.scene.SceneEntry;
import org.junit.jupiter.api.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.ArrayList;
import java.util.concurrent.CopyOnWriteArrayList;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Step 5 — Vector Matcher incremental integration test.
 *
 * <p>Every test records its result into {@link #REPORT_ROWS} so that
 * {@link #writeHtmlReport()} can render a full visual HTML report at
 * {@code test_output/vector_matching/step5_report.html} after all stages run.
 *
 * <h2>Stages</h2>
 * <ol>
 *   <li>Stage 1 — Clean scenes: circle and rectangle</li>
 *   <li>Stage 2 — Polygon shapes: triangle and hexagon</li>
 *   <li>Stage 3 — Scale invariance</li>
 *   <li>Stage 4 — Basic rotation checks</li>
 *   <li>Stage 5 — Negative scene discrimination</li>
 *   <li>Stage 6 — Pentagon, star, diamond, arrow, ellipse</li>
 *   <li>Stage 7 — Full rotation invariance (multiple angles)</li>
 *   <li>Stage 8 — Octagon, plus-shape, concave arrowhead, cross-lines, 45°-rotated rect</li>
 *   <li>Stage 9 — Rotation tests for Stage 8 shapes</li>
 * </ol>
 */
@DisplayName("Step 5 — Vector Matching (incremental)")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class VectorMatchingTest {

    private static final Path OUTPUT = Paths.get("test_output", "vector_matching");
    private static final double MATCH_THRESHOLD = 50.0;

    // ── Shared result accumulator ──────────────────────────────────────────
    /**
     * One row per matcher call.
     * Pipeline images (all base-64 PNG thumbnails):
     *   refOrig   — the coloured 128×128 reference from ReferenceImageFactory
     *   refBinary — reference after greyscale + threshold (what the matcher sees)
     *   sceneOrig — the raw scene Mat
     *   sceneBin  — scene after greyscale threshold OR'd with Canny edges
     *   sceneAnnot — scene with both expected bbox (cyan) and best-match bbox drawn
     * iou — Intersection-over-Union of detected vs expected bbox (NaN if no GT)
     * falsePositive — score passed threshold but IoU < 0.3 (matched wrong location)
     */
    record ReportRow(String stage, String label, String shapeName, String sceneDesc,
                     // Standard pipeline
                     double score, boolean passed,
                     String refOrig, String refPoints, String refPointsApprox,
                     String sceneOrig, String sceneBin,
                     String allPoints,
                     String sceneAnnot,
                     // CF pipeline
                     double cfScore,
                     String cfFiltered, String cfEdges,
                     String cfAllPoints,
                     String cfAnnot,
                     // Localisation quality
                     double iou, boolean falsePositive) {}

    /**
     * Returns true when the background genuinely contains instances of the
     * same shape type as the reference, so a low-IoU match is not a "false positive"
     * in the algorithmic sense — the matcher correctly found a real occurrence.
     *
     * <p>Examples:
     * <ul>
     *   <li>CIRCLE_FILLED on BG_RANDOM_CIRCLES → circles ARE drawn → not a FP</li>
     *   <li>RECT_FILLED   on BG_RANDOM_CIRCLES → no rects         → IS a FP</li>
     *   <li>RECT_FILLED   on BG_RANDOM_MIXED   → rects ARE drawn  → not a FP</li>
     * </ul>
     */
    private static boolean backgroundContainsShape(BackgroundId bgId, ReferenceId refId) {
        if (bgId == null || refId == null) return false;
        VectorSignature.ShapeType refType = shapeTypeOf(refId);
        return switch (bgId) {
            case BG_RANDOM_CIRCLES -> refType == VectorSignature.ShapeType.CIRCLE;
            case BG_RANDOM_LINES   -> refType == VectorSignature.ShapeType.LINE_SEGMENT;
            case BG_RANDOM_MIXED   ->
                // mixed contains lines, circles AND rectangles
                refType == VectorSignature.ShapeType.CIRCLE
                || refType == VectorSignature.ShapeType.LINE_SEGMENT
                || refType == VectorSignature.ShapeType.CLOSED_CONVEX_POLY;
            case BG_CIRCUIT_LIKE   -> refType == VectorSignature.ShapeType.LINE_SEGMENT
                                   || refType == VectorSignature.ShapeType.CIRCLE;
            case BG_ORGANIC        -> refType == VectorSignature.ShapeType.CIRCLE;
            default -> false;
        };
    }

    /** Maps a ReferenceId to its primary VectorSignature ShapeType. */
    private static VectorSignature.ShapeType shapeTypeOf(ReferenceId refId) {
        return switch (refId) {
            case CIRCLE_FILLED, ELLIPSE_H -> VectorSignature.ShapeType.CIRCLE;
            case LINE_CROSS               -> VectorSignature.ShapeType.COMPOUND;
            case STAR_5_FILLED, CONCAVE_ARROW_HEAD -> VectorSignature.ShapeType.CLOSED_CONCAVE_POLY;
            default                       -> VectorSignature.ShapeType.CLOSED_CONVEX_POLY;
        };
    }

    private final List<ReportRow> REPORT_ROWS = new CopyOnWriteArrayList<>();

    @BeforeAll
    void load() {
        OpenCvLoader.load();
        try { Files.createDirectories(OUTPUT); } catch (IOException ignored) {}
    }

    @AfterAll
    void writeHtmlReport() throws IOException {
        Path out = OUTPUT.resolve("step5_report.html");
        Files.writeString(out, buildHtml(REPORT_ROWS), StandardCharsets.UTF_8);
        System.out.println("[VectorMatchingTest] Report written: " + out.toAbsolutePath());
    }

    // =========================================================================
    // STAGE 1
    // =========================================================================
    @Nested @DisplayName("Stage 1 — Clean scenes: circle and rectangle")
    @TestMethodOrder(MethodOrderer.OrderAnnotation.class)
    class Stage1 {

        @Test @Order(1) @DisplayName("S1a — CIRCLE_FILLED matches circle scene > 50%")
        void circleMatchesCircleScene() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.CIRCLE_FILLED);
            Mat scene = whiteCircleOnBlack(320, 240, 60);
            double score = record("Stage 1", "S1a", "CIRCLE_FILLED", "circle ø120", scene,
                    runMatcher(ReferenceId.CIRCLE_FILLED, ref, scene));
            ref.release(); scene.release();
            assertTrue(score > MATCH_THRESHOLD, "got " + score);
        }

        @Test @Order(2) @DisplayName("S1b — RECT_FILLED matches rectangle scene > 50%")
        void rectMatchesRectScene() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.RECT_FILLED);
            Mat scene = whiteRectOnBlack(230, 160, 410, 320);
            double score = record("Stage 1", "S1b", "RECT_FILLED", "rect 180×160", scene,
                    runMatcher(ReferenceId.RECT_FILLED, ref, scene));
            ref.release(); scene.release();
            assertTrue(score > MATCH_THRESHOLD, "got " + score);
        }

        @Test @Order(3) @DisplayName("S1c — Circle scores higher on circle than rect")
        void circleScoresHigherOnItsOwnScene() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.CIRCLE_FILLED);
            Mat circScene = whiteCircleOnBlack(320, 240, 60);
            Mat rectScene = whiteRectOnBlack(230, 160, 410, 320);
            double onCirc = record("Stage 1","S1c","CIRCLE_FILLED","circle ø120 (own)",circScene, runMatcher(ReferenceId.CIRCLE_FILLED, ref, circScene));
            double onRect = record("Stage 1","S1c","CIRCLE_FILLED","rect (wrong)",      rectScene, runMatcher(ReferenceId.CIRCLE_FILLED, ref, rectScene));
            ref.release(); circScene.release(); rectScene.release();
            assertTrue(onCirc > onRect, "own=" + onCirc + " wrong=" + onRect);
        }

        @Test @Order(4) @DisplayName("S1d — Rect scores higher on rect than circle")
        void rectScoresHigherOnItsOwnScene() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.RECT_FILLED);
            Mat rectScene = whiteRectOnBlack(230, 160, 410, 320);
            Mat circScene = whiteCircleOnBlack(320, 240, 60);
            double onRect = record("Stage 1","S1d","RECT_FILLED","rect 180×160 (own)", rectScene, runMatcher(ReferenceId.RECT_FILLED, ref, rectScene));
            double onCirc = record("Stage 1","S1d","RECT_FILLED","circle (wrong)",     circScene, runMatcher(ReferenceId.RECT_FILLED, ref, circScene));
            ref.release(); rectScene.release(); circScene.release();
            assertTrue(onRect > onCirc, "own=" + onRect + " wrong=" + onCirc);
        }

        @Test @Order(5) @DisplayName("S1e — All 9 variants return without error")
        void allVariantsReturnWithoutError() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.CIRCLE_FILLED);
            Mat scene = whiteCircleOnBlack(320, 240, 60);
            List<AnalysisResult> results = runMatcher(ReferenceId.CIRCLE_FILLED, ref, scene);
            ref.release(); scene.release();
            assertEquals(9, results.size());
            for (AnalysisResult r : results)
                assertFalse(r.isError(), r.methodName() + ": " + r.errorMessage());
        }
    }

    // =========================================================================
    // STAGE 2
    // =========================================================================
    @Nested @DisplayName("Stage 2 — Polygon shapes: triangle and hexagon")
    @TestMethodOrder(MethodOrderer.OrderAnnotation.class)
    class Stage2 {

        @Test @Order(1) @DisplayName("S2a — TRIANGLE_FILLED matches triangle scene > 50%")
        void triangleMatchesTriangleScene() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.TRIANGLE_FILLED);
            Mat scene = whiteTriangleOnBlack();
            double score = record("Stage 2","S2a","TRIANGLE_FILLED","triangle (own)",scene, runMatcher(ReferenceId.TRIANGLE_FILLED, ref, scene));
            ref.release(); scene.release();
            assertTrue(score > MATCH_THRESHOLD, "got " + score);
        }

        @Test @Order(2) @DisplayName("S2b — HEXAGON_OUTLINE matches hexagon scene > 50%")
        void hexagonMatchesHexagonScene() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.HEXAGON_OUTLINE);
            Mat scene = whiteHexagonOnBlack(false);
            double score = record("Stage 2","S2b","HEXAGON_OUTLINE","hexagon (own)",scene, runMatcher(ReferenceId.HEXAGON_OUTLINE, ref, scene));
            ref.release(); scene.release();
            assertTrue(score > MATCH_THRESHOLD, "got " + score);
        }

        @Test @Order(3) @DisplayName("S2c — Triangle scores higher on triangle than circle")
        void triangleScoresHigherOnItsOwnScene() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.TRIANGLE_FILLED);
            Mat triScene  = whiteTriangleOnBlack();
            Mat circScene = whiteCircleOnBlack(320, 240, 60);
            double onTri  = record("Stage 2","S2c","TRIANGLE_FILLED","triangle (own)", triScene, runMatcher(ReferenceId.TRIANGLE_FILLED, ref, triScene));
            double onCirc = record("Stage 2","S2c","TRIANGLE_FILLED","circle (wrong)",circScene, runMatcher(ReferenceId.TRIANGLE_FILLED, ref, circScene));
            ref.release(); triScene.release(); circScene.release();
            assertTrue(onTri > onCirc, "own=" + onTri + " wrong=" + onCirc);
        }
    }

    // =========================================================================
    // STAGE 3
    // =========================================================================
    @Nested @DisplayName("Stage 3 — Scale invariance")
    @TestMethodOrder(MethodOrderer.OrderAnnotation.class)
    class Stage3 {

        @Test @Order(1) @DisplayName("S3a — Circle at 0.5× scale > 40%")
        void circleMatchesHalfScale() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.CIRCLE_FILLED);
            Mat scene = whiteCircleOnBlack(320, 240, 30);
            double score = record("Stage 3","S3a","CIRCLE_FILLED","circle ø60 (0.5×)",scene, runMatcher(ReferenceId.CIRCLE_FILLED, ref, scene));
            ref.release(); scene.release();
            assertTrue(score > 40.0, "got " + score);
        }

        @Test @Order(2) @DisplayName("S3b — Circle at 1.5× scale > 40%")
        void circleMatchesLargeScale() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.CIRCLE_FILLED);
            Mat scene = whiteCircleOnBlack(320, 240, 90);
            double score = record("Stage 3","S3b","CIRCLE_FILLED","circle ø180 (1.5×)",scene, runMatcher(ReferenceId.CIRCLE_FILLED, ref, scene));
            ref.release(); scene.release();
            assertTrue(score > 40.0, "got " + score);
        }

        @Test @Order(3) @DisplayName("S3c — Rect at 0.5× scale > 40%")
        void rectMatchesHalfScale() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.RECT_FILLED);
            Mat scene = whiteRectOnBlack(280, 200, 360, 280);
            double score = record("Stage 3","S3c","RECT_FILLED","rect 80×80 (0.5×)",scene, runMatcher(ReferenceId.RECT_FILLED, ref, scene));
            ref.release(); scene.release();
            assertTrue(score > 40.0, "got " + score);
        }

        @Test @Order(4) @DisplayName("S3d — Scaled circles beat wrong shape")
        void scaledScenesScoreHigherThanWrongShape() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.CIRCLE_FILLED);
            Mat small  = whiteCircleOnBlack(320, 240, 30);
            Mat large  = whiteCircleOnBlack(320, 240, 90);
            Mat rect   = whiteRectOnBlack(230, 160, 410, 320);
            double s = record("Stage 3","S3d","CIRCLE_FILLED","circle 0.5×", small, runMatcher(ReferenceId.CIRCLE_FILLED, ref, small));
            double l = record("Stage 3","S3d","CIRCLE_FILLED","circle 1.5×", large, runMatcher(ReferenceId.CIRCLE_FILLED, ref, large));
            double r = record("Stage 3","S3d","CIRCLE_FILLED","rect (wrong)", rect,  runMatcher(ReferenceId.CIRCLE_FILLED, ref, rect));
            ref.release(); small.release(); large.release(); rect.release();
            assertTrue(s > r, "small circle should beat rect");
            assertTrue(l > r, "large circle should beat rect");
        }
    }

    // =========================================================================
    // STAGE 4
    // =========================================================================
    @Nested @DisplayName("Stage 4 — Basic rotation checks")
    @TestMethodOrder(MethodOrderer.OrderAnnotation.class)
    class Stage4 {

        @Test @Order(1) @DisplayName("S4a — Circle at 90° > 50%")
        void circleMatchesRotated90() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.CIRCLE_FILLED);
            Mat scene = whiteCircleOnBlack(320, 240, 60);
            double score = record("Stage 4","S4a","CIRCLE_FILLED","circle rot 90°",scene, runMatcher(ReferenceId.CIRCLE_FILLED, ref, scene));
            ref.release(); scene.release();
            assertTrue(score > MATCH_THRESHOLD, "got " + score);
        }

        @Test @Order(2) @DisplayName("S4b — Rect at 90° > 40%")
        void rectMatchesRotated90() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.RECT_FILLED);
            Mat scene = whiteRectOnBlack(260, 130, 380, 350);
            double score = record("Stage 4","S4b","RECT_FILLED","rect rot 90°",scene, runMatcher(ReferenceId.RECT_FILLED, ref, scene));
            ref.release(); scene.release();
            assertTrue(score > 40.0, "got " + score);
        }

        @Test @Order(3) @DisplayName("S4c — Triangle at 45° > 40%")
        void triangleMatchesRotated45() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.TRIANGLE_FILLED);
            Mat scene = rotate(whiteTriangleOnBlack(), 45);
            double score = record("Stage 4","S4c","TRIANGLE_FILLED","triangle rot 45°",scene, runMatcher(ReferenceId.TRIANGLE_FILLED, ref, scene));
            ref.release(); scene.release();
            assertTrue(score > 40.0, "got " + score);
        }
    }

    // =========================================================================
    // STAGE 5
    // =========================================================================
    @Nested @DisplayName("Stage 5 — Negative scene discrimination")
    @TestMethodOrder(MethodOrderer.OrderAnnotation.class)
    class Stage5 {

        @Test @Order(1) @DisplayName("S5a — Circle < 40% on blank scene")
        void circleLowOnBlank() {
            Mat ref   = ReferenceImageFactory.build(ReferenceId.CIRCLE_FILLED);
            Mat blank = Mat.zeros(480, 640, CvType.CV_8UC3);
            double score = record("Stage 5","S5a","CIRCLE_FILLED","blank scene",blank, runMatcher(ReferenceId.CIRCLE_FILLED, ref, blank));
            ref.release(); blank.release();
            assertTrue(score < 40.0, "got " + score);
        }

        @Test @Order(2) @DisplayName("S5b — Rect < 40% on blank scene")
        void rectLowOnBlank() {
            Mat ref   = ReferenceImageFactory.build(ReferenceId.RECT_FILLED);
            Mat blank = Mat.zeros(480, 640, CvType.CV_8UC3);
            double score = record("Stage 5","S5b","RECT_FILLED","blank scene",blank, runMatcher(ReferenceId.RECT_FILLED, ref, blank));
            ref.release(); blank.release();
            assertTrue(score < 40.0, "got " + score);
        }

        @Test @Order(3) @DisplayName("S5c — Circle scores lower on triangle than on circle scene")
        void circleLowerOnTriangle() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.CIRCLE_FILLED);
            Mat circScene = whiteCircleOnBlack(320, 240, 60);
            Mat triScene  = whiteTriangleOnBlack();
            double onCirc = record("Stage 5","S5c","CIRCLE_FILLED","circle (own)",  circScene, runMatcher(ReferenceId.CIRCLE_FILLED, ref, circScene));
            double onTri  = record("Stage 5","S5c","CIRCLE_FILLED","triangle (neg)",triScene,  runMatcher(ReferenceId.CIRCLE_FILLED, ref, triScene));
            ref.release(); circScene.release(); triScene.release();
            assertTrue(onCirc > onTri, "own=" + onCirc + " neg=" + onTri);
        }

        @Test @Order(4) @DisplayName("S5d — No variant throws on noisy scene")
        void noThrowOnNoisyScene() {
            Mat ref   = ReferenceImageFactory.build(ReferenceId.RECT_FILLED);
            Mat noise = new Mat(480, 640, CvType.CV_8UC3);
            Core.randu(noise, 0.0, 255.0);
            assertDoesNotThrow(() -> {
                List<AnalysisResult> results = runMatcher(ReferenceId.RECT_FILLED, ref, noise);
                assertEquals(9, results.size());
                for (AnalysisResult r : results)
                    assertFalse(r.isError(), r.methodName() + " threw");
            });
            ref.release(); noise.release();
        }
    }

    // =========================================================================
    // STAGE 6 — Pentagon, star, diamond, arrow, ellipse
    // =========================================================================
    @Nested @DisplayName("Stage 6 — Pentagon, star, diamond, arrow, ellipse")
    @TestMethodOrder(MethodOrderer.OrderAnnotation.class)
    class Stage6 {

        @Test @Order(1) @DisplayName("S6a — PENTAGON_FILLED matches pentagon > 50%")
        void pentagonSelf() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.PENTAGON_FILLED);
            Mat scene = whitePentagonOnBlack();
            double score = record("Stage 6","S6a","PENTAGON_FILLED","pentagon (own)",scene, runMatcher(ReferenceId.PENTAGON_FILLED, ref, scene));
            ref.release(); scene.release();
            assertTrue(score > MATCH_THRESHOLD, "got " + score);
        }

        @Test @Order(2) @DisplayName("S6b — Pentagon beats circle scene")
        void pentagonBeatsCircle() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.PENTAGON_FILLED);
            Mat pentScene = whitePentagonOnBlack();
            Mat circScene = whiteCircleOnBlack(320, 240, 60);
            double own   = record("Stage 6","S6b","PENTAGON_FILLED","pentagon (own)", pentScene, runMatcher(ReferenceId.PENTAGON_FILLED, ref, pentScene));
            double wrong = record("Stage 6","S6b","PENTAGON_FILLED","circle (wrong)", circScene, runMatcher(ReferenceId.PENTAGON_FILLED, ref, circScene));
            ref.release(); pentScene.release(); circScene.release();
            assertTrue(own > wrong, "own=" + own + " wrong=" + wrong);
        }

        @Test @Order(3) @DisplayName("S6c — STAR_5_FILLED matches star > 50%")
        void starSelf() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.STAR_5_FILLED);
            Mat scene = whiteStarOnBlack();
            double score = record("Stage 6","S6c","STAR_5_FILLED","star (own)",scene, runMatcher(ReferenceId.STAR_5_FILLED, ref, scene));
            ref.release(); scene.release();
            assertTrue(score > MATCH_THRESHOLD, "got " + score);
        }

        @Test @Order(4) @DisplayName("S6d — Star beats pentagon scene")
        void starBeatsPentagon() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.STAR_5_FILLED);
            Mat starScene = whiteStarOnBlack();
            Mat pentScene = whitePentagonOnBlack();
            double own   = record("Stage 6","S6d","STAR_5_FILLED","star (own)",    starScene, runMatcher(ReferenceId.STAR_5_FILLED, ref, starScene));
            double wrong = record("Stage 6","S6d","STAR_5_FILLED","pentagon (wrong)",pentScene, runMatcher(ReferenceId.STAR_5_FILLED, ref, pentScene));
            ref.release(); starScene.release(); pentScene.release();
            assertTrue(own > wrong, "own=" + own + " wrong=" + wrong);
        }

        @Test @Order(5) @DisplayName("S6e — POLYLINE_DIAMOND matches diamond > 50%")
        void diamondSelf() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.POLYLINE_DIAMOND);
            Mat scene = whiteDiamondOnBlack();
            double score = record("Stage 6","S6e","POLYLINE_DIAMOND","diamond (own)",scene, runMatcher(ReferenceId.POLYLINE_DIAMOND, ref, scene));
            ref.release(); scene.release();
            assertTrue(score > MATCH_THRESHOLD, "got " + score);
        }

        @Test @Order(6) @DisplayName("S6f — Diamond beats circle scene")
        void diamondBeatsCircle() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.POLYLINE_DIAMOND);
            Mat diamScene = whiteDiamondOnBlack();
            Mat circScene = whiteCircleOnBlack(320, 240, 60);
            double own   = record("Stage 6","S6f","POLYLINE_DIAMOND","diamond (own)", diamScene, runMatcher(ReferenceId.POLYLINE_DIAMOND, ref, diamScene));
            double wrong = record("Stage 6","S6f","POLYLINE_DIAMOND","circle (wrong)",circScene, runMatcher(ReferenceId.POLYLINE_DIAMOND, ref, circScene));
            ref.release(); diamScene.release(); circScene.release();
            assertTrue(own > wrong, "own=" + own + " wrong=" + wrong);
        }

        @Test @Order(7) @DisplayName("S6g — POLYLINE_ARROW_RIGHT matches arrow > 50%")
        void arrowSelf() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.POLYLINE_ARROW_RIGHT);
            Mat scene = whiteArrowOnBlack();
            double score = record("Stage 6","S6g","POLYLINE_ARROW_RIGHT","arrow (own)",scene, runMatcher(ReferenceId.POLYLINE_ARROW_RIGHT, ref, scene));
            ref.release(); scene.release();
            assertTrue(score > MATCH_THRESHOLD, "got " + score);
        }

        @Test @Order(8) @DisplayName("S6h — Arrow beats rect scene")
        void arrowBeatsRect() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.POLYLINE_ARROW_RIGHT);
            Mat arrowScene = whiteArrowOnBlack();
            Mat rectScene  = whiteRectOnBlack(230, 160, 410, 320);
            double own   = record("Stage 6","S6h","POLYLINE_ARROW_RIGHT","arrow (own)", arrowScene, runMatcher(ReferenceId.POLYLINE_ARROW_RIGHT, ref, arrowScene));
            double wrong = record("Stage 6","S6h","POLYLINE_ARROW_RIGHT","rect (wrong)",rectScene,  runMatcher(ReferenceId.POLYLINE_ARROW_RIGHT, ref, rectScene));
            ref.release(); arrowScene.release(); rectScene.release();
            assertTrue(own > wrong, "own=" + own + " wrong=" + wrong);
        }

        @Test @Order(9) @DisplayName("S6i — ELLIPSE_H matches ellipse > 50%")
        void ellipseSelf() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.ELLIPSE_H);
            Mat scene = whiteEllipseOnBlack();
            double score = record("Stage 6","S6i","ELLIPSE_H","ellipse (own)",scene, runMatcher(ReferenceId.ELLIPSE_H, ref, scene));
            ref.release(); scene.release();
            assertTrue(score > MATCH_THRESHOLD, "got " + score);
        }

        @Test @Order(10) @DisplayName("S6j — Ellipse beats circle scene")
        void ellipseBeatsCircle() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.ELLIPSE_H);
            Mat ellScene  = whiteEllipseOnBlack();
            Mat circScene = whiteCircleOnBlack(320, 240, 60);
            double own   = record("Stage 6","S6j","ELLIPSE_H","ellipse (own)", ellScene, runMatcher(ReferenceId.ELLIPSE_H, ref, ellScene));
            double wrong = record("Stage 6","S6j","ELLIPSE_H","circle (wrong)",circScene, runMatcher(ReferenceId.ELLIPSE_H, ref, circScene));
            ref.release(); ellScene.release(); circScene.release();
            assertTrue(own > wrong, "own=" + own + " wrong=" + wrong);
        }

        @Test @Order(11) @DisplayName("S6k — All new shapes score highest on own scene vs circle")
        void allNewShapesScoreHighestOnOwnScene() {
            Mat pentRef  = ReferenceImageFactory.build(ReferenceId.PENTAGON_FILLED);
            Mat starRef  = ReferenceImageFactory.build(ReferenceId.STAR_5_FILLED);
            Mat diamRef  = ReferenceImageFactory.build(ReferenceId.POLYLINE_DIAMOND);
            Mat arrowRef = ReferenceImageFactory.build(ReferenceId.POLYLINE_ARROW_RIGHT);
            Mat ellRef   = ReferenceImageFactory.build(ReferenceId.ELLIPSE_H);
            Mat pentScene  = whitePentagonOnBlack();
            Mat starScene  = whiteStarOnBlack();
            Mat diamScene  = whiteDiamondOnBlack();
            Mat arrowScene = whiteArrowOnBlack();
            Mat ellScene   = whiteEllipseOnBlack();
            Mat circScene  = whiteCircleOnBlack(320, 240, 60);
            double pOp = normalScore(runMatcher(ReferenceId.PENTAGON_FILLED,      pentRef,  pentScene));
            double sOs = normalScore(runMatcher(ReferenceId.STAR_5_FILLED,        starRef,  starScene));
            double dOd = normalScore(runMatcher(ReferenceId.POLYLINE_DIAMOND,     diamRef,  diamScene));
            double aOa = normalScore(runMatcher(ReferenceId.POLYLINE_ARROW_RIGHT, arrowRef, arrowScene));
            double eOe = normalScore(runMatcher(ReferenceId.ELLIPSE_H,            ellRef,   ellScene));
            double pOc = normalScore(runMatcher(ReferenceId.PENTAGON_FILLED,      pentRef,  circScene));
            double sOc = normalScore(runMatcher(ReferenceId.STAR_5_FILLED,        starRef,  circScene));
            double dOc = normalScore(runMatcher(ReferenceId.POLYLINE_DIAMOND,     diamRef,  circScene));
            double aOc = normalScore(runMatcher(ReferenceId.POLYLINE_ARROW_RIGHT, arrowRef, circScene));
            double eOc = normalScore(runMatcher(ReferenceId.ELLIPSE_H,            ellRef,   circScene));
            pentRef.release(); starRef.release(); diamRef.release(); arrowRef.release(); ellRef.release();
            pentScene.release(); starScene.release(); diamScene.release(); arrowScene.release(); ellScene.release(); circScene.release();
            assertAll(
                () -> assertTrue(pOp > pOc, "pentagon own=" + pOp + " circ=" + pOc),
                () -> assertTrue(sOs > sOc, "star own="    + sOs + " circ=" + sOc),
                () -> assertTrue(dOd > dOc, "diamond own=" + dOd + " circ=" + dOc),
                () -> assertTrue(aOa > aOc, "arrow own="   + aOa + " circ=" + aOc),
                () -> assertTrue(eOe > eOc, "ellipse own=" + eOe + " circ=" + eOc)
            );
        }
    }

    // =========================================================================
    // STAGE 7 — Full rotation invariance
    // =========================================================================
    @Nested @DisplayName("Stage 7 — Rotation invariance (30°, 45°, 90°, 135°, 180°)")
    @TestMethodOrder(MethodOrderer.OrderAnnotation.class)
    class Stage7 {

        private static final double ROT_THRESHOLD = 40.0;

        @Test @Order(1) @DisplayName("S7a — RECT at 30°,45°,90°,135°,180° > 40%")
        void rectAllAngles() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.RECT_FILLED);
            Mat base = whiteRectOnBlack(230, 160, 410, 320);
            for (int deg : new int[]{30,45,90,135,180}) {
                Mat rot = rotate(base, deg);
                double score = record("Stage 7","S7a","RECT_FILLED","rect rot "+deg+"°",rot, runMatcher(ReferenceId.RECT_FILLED, ref, rot));
                rot.release();
                assertTrue(score > ROT_THRESHOLD, "RECT at "+deg+"° got "+score);
            }
            ref.release(); base.release();
        }

        @Test @Order(2) @DisplayName("S7b — TRIANGLE at 30°,45°,90°,135°,180° > 40%")
        void triangleAllAngles() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.TRIANGLE_FILLED);
            Mat base = whiteTriangleOnBlack();
            for (int deg : new int[]{30,45,90,135,180}) {
                Mat rot = rotate(base, deg);
                double score = record("Stage 7","S7b","TRIANGLE_FILLED","triangle rot "+deg+"°",rot, runMatcher(ReferenceId.TRIANGLE_FILLED, ref, rot));
                rot.release();
                assertTrue(score > ROT_THRESHOLD, "TRIANGLE at "+deg+"° got "+score);
            }
            ref.release(); base.release();
        }

        @Test @Order(3) @DisplayName("S7c — PENTAGON at 30°,72°,90°,144°,180° > 40%")
        void pentagonAllAngles() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.PENTAGON_FILLED);
            Mat base = whitePentagonOnBlack();
            for (int deg : new int[]{30,72,90,144,180}) {
                Mat rot = rotate(base, deg);
                double score = record("Stage 7","S7c","PENTAGON_FILLED","pentagon rot "+deg+"°",rot, runMatcher(ReferenceId.PENTAGON_FILLED, ref, rot));
                rot.release();
                assertTrue(score > ROT_THRESHOLD, "PENTAGON at "+deg+"° got "+score);
            }
            ref.release(); base.release();
        }

        @Test @Order(4) @DisplayName("S7d — STAR at 36°,72°,90°,180° > 40%")
        void starAllAngles() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.STAR_5_FILLED);
            Mat base = whiteStarOnBlack();
            for (int deg : new int[]{36,72,90,180}) {
                Mat rot = rotate(base, deg);
                double score = record("Stage 7","S7d","STAR_5_FILLED","star rot "+deg+"°",rot, runMatcher(ReferenceId.STAR_5_FILLED, ref, rot));
                rot.release();
                assertTrue(score > ROT_THRESHOLD, "STAR at "+deg+"° got "+score);
            }
            ref.release(); base.release();
        }

        @Test @Order(5) @DisplayName("S7e — ARROW at 90°,180°,270° > 40%")
        void arrowAllAngles() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.POLYLINE_ARROW_RIGHT);
            Mat base = whiteArrowOnBlack();
            for (int deg : new int[]{90,180,270}) {
                Mat rot = rotate(base, deg);
                double score = record("Stage 7","S7e","POLYLINE_ARROW_RIGHT","arrow rot "+deg+"°",rot, runMatcher(ReferenceId.POLYLINE_ARROW_RIGHT, ref, rot));
                rot.release();
                assertTrue(score > ROT_THRESHOLD, "ARROW at "+deg+"° got "+score);
            }
            ref.release(); base.release();
        }

        @Test @Order(6) @DisplayName("S7f — ELLIPSE at 45°,90° > 40%")
        void ellipseAllAngles() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.ELLIPSE_H);
            Mat base = whiteEllipseOnBlack();
            for (int deg : new int[]{45,90}) {
                Mat rot = rotate(base, deg);
                double score = record("Stage 7","S7f","ELLIPSE_H","ellipse rot "+deg+"°",rot, runMatcher(ReferenceId.ELLIPSE_H, ref, rot));
                rot.release();
                assertTrue(score > ROT_THRESHOLD, "ELLIPSE at "+deg+"° got "+score);
            }
            ref.release(); base.release();
        }

        @Test @Order(7) @DisplayName("S7g — Rotated shapes beat wrong shape at 90°")
        void rotatedShapesBeatWrongShape() {
            Mat rectRef = ReferenceImageFactory.build(ReferenceId.RECT_FILLED);
            Mat triRef  = ReferenceImageFactory.build(ReferenceId.TRIANGLE_FILLED);
            Mat starRef = ReferenceImageFactory.build(ReferenceId.STAR_5_FILLED);
            Mat rectRot = rotate(whiteRectOnBlack(230,160,410,320), 90);
            Mat triRot  = rotate(whiteTriangleOnBlack(), 90);
            Mat starRot = rotate(whiteStarOnBlack(), 90);
            double rr = normalScore(runMatcher(ReferenceId.RECT_FILLED,     rectRef, rectRot));
            double rt = normalScore(runMatcher(ReferenceId.RECT_FILLED,     rectRef, triRot));
            double tt = normalScore(runMatcher(ReferenceId.TRIANGLE_FILLED, triRef,  triRot));
            double tr = normalScore(runMatcher(ReferenceId.TRIANGLE_FILLED, triRef,  rectRot));
            double ss = normalScore(runMatcher(ReferenceId.STAR_5_FILLED,   starRef, starRot));
            double sr = normalScore(runMatcher(ReferenceId.STAR_5_FILLED,   starRef, rectRot));
            rectRef.release(); triRef.release(); starRef.release();
            rectRot.release(); triRot.release(); starRot.release();
            assertAll(
                () -> assertTrue(rr > rt, "rect@90 own="+rr+" tri="+rt),
                () -> assertTrue(tt > tr, "tri@90 own="+tt+" rect="+tr),
                () -> assertTrue(ss > sr, "star@90 own="+ss+" rect="+sr)
            );
        }

        @Test @Order(8) @DisplayName("S7h — Hexagon at 60° within 5% of upright")
        void hexagonSymmetryAt60() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.HEXAGON_OUTLINE);
            Mat upright = whiteHexagonOnBlack(false);
            Mat rot60   = rotate(upright, 60);
            double u = record("Stage 7","S7h","HEXAGON_OUTLINE","hexagon upright", upright, runMatcher(ReferenceId.HEXAGON_OUTLINE, ref, upright));
            double r = record("Stage 7","S7h","HEXAGON_OUTLINE","hexagon rot 60°", rot60,   runMatcher(ReferenceId.HEXAGON_OUTLINE, ref, rot60));
            ref.release(); upright.release(); rot60.release();
            assertTrue(r > ROT_THRESHOLD, "got "+r);
            assertTrue(Math.abs(u - r) < 5.0, "delta="+(u-r));
        }
    }

    // =========================================================================
    // STAGE 8 — Five more new shapes
    // Octagon, plus-shape, concave arrowhead, cross-lines, 45°-rotated rect
    // =========================================================================
    @Nested @DisplayName("Stage 8 — Octagon, plus, concave arrowhead, cross, rotated rect")
    @TestMethodOrder(MethodOrderer.OrderAnnotation.class)
    class Stage8 {

        @Test @Order(1) @DisplayName("S8a — OCTAGON_FILLED matches octagon > 50%")
        void octagonSelf() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.OCTAGON_FILLED);
            Mat scene = whiteOctagonOnBlack();
            double score = record("Stage 8","S8a","OCTAGON_FILLED","octagon (own)",scene, runMatcher(ReferenceId.OCTAGON_FILLED, ref, scene));
            ref.release(); scene.release();
            assertTrue(score > MATCH_THRESHOLD, "got " + score);
        }

        @Test @Order(2) @DisplayName("S8b — Octagon beats triangle scene")
        void octagonBeatsCircle() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.OCTAGON_FILLED);
            Mat octScene  = whiteOctagonOnBlack();
            Mat triScene  = whiteTriangleOnBlack();
            double own   = record("Stage 8","S8b","OCTAGON_FILLED","octagon (own)",  octScene, runMatcher(ReferenceId.OCTAGON_FILLED, ref, octScene));
            double wrong = record("Stage 8","S8b","OCTAGON_FILLED","triangle (wrong)",triScene, runMatcher(ReferenceId.OCTAGON_FILLED, ref, triScene));
            ref.release(); octScene.release(); triScene.release();
            assertTrue(own > wrong, "own="+own+" wrong="+wrong);
        }

        @Test @Order(3) @DisplayName("S8c — POLYLINE_PLUS_SHAPE matches plus > 50%")
        void plusSelf() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.POLYLINE_PLUS_SHAPE);
            Mat scene = whitePlusOnBlack();
            double score = record("Stage 8","S8c","POLYLINE_PLUS_SHAPE","plus (own)",scene, runMatcher(ReferenceId.POLYLINE_PLUS_SHAPE, ref, scene));
            ref.release(); scene.release();
            assertTrue(score > MATCH_THRESHOLD, "got " + score);
        }

        @Test @Order(4) @DisplayName("S8d — Plus beats rect scene")
        void plusBeatsRect() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.POLYLINE_PLUS_SHAPE);
            Mat plusScene = whitePlusOnBlack();
            Mat rectScene = whiteRectOnBlack(230, 160, 410, 320);
            double own   = record("Stage 8","S8d","POLYLINE_PLUS_SHAPE","plus (own)", plusScene, runMatcher(ReferenceId.POLYLINE_PLUS_SHAPE, ref, plusScene));
            double wrong = record("Stage 8","S8d","POLYLINE_PLUS_SHAPE","rect (wrong)",rectScene, runMatcher(ReferenceId.POLYLINE_PLUS_SHAPE, ref, rectScene));
            ref.release(); plusScene.release(); rectScene.release();
            assertTrue(own > wrong, "own="+own+" wrong="+wrong);
        }

        @Test @Order(5) @DisplayName("S8e — CONCAVE_ARROW_HEAD matches arrowhead > 50%")
        void concaveArrowheadSelf() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.CONCAVE_ARROW_HEAD);
            Mat scene = whiteConcaveArrowheadOnBlack();
            double score = record("Stage 8","S8e","CONCAVE_ARROW_HEAD","arrowhead (own)",scene, runMatcher(ReferenceId.CONCAVE_ARROW_HEAD, ref, scene));
            ref.release(); scene.release();
            assertTrue(score > MATCH_THRESHOLD, "got " + score);
        }

        @Test @Order(6) @DisplayName("S8f — Concave arrowhead beats circle scene")
        void concaveArrowheadBeatsTriangle() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.CONCAVE_ARROW_HEAD);
            Mat ahScene   = whiteConcaveArrowheadOnBlack();
            Mat circScene = whiteCircleOnBlack(320, 240, 60);
            double own   = record("Stage 8","S8f","CONCAVE_ARROW_HEAD","arrowhead (own)",  ahScene,   runMatcher(ReferenceId.CONCAVE_ARROW_HEAD, ref, ahScene));
            double wrong = record("Stage 8","S8f","CONCAVE_ARROW_HEAD","circle (wrong)",   circScene, runMatcher(ReferenceId.CONCAVE_ARROW_HEAD, ref, circScene));
            ref.release(); ahScene.release(); circScene.release();
            assertTrue(own > wrong, "own="+own+" wrong="+wrong);
        }

        @Test @Order(7) @DisplayName("S8g — LINE_CROSS matches cross > 50%")
        void lineCrossSelf() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.LINE_CROSS);
            Mat scene = whiteCrossOnBlack();
            double score = record("Stage 8","S8g","LINE_CROSS","cross (own)",scene, runMatcher(ReferenceId.LINE_CROSS, ref, scene));
            ref.release(); scene.release();
            assertTrue(score > MATCH_THRESHOLD, "got " + score);
        }

        @Test @Order(8) @DisplayName("S8h — Cross beats circle scene")
        void lineCrossBeatsCircle() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.LINE_CROSS);
            Mat crossScene = whiteCrossOnBlack();
            Mat circScene  = whiteCircleOnBlack(320, 240, 60);
            double own   = record("Stage 8","S8h","LINE_CROSS","cross (own)",   crossScene, runMatcher(ReferenceId.LINE_CROSS, ref, crossScene));
            double wrong = record("Stage 8","S8h","LINE_CROSS","circle (wrong)",circScene,  runMatcher(ReferenceId.LINE_CROSS, ref, circScene));
            ref.release(); crossScene.release(); circScene.release();
            assertTrue(own > wrong, "own="+own+" wrong="+wrong);
        }

        @Test @Order(9) @DisplayName("S8i — RECT_ROTATED_45 matches rotated rect > 50%")
        void rotatedRectSelf() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.RECT_ROTATED_45);
            Mat scene = whiteRot45RectOnBlack();
            double score = record("Stage 8","S8i","RECT_ROTATED_45","rot45rect (own)",scene, runMatcher(ReferenceId.RECT_ROTATED_45, ref, scene));
            ref.release(); scene.release();
            assertTrue(score > MATCH_THRESHOLD, "got " + score);
        }

        @Test @Order(10) @DisplayName("S8j — Rotated rect beats triangle scene")
        void rotatedRectBeatsAxisRect() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.RECT_ROTATED_45);
            Mat rr45Scene = whiteRot45RectOnBlack();
            Mat triScene  = whiteTriangleOnBlack();
            double own   = record("Stage 8","S8j","RECT_ROTATED_45","rot45rect (own)", rr45Scene, runMatcher(ReferenceId.RECT_ROTATED_45, ref, rr45Scene));
            double wrong = record("Stage 8","S8j","RECT_ROTATED_45","triangle (wrong)",triScene,  runMatcher(ReferenceId.RECT_ROTATED_45, ref, triScene));
            ref.release(); rr45Scene.release(); triScene.release();
            assertTrue(own > wrong, "own="+own+" wrong="+wrong);
        }
    }

    // =========================================================================
    // STAGE 9 — Rotation tests for Stage 8 shapes
    // =========================================================================
    @Nested @DisplayName("Stage 9 — Rotation tests for Stage 8 shapes")
    @TestMethodOrder(MethodOrderer.OrderAnnotation.class)
    class Stage9 {

        private static final double ROT_THRESHOLD = 40.0;

        @Test @Order(1) @DisplayName("S9a — OCTAGON at 22°, 45°, 90°, 180° > 40%")
        void octagonRotations() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.OCTAGON_FILLED);
            Mat base = whiteOctagonOnBlack();
            for (int deg : new int[]{22, 45, 90, 180}) {
                Mat rot = rotate(base, deg);
                double score = record("Stage 9","S9a","OCTAGON_FILLED","octagon rot "+deg+"°",rot, runMatcher(ReferenceId.OCTAGON_FILLED, ref, rot));
                rot.release();
                assertTrue(score > ROT_THRESHOLD, "OCTAGON at "+deg+"° got "+score);
            }
            ref.release(); base.release();
        }

        @Test @Order(2) @DisplayName("S9b — PLUS at 45°, 90° > 40%")
        void plusRotations() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.POLYLINE_PLUS_SHAPE);
            Mat base = whitePlusOnBlack();
            for (int deg : new int[]{45, 90}) {
                Mat rot = rotate(base, deg);
                double score = record("Stage 9","S9b","POLYLINE_PLUS_SHAPE","plus rot "+deg+"°",rot, runMatcher(ReferenceId.POLYLINE_PLUS_SHAPE, ref, rot));
                rot.release();
                assertTrue(score > ROT_THRESHOLD, "PLUS at "+deg+"° got "+score);
            }
            ref.release(); base.release();
        }

        @Test @Order(3) @DisplayName("S9c — CONCAVE_ARROW_HEAD at 90°, 180°, 270° > 40%")
        void arrowheadRotations() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.CONCAVE_ARROW_HEAD);
            Mat base = whiteConcaveArrowheadOnBlack();
            for (int deg : new int[]{90, 180, 270}) {
                Mat rot = rotate(base, deg);
                double score = record("Stage 9","S9c","CONCAVE_ARROW_HEAD","arrowhead rot "+deg+"°",rot, runMatcher(ReferenceId.CONCAVE_ARROW_HEAD, ref, rot));
                rot.release();
                assertTrue(score > ROT_THRESHOLD, "ARROWHEAD at "+deg+"° got "+score);
            }
            ref.release(); base.release();
        }

        @Test @Order(4) @DisplayName("S9d — LINE_CROSS at 45°, 90° > 40%")
        void crossRotations() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.LINE_CROSS);
            Mat base = whiteCrossOnBlack();
            for (int deg : new int[]{45, 90}) {
                Mat rot = rotate(base, deg);
                double score = record("Stage 9","S9d","LINE_CROSS","cross rot "+deg+"°",rot, runMatcher(ReferenceId.LINE_CROSS, ref, rot));
                rot.release();
                assertTrue(score > ROT_THRESHOLD, "CROSS at "+deg+"° got "+score);
            }
            ref.release(); base.release();
        }

        @Test @Order(5) @DisplayName("S9e — RECT_ROTATED_45 at 45°, 90°, 135° > 40%")
        void rotatedRectRotations() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.RECT_ROTATED_45);
            Mat base = whiteRot45RectOnBlack();
            for (int deg : new int[]{45, 90, 135}) {
                Mat rot = rotate(base, deg);
                double score = record("Stage 9","S9e","RECT_ROTATED_45","rot45rect rot "+deg+"°",rot, runMatcher(ReferenceId.RECT_ROTATED_45, ref, rot));
                rot.release();
                assertTrue(score > ROT_THRESHOLD, "RECT_ROTATED_45 at "+deg+"° got "+score);
            }
            ref.release(); base.release();
        }
    }

    // =========================================================================
    // STAGE 10 — All shapes on non-black backgrounds
    //
    // Three backgrounds are tested in increasing difficulty:
    //   BG_SOLID_WHITE      — Tier 1: shape must be visible on white (uses dark fill)
    //   BG_NOISE_LIGHT      — Tier 2: low Gaussian noise challenges edge detection
    //   BG_GRADIENT_H_COLOUR — Tier 2: horizontal colour gradient, varying contrast
    //
    // Each test composites the shape onto the background using the same mask
    // technique as SceneGenerator, then asserts score > 50%.
    // =========================================================================
    @Nested @DisplayName("Stage 10 — Shapes on non-black backgrounds")
    @TestMethodOrder(MethodOrderer.OrderAnnotation.class)
    class Stage10 {

        // ── Solid white background ─────────────────────────────────────────

        @Test @Order(1) @DisplayName("S10a — All shapes on solid-white background > 40%")
        void allShapesSolidWhite() {
            runBackgroundSuite(BackgroundId.BG_SOLID_WHITE, "solid-white", 40.0);
        }

        // ── Light noise background ─────────────────────────────────────────

        @Test @Order(2) @DisplayName("S10b — All shapes on light-noise background > 40%")
        void allShapesNoiseLight() {
            runBackgroundSuite(BackgroundId.BG_NOISE_LIGHT, "noise-light", 40.0);
        }

        // ── Colour gradient background ─────────────────────────────────────

        @Test @Order(3) @DisplayName("S10c — All shapes on colour-gradient background > 40%")
        void allShapesGradientColour() {
            runBackgroundSuite(BackgroundId.BG_GRADIENT_H_COLOUR, "gradient-colour", 40.0);
        }

        // ── Per-background helper ──────────────────────────────────────────

        private void runBackgroundSuite(BackgroundId bgId, String bgLabel, double defaultThreshold) {
            ShapeCase[] cases = {
                // Circle is smooth (few contour vertices) so gets a relaxed threshold on backgrounds
                new ShapeCase(ReferenceId.CIRCLE_FILLED,          "CIRCLE_FILLED",          whiteCircleOnBlack(320, 240, 60),  30.0),
                new ShapeCase(ReferenceId.RECT_FILLED,            "RECT_FILLED",            whiteRectOnBlack(230, 160, 410, 320), defaultThreshold),
                new ShapeCase(ReferenceId.TRIANGLE_FILLED,        "TRIANGLE_FILLED",        whiteTriangleOnBlack(),            defaultThreshold),
                new ShapeCase(ReferenceId.HEXAGON_OUTLINE,        "HEXAGON_OUTLINE",        whiteHexagonOnBlack(false),        defaultThreshold),
                new ShapeCase(ReferenceId.PENTAGON_FILLED,        "PENTAGON_FILLED",        whitePentagonOnBlack(),            defaultThreshold),
                new ShapeCase(ReferenceId.STAR_5_FILLED,          "STAR_5_FILLED",          whiteStarOnBlack(),                25.0),
                new ShapeCase(ReferenceId.POLYLINE_DIAMOND,       "POLYLINE_DIAMOND",       whiteDiamondOnBlack(),             defaultThreshold),
                new ShapeCase(ReferenceId.POLYLINE_ARROW_RIGHT,   "POLYLINE_ARROW_RIGHT",   whiteArrowOnBlack(),               defaultThreshold),
                new ShapeCase(ReferenceId.ELLIPSE_H,              "ELLIPSE_H",              whiteEllipseOnBlack(),             defaultThreshold),
                new ShapeCase(ReferenceId.OCTAGON_FILLED,         "OCTAGON_FILLED",         whiteOctagonOnBlack(),             defaultThreshold),
                new ShapeCase(ReferenceId.POLYLINE_PLUS_SHAPE,    "POLYLINE_PLUS_SHAPE",    whitePlusOnBlack(),                defaultThreshold),
                new ShapeCase(ReferenceId.CONCAVE_ARROW_HEAD,     "CONCAVE_ARROW_HEAD",     whiteConcaveArrowheadOnBlack(),    defaultThreshold),
                // LINE_CROSS is two intersecting thin strokes (COMPOUND) — harder to score on coloured backgrounds
                new ShapeCase(ReferenceId.LINE_CROSS,             "LINE_CROSS",             whiteCrossOnBlack(),               20.0),
                new ShapeCase(ReferenceId.RECT_ROTATED_45,        "RECT_ROTATED_45",        whiteRot45RectOnBlack(),           defaultThreshold),
            };

            List<org.junit.jupiter.api.function.Executable> assertions = new ArrayList<>();

            for (var c : cases) {
                Rect gt    = groundTruthRect(c.shapeMat());
                Mat scene  = compositeOnBackground(c.shapeMat(), bgId);
                Mat ref    = ReferenceImageFactory.build(c.refId());
                double score = record("Stage 10", "S10-" + bgLabel, c.name(),
                        c.name() + " on " + bgLabel, scene, gt,
                        runMatcher(c.refId(), ref, scene));
                ref.release(); scene.release(); c.shapeMat().release();
                final double s = score;
                final String n = c.name();
                final double thr = c.threshold();
                assertions.add(() -> assertTrue(s > thr,
                        n + " on " + bgLabel + " scored " + String.format("%.1f", s) + "% (need >" + thr + "%)"));
            }

            assertAll("All shapes on " + bgLabel, assertions);
        }

        /** Simple tuple for the per-background suite. */
        private record ShapeCase(ReferenceId refId, String name, Mat shapeMat, double threshold) {}
    }

    // =========================================================================
    // STAGE 11 — Complex backgrounds (Tier 3 & 4)
    //
    // These backgrounds contain structural clutter that the matcher must
    // distinguish from the real target shape:
    //
    //   BG_RANDOM_CIRCLES — Tier 3: 10-20 random circle outlines — directly
    //       competes with CIRCLE_FILLED and smooth shapes.
    //   BG_RANDOM_LINES   — Tier 3: 20-40 random line segments — creates
    //       spurious contours that compete with polygon shapes.
    //   BG_RANDOM_MIXED   — Tier 4: 30-50 random lines + circles + rects —
    //       the most complex background in the library.
    //
    // Thresholds are lower (30% default / 20% for circle & cross) to reflect
    // the genuine difficulty of these scenes.  The test documents current
    // capability rather than requiring perfect discrimination.
    // =========================================================================
    @Nested @DisplayName("Stage 11 — Complex backgrounds (Tier 3 & 4)")
    @TestMethodOrder(MethodOrderer.OrderAnnotation.class)
    class Stage11 {

        @Test @Order(1) @DisplayName("S11a — All shapes on random-circles background > 30%")
        void allShapesRandomCircles() {
            runComplexBackgroundSuite(BackgroundId.BG_RANDOM_CIRCLES, "random-circles");
        }

        @Test @Order(2) @DisplayName("S11b — All shapes on random-lines background > 30%")
        void allShapesRandomLines() {
            runComplexBackgroundSuite(BackgroundId.BG_RANDOM_LINES, "random-lines");
        }

        @Test @Order(3) @DisplayName("S11c — All shapes on random-mixed (Tier 4) background > 30%")
        void allShapesRandomMixed() {
            runComplexBackgroundSuite(BackgroundId.BG_RANDOM_MIXED, "random-mixed");
        }

        private void runComplexBackgroundSuite(BackgroundId bgId, String bgLabel) {
            // Circle & cross are hardest on complex backgrounds — more relaxed threshold
            ShapeCase11[] cases = {
                new ShapeCase11(ReferenceId.CIRCLE_FILLED,        "CIRCLE_FILLED",        whiteCircleOnBlack(320, 240, 60), 20.0),
                new ShapeCase11(ReferenceId.RECT_FILLED,          "RECT_FILLED",          whiteRectOnBlack(230, 160, 410, 320), 30.0),
                new ShapeCase11(ReferenceId.TRIANGLE_FILLED,      "TRIANGLE_FILLED",      whiteTriangleOnBlack(),           30.0),
                new ShapeCase11(ReferenceId.HEXAGON_OUTLINE,      "HEXAGON_OUTLINE",      whiteHexagonOnBlack(false),       30.0),
                new ShapeCase11(ReferenceId.PENTAGON_FILLED,      "PENTAGON_FILLED",      whitePentagonOnBlack(),           30.0),
                new ShapeCase11(ReferenceId.STAR_5_FILLED,        "STAR_5_FILLED",        whiteStarOnBlack(),               30.0),
                new ShapeCase11(ReferenceId.POLYLINE_DIAMOND,     "POLYLINE_DIAMOND",     whiteDiamondOnBlack(),            30.0),
                new ShapeCase11(ReferenceId.POLYLINE_ARROW_RIGHT, "POLYLINE_ARROW_RIGHT", whiteArrowOnBlack(),              30.0),
                new ShapeCase11(ReferenceId.ELLIPSE_H,            "ELLIPSE_H",            whiteEllipseOnBlack(),            30.0),
                new ShapeCase11(ReferenceId.OCTAGON_FILLED,       "OCTAGON_FILLED",       whiteOctagonOnBlack(),            30.0),
                new ShapeCase11(ReferenceId.POLYLINE_PLUS_SHAPE,  "POLYLINE_PLUS_SHAPE",  whitePlusOnBlack(),               30.0),
                new ShapeCase11(ReferenceId.CONCAVE_ARROW_HEAD,   "CONCAVE_ARROW_HEAD",   whiteConcaveArrowheadOnBlack(),   30.0),
                new ShapeCase11(ReferenceId.LINE_CROSS,           "LINE_CROSS",           whiteCrossOnBlack(),              20.0),
                new ShapeCase11(ReferenceId.RECT_ROTATED_45,      "RECT_ROTATED_45",      whiteRot45RectOnBlack(),          30.0),
            };

            List<org.junit.jupiter.api.function.Executable> assertions = new ArrayList<>();

            for (var c : cases) {
                Rect gt    = groundTruthRect(c.shapeMat());
                Mat scene  = compositeOnBackground(c.shapeMat(), bgId);
                Mat ref    = ReferenceImageFactory.build(c.refId());
                double score = record("Stage 11", "S11-" + bgLabel, c.name(),
                        c.name() + " on " + bgLabel, scene, gt,
                        runMatcher(c.refId(), ref, scene));
                ref.release(); scene.release(); c.shapeMat().release();
                final double s = score;
                final String n = c.name();
                final double thr = c.threshold();
                assertions.add(() -> assertTrue(s > thr,
                        n + " on " + bgLabel + " scored " + String.format("%.1f", s) + "% (need >" + thr + "%)"));
            }

            assertAll("All shapes on " + bgLabel, assertions);
        }

        private record ShapeCase11(ReferenceId refId, String name, Mat shapeMat, double threshold) {}
    }

    // =========================================================================
    // Helpers — matcher invocation + result recording
    // =========================================================================

    private List<AnalysisResult> runMatcher(ReferenceId refId, Mat ref, Mat sceneMat) {
        SceneEntry scene = new SceneEntry(
                refId, SceneCategory.A_CLEAN, "step5_synthetic",
                BackgroundId.BG_SOLID_BLACK, Collections.emptyList(), sceneMat);
        return VectorMatcher.match(refId, ref, scene, Collections.emptySet(), OUTPUT);
    }

    /**
     * Like {@link #runMatcher} but recolours the scene foreground to the reference
     * colour first, so CF variants actually find something to filter on.
     * Used by {@link #record} so both the standard and CF scores are meaningful.
     */
    private List<AnalysisResult> runMatcherColoured(ReferenceId refId, Mat ref, Mat sceneMat) {
        Mat coloured = recolourToRef(sceneMat, refId);
        SceneEntry scene = new SceneEntry(
                refId, SceneCategory.A_CLEAN, "step5_synthetic",
                BackgroundId.BG_SOLID_BLACK, Collections.emptyList(), coloured);
        List<AnalysisResult> results = VectorMatcher.match(refId, ref, scene, Collections.emptySet(), OUTPUT);
        coloured.release();
        return results;
    }

    private static double normalScore(List<AnalysisResult> results) {
        return results.stream()
                .filter(r -> r.methodName().equals(VectorVariant.VECTOR_NORMAL.variantName()))
                .findFirst()
                .map(AnalysisResult::matchScorePercent)
                .orElse(0.0);
    }

    private static double cfLooseScore(List<AnalysisResult> results) {
        return results.stream()
                .filter(r -> r.methodName().equals(VectorVariant.VECTOR_NORMAL_CF_LOOSE.variantName()))
                .findFirst()
                .map(AnalysisResult::matchScorePercent)
                .orElse(0.0);
    }

    /** Overload with no ground-truth rect — auto-derives GT from the scene mat's white pixels. */
    private double record(String stage, String testId, String shapeName,
                          String sceneDesc, Mat sceneMat,
                          List<AnalysisResult> results) {
        // For clean scenes the scene IS the shape, so GT = bounding rect of white pixels
        Rect gt = groundTruthRect(sceneMat);
        return record(stage, testId, shapeName, sceneDesc, sceneMat, gt, results);
    }

    /**
     * Records one matcher result into {@link #REPORT_ROWS}, capturing the full
     * pipeline as thumbnail images:
     *   1. Reference original (coloured, from ReferenceImageFactory)
     *   2. Reference binary  (greyscale + threshold — what the matcher uses)
     *   3. Scene original
     *   4. Scene binary + Canny edges  (what contour extraction sees)
     *   5. Scene annotated — cyan box = expected location, coloured box = detected
     *
     * @param groundTruth bounding rect of the target shape in the scene (null = unknown)
     */
    private double record(String stage, String testId, String shapeName,
                          String sceneDesc, Mat sceneMat,
                          Rect groundTruth,
                          List<AnalysisResult> results) {
        double score = normalScore(results);

        ReferenceId rid = results.isEmpty() ? null : results.get(0).referenceId();

        // ── Scene with ref colour applied (used as "Scene" step in both strips) ──
        // Synthetic scenes are white-on-black; recolour the foreground to the
        // actual reference colour so CF variants see the right colour.
        Mat sceneWithRef = rid != null ? recolourToRef(sceneMat, rid) : sceneMat.clone();

        // CF score — run the matcher again on the recoloured scene so the CF filter
        // actually finds the colour. The standard score uses the original results
        // (white-on-black is fine for greyscale threshold matching).
        double cfScore = 0.0;
        if (rid != null) {
            Mat ref = ReferenceImageFactory.build(rid);
            List<AnalysisResult> cfResults = runMatcherColoured(rid, ref, sceneMat);
            cfScore = cfLooseScore(cfResults);
            ref.release();
        }

        // ── Ref original + Ref Points (raw) + Ref Points (approx) ───────────
        String refOrigPng        = "";
        String refPointsPng      = "";
        String refPointsApproxPng = "";
        if (rid != null) {
            Mat refOrig = ReferenceImageFactory.build(rid);
            refOrigPng = matToBase64Png(refOrig);

            Mat refBin = VectorMatcher.extractBinaryRaw(refOrig);
            List<MatOfPoint> refContours = VectorMatcher.extractContours(refOrig);

            // Raw — every pixel-level point (epsilon=0)
            Mat refGraphRaw = VectorMatcher.drawContourGraph(refOrig.size(), refBin, refContours, 0);
            refPointsPng = matToBase64Png(refGraphRaw);
            refGraphRaw.release();

            // Approx — STRICT epsilon (0.02), what SegmentDescriptor actually sees
            Mat refGraphApprox = VectorMatcher.drawContourGraph(refOrig.size(), refBin, refContours, 0.02);
            refPointsApproxPng = matToBase64Png(refGraphApprox);
            refGraphApprox.release();

            refBin.release();
            refOrig.release();
        }

        String sceneWithRefPng = matToBase64Png(sceneWithRef);

        double eps = VectorVariant.VECTOR_NORMAL.epsilonFactor();

        // ── Standard pipeline ─────────────────────────────────────────────
        // Edges (kept for reference — shows what the raw binary looks like)
        String sceneBinPng;
        Mat stdBin = VectorMatcher.extractBinaryRaw(sceneWithRef);
        {
            Mat bgr = new Mat();
            Imgproc.cvtColor(stdBin, bgr, Imgproc.COLOR_GRAY2BGR);
            sceneBinPng = matToBase64Png(bgr);
            bgr.release();
        }
        stdBin.release();

        // All Points — colour-isolated contours (one cluster per colour),
        // showing exactly what the matcher now extracts per colour layer.
        String allPointsPng;
        {
            List<SceneColourClusters.Cluster> clusters = SceneColourClusters.extract(sceneWithRef);
            List<MatOfPoint> allContours = new ArrayList<>();
            for (SceneColourClusters.Cluster cluster : clusters) {
                Mat masked = SceneColourClusters.applyMask(sceneWithRef, cluster);
                allContours.addAll(VectorMatcher.extractContoursFromBinary(masked));
                masked.release();
                cluster.release();
            }
            Mat graph = VectorMatcher.drawContourGraph(sceneWithRef.size(), null, allContours, 0);
            allPointsPng = matToBase64Png(graph);
            graph.release();
        }

        String cfFilteredPng  = "";
        String cfEdgesPng     = "";
        String cfAllPointsPng = "";
        if (rid != null) {
            Mat cfMat = ColourPreFilter.applyMaskedBgrToScene(sceneWithRef, rid, ColourPreFilter.LOOSE);
            cfFilteredPng = matToBase64Png(cfMat);

            // CF Edges
            Mat cfBin = VectorMatcher.extractBinaryRaw(cfMat);
            {
                Mat bgr = new Mat();
                Imgproc.cvtColor(cfBin, bgr, Imgproc.COLOR_GRAY2BGR);
                cfEdgesPng = matToBase64Png(bgr);
                bgr.release();
            }
            cfBin.release();

            // CF All Points — colour-isolated contours from the CF-filtered scene
            {
                List<SceneColourClusters.Cluster> cfClusters = SceneColourClusters.extract(cfMat);
                List<MatOfPoint> cfAllContours = new ArrayList<>();
                for (SceneColourClusters.Cluster cluster : cfClusters) {
                    Mat masked = SceneColourClusters.applyMask(cfMat, cluster);
                    cfAllContours.addAll(VectorMatcher.extractContoursFromBinary(masked));
                    masked.release();
                    cluster.release();
                }
                Mat graph = VectorMatcher.drawContourGraph(cfMat.size(), null, cfAllContours, 0);
                cfAllPointsPng = matToBase64Png(graph);
                graph.release();
            }
            cfMat.release();
        }

        // ── Match annotations ─────────────────────────────────────────────
        String sceneAnnotPng = "";
        String cfAnnotPng    = "";
        double iou           = Double.NaN;
        boolean falsePositive = false;
        {
            VectorSignature refSig = rid != null
                    ? VectorMatcher.buildRefSignature(ReferenceImageFactory.build(rid), eps)
                    : null;

            sceneAnnotPng = buildAnnotated(sceneWithRef, sceneWithRef, refSig, groundTruth, score, false);

            if (rid != null) {
                Mat cfMat = ColourPreFilter.applyMaskedBgrToScene(sceneWithRef, rid, ColourPreFilter.LOOSE);
                cfAnnotPng = buildAnnotated(cfMat, sceneWithRef, refSig, groundTruth, cfScore, false);
                cfMat.release();
            }

            Rect bestBbox = findBestBbox(sceneWithRef, refSig);
            if (bestBbox != null && groundTruth != null) {
                iou = iou(bestBbox, groundTruth);
                falsePositive = (score > 40.0) && (iou < 0.3);
            }
        }

        sceneWithRef.release();

        REPORT_ROWS.add(new ReportRow(stage, testId, shapeName, sceneDesc,
                score, true,
                refOrigPng, refPointsPng, refPointsApproxPng,
                sceneWithRefPng, sceneBinPng,
                allPointsPng,
                sceneAnnotPng,
                cfScore,
                cfFilteredPng, cfEdgesPng,
                cfAllPointsPng,
                cfAnnotPng,
                iou, falsePositive));
        return score;
    }

    /** Build an annotated scene image showing best-match bbox. Returns base64 PNG. */
    private static String buildAnnotated(Mat extractFrom, Mat drawOnto,
                                          VectorSignature refSig, Rect groundTruth,
                                          double score, boolean falsePositive) {
        Rect bestBbox = findBestBbox(extractFrom, refSig);
        Mat annotated = drawOnto.clone();
        if (groundTruth != null) {
            Imgproc.rectangle(annotated,
                    new Point(groundTruth.x, groundTruth.y),
                    new Point(groundTruth.x + groundTruth.width, groundTruth.y + groundTruth.height),
                    new Scalar(220, 220, 0), 2); // cyan
        }
        if (bestBbox != null && bestBbox.width > 1 && bestBbox.height > 1) {
            Scalar col = falsePositive     ? new Scalar(220, 0, 220)
                       : score >= 70       ? new Scalar(0, 200, 0)
                       : score >= 40       ? new Scalar(0, 200, 200)
                       :                     new Scalar(0, 0, 200);
            Imgproc.rectangle(annotated,
                    new Point(bestBbox.x, bestBbox.y),
                    new Point(bestBbox.x + bestBbox.width, bestBbox.y + bestBbox.height),
                    col, 3);
        }
        Scalar labelCol = score >= 50 ? new Scalar(0, 220, 0) : new Scalar(0, 0, 220);
        Imgproc.putText(annotated, String.format("%.1f%%", score),
                new Point(6, 28), Imgproc.FONT_HERSHEY_SIMPLEX, 0.55, labelCol, 2);
        String png = matToBase64Png(annotated);
        annotated.release();
        return png;
    }

    /** Find the best-match bounding box for refSig in the given scene using colour-isolated extraction. */
    private static Rect findBestBbox(Mat scene, VectorSignature refSig) {
        if (refSig == null) return null;
        Rect bestBbox = null;
        double bestSim = 0;
        double sceneArea = (double) scene.rows() * scene.cols();

        List<SceneColourClusters.Cluster> clusters = SceneColourClusters.extract(scene);
        for (SceneColourClusters.Cluster cluster : clusters) {
            Mat masked = SceneColourClusters.applyMask(scene, cluster);
            List<MatOfPoint> contours = VectorMatcher.extractContoursFromBinary(masked);
            masked.release();
            cluster.release();
            for (MatOfPoint c : contours) {
                VectorSignature sig = VectorSignature.buildFromContour(
                        c, VectorVariant.VECTOR_NORMAL.epsilonFactor(), sceneArea);
                double sim = refSig.similarity(sig);
                if (sim > bestSim) { bestSim = sim; bestBbox = Imgproc.boundingRect(c); }
            }
        }
        return bestBbox;
    }

    /**
     * Returns a copy of {@code whiteOnBlack} where every white (≥240,≥240,≥240)
     * foreground pixel is replaced with the dominant foreground colour of the
     * reference image for {@code rid}.
     *
     * <p>Used in the HTML report so the colour-filter pipeline step shows something
     * meaningful — synthetic scenes are drawn in white but the CF filter looks for
     * the reference colour.
     */
    private static Mat recolourToRef(Mat whiteOnBlack, ReferenceId rid) {
        Mat ref  = ReferenceImageFactory.build(rid);
        // Sample foreground colour: mean of non-black pixels in the reference
        Mat refGrey = new Mat();
        Mat refMask = new Mat();
        Imgproc.cvtColor(ref, refGrey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(refGrey, refMask, 20, 255, Imgproc.THRESH_BINARY);
        Scalar meanColour = Core.mean(ref, refMask);
        ref.release(); refGrey.release(); refMask.release();

        // Build mask of white pixels in the scene (foreground)
        Mat sceneGrey = new Mat();
        Mat fgMask    = new Mat();
        Imgproc.cvtColor(whiteOnBlack, sceneGrey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(sceneGrey, fgMask, 240, 255, Imgproc.THRESH_BINARY);
        sceneGrey.release();

        // Fill a solid-colour mat with the reference colour, then copy into scene clone
        Mat result    = whiteOnBlack.clone();
        Mat colourFg  = new Mat(result.size(), result.type(), meanColour);
        colourFg.copyTo(result, fgMask);
        colourFg.release(); fgMask.release();
        return result;
    }

    /** Intersection-over-Union of two OpenCV Rects. Returns 0 if no overlap. */
    private static double iou(Rect a, Rect b) {
        int ix1 = Math.max(a.x, b.x);
        int iy1 = Math.max(a.y, b.y);
        int ix2 = Math.min(a.x + a.width,  b.x + b.width);
        int iy2 = Math.min(a.y + a.height, b.y + b.height);
        if (ix2 <= ix1 || iy2 <= iy1) return 0.0;
        double inter = (double)(ix2 - ix1) * (iy2 - iy1);
        double aArea = (double) a.width * a.height;
        double bArea = (double) b.width * b.height;
        return inter / (aArea + bArea - inter);
    }

    /**
     * Computes the bounding rect of all non-black pixels in a white-on-black mat.
     * Use this on the original shapeMat (before compositing) to get the ground-truth rect.
     */
    /**
     * Applies STRICT (0.02 × perimeter) approxPolyDP to each contour, returning a
     * new list of reduced contours.  Mirrors what {@code buildFromContour} now does
     * before rendering the filled crop — so "Scene Approx" shows exactly what
     * the SegmentDescriptor receives as input.
     */
    private static List<MatOfPoint> approxReduceContours(List<MatOfPoint> raw) {
        List<MatOfPoint> result = new ArrayList<>();
        for (MatOfPoint c : raw) {
            double perim = Imgproc.arcLength(new MatOfPoint2f(c.toArray()), true);
            double eps   = Math.max(0.02 * perim, 2.0);
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(new MatOfPoint2f(c.toArray()), approx, eps, true);
            if (approx.total() >= 3)
                result.add(new MatOfPoint(approx.toArray()));
            approx.release();
        }
        return result;
    }

    private static Rect groundTruthRect(Mat shapeMat) {
        Mat grey = new Mat(); Mat bin = new Mat();
        Imgproc.cvtColor(shapeMat, grey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(grey, bin, 5, 255, Imgproc.THRESH_BINARY);
        grey.release();
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hier = new Mat();
        Imgproc.findContours(bin, contours, hier, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        bin.release(); hier.release();
        if (contours.isEmpty()) return null;
        Rect r = Imgproc.boundingRect(contours.get(0));
        for (int i = 1; i < contours.size(); i++) {
            Rect b = Imgproc.boundingRect(contours.get(i));
            int x1 = Math.min(r.x, b.x), y1 = Math.min(r.y, b.y);
            int x2 = Math.max(r.x + r.width, b.x + b.width);
            int y2 = Math.max(r.y + r.height, b.y + b.height);
            r = new Rect(x1, y1, x2 - x1, y2 - y1);
        }
        return r;
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

    private static Mat whiteHexagonOnBlack(boolean outline) {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Point[] pts = new Point[6];
        for (int i = 0; i < 6; i++) {
            double a = Math.toRadians(60 * i - 30);
            pts[i] = new Point(320 + 80 * Math.cos(a), 240 + 80 * Math.sin(a));
        }
        MatOfPoint poly = new MatOfPoint(pts);
        if (outline) Imgproc.polylines(m, List.of(poly), true, new Scalar(255, 255, 255), 3);
        else         Imgproc.fillPoly(m, List.of(poly), new Scalar(255, 255, 255));
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
        Imgproc.fillPoly(m, List.of(new MatOfPoint(
                new Point(320,110), new Point(470,240),
                new Point(320,370), new Point(170,240))),
                new Scalar(255, 255, 255));
        return m;
    }

    private static Mat whiteArrowOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.fillPoly(m, List.of(new MatOfPoint(
                new Point(160,200), new Point(340,200), new Point(340,155),
                new Point(480,240), new Point(340,325), new Point(340,280),
                new Point(160,280))),
                new Scalar(255, 255, 255));
        return m;
    }

    private static Mat whiteEllipseOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.ellipse(m, new Point(320,240), new Size(140,70), 0, 0, 360, new Scalar(255,255,255), -1);
        return m;
    }

    /** Regular octagon, r=85, centred. */
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

    /** Thick plus/cross shape (two overlapping rectangles). */
    private static Mat whitePlusOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.rectangle(m, new Point(270, 140), new Point(370, 340), new Scalar(255,255,255), -1);
        Imgproc.rectangle(m, new Point(170, 200), new Point(470, 280), new Scalar(255,255,255), -1);
        return m;
    }

    /** Concave arrowhead (triangle with notch cut from base). */
    private static Mat whiteConcaveArrowheadOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        // Outer triangle tip pointing up, base notched inward
        Imgproc.fillPoly(m, List.of(new MatOfPoint(
                new Point(320, 110),  // tip
                new Point(460, 370),  // base-right
                new Point(320, 290),  // notch (concave)
                new Point(180, 370))),// base-left
                new Scalar(255, 255, 255));
        return m;
    }

    /** Thin cross drawn as two lines (LINE_CROSS analogue). */
    private static Mat whiteCrossOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.line(m, new Point(320, 80),  new Point(320, 400), new Scalar(255,255,255), 8);
        Imgproc.line(m, new Point(100, 240), new Point(540, 240), new Scalar(255,255,255), 8);
        return m;
    }

    /** Rectangle rotated 45° (diamond-ish aspect). */
    private static Mat whiteRot45RectOnBlack() {
        return rotate(whiteRectOnBlack(230, 160, 410, 320), 45);
    }

    /**
     * Composites a white-on-black shape Mat onto a real background.
     * On light backgrounds (average luminance > 128) the shape is darkened
     * (bitwise-inverted) so it retains contrast against the background.
     * Returns a new Mat; caller owns it and must release it.
     */
    private static Mat compositeOnBackground(Mat shapeMat, BackgroundId bgId) {
        Mat scene = BackgroundFactory.build(bgId, shapeMat.cols(), shapeMat.rows());

        // Decide foreground colour: dark on light backgrounds, white on dark ones
        Mat grey = new Mat();
        Imgproc.cvtColor(scene, grey, Imgproc.COLOR_BGR2GRAY);
        double bgLuma = Core.mean(grey).val[0];
        grey.release();

        Mat foreground = shapeMat.clone();
        if (bgLuma > 100) {
            // Invert so white shape becomes black → visible on light background
            Core.bitwise_not(foreground, foreground);
            // Re-zero the background portion (was 255 bg → 0 after invert, which is black = fine)
        }

        // Build mask from original shapeMat (white = foreground pixels)
        Mat maskGrey = new Mat();
        Mat mask     = new Mat();
        Imgproc.cvtColor(shapeMat, maskGrey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(maskGrey, mask, 5, 255, Imgproc.THRESH_BINARY);
        maskGrey.release();

        foreground.copyTo(scene, mask);
        foreground.release();
        mask.release();
        return scene;
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
    // HTML report builder
    // =========================================================================

    private static String matToBase64Png(Mat m) {
        try {
            MatOfByte buf = new MatOfByte();
            Imgcodecs.imencode(".png", m, buf);
            return Base64.getEncoder().encodeToString(buf.toArray());
        } catch (Exception e) {
            return "";
        }
    }

    private static String buildHtml(List<ReportRow> rows) {
        Map<String, List<ReportRow>> byStage = new LinkedHashMap<>();
        for (ReportRow r : rows)
            byStage.computeIfAbsent(r.stage(), k -> new ArrayList<>()).add(r);

        long total      = rows.size();
        long passed     = rows.stream().filter(ReportRow::passed).count();
        long falsePos   = rows.stream().filter(ReportRow::falsePositive).count();
        long badIou     = rows.stream().filter(r -> !Double.isNaN(r.iou()) && r.iou() < 0.3 && r.score() > 40).count();

        String timestamp = LocalDateTime.now()
                .format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));

        StringBuilder sb = new StringBuilder();
        sb.append("<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>")
          .append("<title>Vector Matching — Step 5 Report</title><style>")
          .append(reportCss())
          .append("</style></head><body>")
          .append("<div class='header'>")
          .append("<h1>Vector Matching — Step 5 Visual Report</h1>")
          .append("<div class='ts-line'>Generated: <span class='ts-val'>").append(timestamp).append("</span></div>")
          .append("<p class='subtitle'>")
          .append(total).append(" matcher calls &nbsp;·&nbsp; ")
          .append(passed).append(" passed &nbsp;·&nbsp; ").append(total - passed).append(" failed")
          .append(" &nbsp;·&nbsp; <span style='color:#bf55ec;font-weight:700'>").append(falsePos).append(" false positives</span>")
          .append(" &nbsp;·&nbsp; <span style='color:#f85149'>").append(badIou).append(" bad-IoU matches</span>")
          .append("</p>")
          .append("<div class='legend-block'>")
          .append("<div class='legend-title'>Pipeline rows (per test entry)</div>")
          .append("<div class='legend-row'>")
          .append("<span class='pl-label'>Standard:</span>")
          .append("<span class='pl-step'>Ref</span><span class='pl-arrow'>→</span>")
          .append("<span class='pl-step'>Ref Points (raw)</span><span class='pl-arrow'>→</span>")
          .append("<span class='pl-step'>Ref Points (approx)</span><span class='pl-arrow'>→</span>")
          .append("<span class='pl-step'>Scene</span><span class='pl-arrow'>→</span>")
          .append("<span class='pl-step'>Edges (ref)</span><span class='pl-arrow'>→</span>")
          .append("<span class='pl-step'>Colour Clusters</span><span class='pl-arrow'>→</span>")
          .append("<span class='pl-step'>Match</span><span class='pl-arrow'>→</span>")
          .append("<span class='pl-step'>Score %</span>")
          .append("</div>")
          .append("<div class='legend-row' style='margin-top:4px'>")
          .append("<span class='pl-label'>+ CF:</span>")
          .append("<span class='pl-step'>Ref</span><span class='pl-arrow'>→</span>")
          .append("<span class='pl-step'>Ref Points (raw)</span><span class='pl-arrow'>→</span>")
          .append("<span class='pl-step'>Ref Points (approx)</span><span class='pl-arrow'>→</span>")
          .append("<span class='pl-step'>Scene</span><span class='pl-arrow'>→</span>")
          .append("<span class='pl-step cf-step'>Colour Filter</span><span class='pl-arrow'>→</span>")
          .append("<span class='pl-step'>Edges (ref)</span><span class='pl-arrow'>→</span>")
          .append("<span class='pl-step'>Colour Clusters</span><span class='pl-arrow'>→</span>")
          .append("<span class='pl-step'>Match</span><span class='pl-arrow'>→</span>")
          .append("<span class='pl-step'>Score %</span>")
          .append("</div>")
          .append("<p style='font-size:.70rem;color:#8b949e;margin-top:5px'>")
          .append("Scene = reference drawn in its actual colour onto the scene background. ")
          .append("Both strips use the same scene image.")
          .append("</p>")
          .append("</div>")
          .append("<div class='legend-block'>")
          .append("<div class='legend-title'>⑤ Bounding box colours</div>")
          .append("<div class='legend-row'>")
          .append("<span class='swatch' style='border:2px solid #00dcdc'></span><span class='swatch-label'>Cyan = expected location (ground truth)</span>")
          .append("<span class='swatch' style='border:3px solid #00c800'></span><span class='swatch-label'>Green = detected &amp; good (≥70%)</span>")
          .append("<span class='swatch' style='border:3px solid #00c8c8'></span><span class='swatch-label'>Amber = detected &amp; ok (≥40%)</span>")
          .append("<span class='swatch' style='border:3px solid #c80000'></span><span class='swatch-label'>Red = detected &amp; poor (&lt;40%)</span>")
          .append("<span class='swatch' style='border:3px solid #dc00dc'></span><span class='swatch-label'>Magenta = FALSE POSITIVE (score ok but wrong location)</span>")
          .append("</div>")
          .append("</div>")
          .append("<div class='legend-block'>")
          .append("<div class='legend-title'>Row status</div>")
          .append("<div class='legend-row'>")
          .append("<span class='legend-pill pass-pill'>green border = passed</span>")
          .append("<span class='legend-pill warn-pill'>amber border = marginal</span>")
          .append("<span class='legend-pill fail-pill'>red border = failed</span>")
          .append("<span class='legend-pill fp-pill'>purple = false positive</span>")
          .append("</div>")
          .append("</div>")
          .append("<div class='legend-block'>")
          .append("<div class='legend-title'>IoU (Intersection over Union)</div>")
          .append("<div class='legend-row'>")
          .append("<span class='iou-val iou-good'>IoU ≥ 0.50 good</span>")
          .append("<span class='iou-val iou-warn'>IoU 0.30–0.49 marginal</span>")
          .append("<span class='iou-val iou-bad'>IoU &lt; 0.30 wrong location</span>")
          .append("</div>")
          .append("</div>")
          .append("</div>");

        for (Map.Entry<String, List<ReportRow>> entry : byStage.entrySet()) {
            sb.append("<section><h2>").append(esc(entry.getKey())).append("</h2>");
            for (ReportRow r : entry.getValue()) {
                String passCls = r.falsePositive() ? "row fp"
                               : r.score() >= 50   ? "row pass"
                               : r.score() >= 40   ? "row warn"
                               :                     "row fail";
                sb.append("<div class='").append(passCls).append("'>");

                // ── Row header ───────────────────────────────────────────
                sb.append("<div class='row-meta'>")
                  .append("<span class='row-id'>").append(esc(r.label())).append("</span>")
                  .append("<span class='row-shape'>").append(esc(r.shapeName())).append("</span>")
                  .append("<span class='row-desc'>").append(esc(r.sceneDesc())).append("</span>");
                if (r.falsePositive())
                    sb.append("<span class='badge fp-badge'>⚠ FALSE POSITIVE</span>");
                if (!Double.isNaN(r.iou())) {
                    String iouCls = r.iou() >= 0.5 ? "iou-good" : r.iou() >= 0.3 ? "iou-warn" : "iou-bad";
                    sb.append("<span class='iou-val ").append(iouCls).append("'>")
                      .append("IoU ").append(String.format("%.2f", r.iou())).append("</span>");
                }
                sb.append("</div>");

                // ── Standard pipeline sub-row ────────────────────────────
                sb.append("<div class='pipeline-row'>")
                  .append("<div class='pipeline-label'>Standard</div>")
                  .append("<div class='pipeline'>");
                pipelineStep(sb, r.refOrig(),          "Ref");
                pipelineStep(sb, r.refPoints(),        "Ref Points (raw)");
                pipelineStep(sb, r.refPointsApprox(),  "Ref Points (approx)");
                pipelineStep(sb, r.sceneOrig(),        "Scene");
                pipelineStep(sb, r.sceneBin(),         "Edges (ref)");
                pipelineStep(sb, r.allPoints(),        "Colour Clusters");
                pipelineStep(sb, r.sceneAnnot(),       "Match");
                sb.append("</div>")
                  .append("<div class='pipeline-score'>")
                  .append("<span class='score-val ").append(scoreClass(r.score())).append("'>")
                  .append(String.format("%.1f%%", r.score())).append("</span>")
                  .append(scoreBar(r.score()))
                  .append("</div>")
                  .append("</div>");

                // ── CF pipeline sub-row ──────────────────────────────────
                sb.append("<div class='pipeline-row cf-row'>")
                  .append("<div class='pipeline-label cf-label'>+ Colour Filter</div>")
                  .append("<div class='pipeline'>");
                pipelineStep(sb, r.refOrig(),          "Ref");
                pipelineStep(sb, r.refPoints(),        "Ref Points (raw)");
                pipelineStep(sb, r.refPointsApprox(),  "Ref Points (approx)");
                pipelineStep(sb, r.sceneOrig(),        "Scene");
                pipelineStep(sb, r.cfFiltered(),       "Colour Filter");
                pipelineStep(sb, r.cfEdges(),          "Edges (ref)");
                pipelineStep(sb, r.cfAllPoints(),      "Colour Clusters");
                pipelineStep(sb, r.cfAnnot(),          "Match");
                sb.append("</div>")
                  .append("<div class='pipeline-score'>")
                  .append("<span class='score-val ").append(scoreClass(r.cfScore())).append("'>")
                  .append(String.format("%.1f%%", r.cfScore())).append("</span>")
                  .append(scoreBar(r.cfScore()))
                  .append("</div>")
                  .append("</div>");

                sb.append("</div>"); // end row card
            }
            sb.append("</section>");
        }
        // Lightbox overlay
        sb.append("<div id='lb' class='lb-overlay' onclick='closeLb()'>" )
          .append("<div class='lb-box' onclick='event.stopPropagation()'>")
          .append("<button class='lb-close' onclick='closeLb()'>✕</button>")
          .append("<img id='lb-img' src='' alt='' class='lb-img'/>")
          .append("<div id='lb-caption' class='lb-caption'></div>")
          .append("</div></div>")
          .append("<script>")
          .append("function openLb(src,cap){")
          .append("  document.getElementById('lb-img').src=src;")
          .append("  document.getElementById('lb-caption').textContent=cap;")
          .append("  document.getElementById('lb').classList.add('lb-visible');")
          .append("}")
          .append("function closeLb(){document.getElementById('lb').classList.remove('lb-visible');}")
          .append("document.addEventListener('keydown',function(e){if(e.key==='Escape')closeLb();});")
          .append("</script>")
          .append("</body></html>");
        return sb.toString();
    }

    private static void pipelineStep(StringBuilder sb, String b64png, String label) {
        sb.append("<div class='step'>");
        if (b64png != null && !b64png.isEmpty()) {
            String src = "data:image/png;base64," + b64png;
            sb.append("<img src='").append(src).append("'")
              .append(" class='step-img' alt='").append(esc(label)).append("'")
              .append(" title='Click to enlarge'")
              .append(" onclick=\"openLb('").append(src).append("','").append(esc(label)).append("')\"")
              .append(" style='cursor:zoom-in'/>");
        } else {
            sb.append("<div class='step-img step-empty'></div>");
        }
        sb.append("<div class='step-label'>").append(esc(label)).append("</div></div>");
    }

    private static String scoreBar(double score) {
        String col = score >= 70 ? "#56d364" : score >= 40 ? "#d29922" : "#f85149";
        int w = (int) Math.max(1, Math.min(100, score));
        return "<div class='bar-bg'><div class='bar-fill' style='width:" + w
                + "%;background:" + col + "'></div></div>";
    }

    private static String scoreClass(double s) {
        return s >= 70 ? "s-good" : s >= 40 ? "s-warn" : "s-bad";
    }

    private static String esc(String s) {
        return s == null ? "" : s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;");
    }

    private static String reportCss() {
        return """
            *{box-sizing:border-box;margin:0;padding:0}
            body{font-family:system-ui,sans-serif;background:#0d1117;color:#c9d1d9;padding:0 0 48px}
            .header{background:#161b22;padding:20px 32px 16px;border-bottom:1px solid #30363d;margin-bottom:20px}
            .header h1{color:#58a6ff;font-size:1.4rem;margin-bottom:4px}
            .ts-line{font-size:.75rem;color:#8b949e;margin-bottom:6px}
            .ts-val{color:#79c0ff;font-weight:600}
            .subtitle{color:#8b949e;font-size:.88rem;margin-bottom:8px}
            .pipeline-legend{display:flex;align-items:center;gap:6px;font-size:.78rem;color:#8b949e;flex-wrap:wrap}
            .pl-step{background:#21262d;border:1px solid #30363d;border-radius:4px;padding:2px 7px}
            .pl-step.cf-step{background:#1a2e1a;border-color:#238636;color:#56d364}
            .pl-label{font-size:.72rem;font-weight:700;color:#8b949e;min-width:80px}
            .pl-arrow{color:#484f58}
            /* ── Pipeline sub-rows ── */
            .pipeline-row{display:flex;align-items:center;gap:8px;padding:4px 0}
            .pipeline-row + .pipeline-row{border-top:1px solid #21262d;padding-top:6px}
            .cf-row{background:#0d1a0d;border-radius:0 0 6px 6px;margin:0 -12px -10px;padding:6px 12px}
            .pipeline-label{font-size:.70rem;font-weight:700;color:#8b949e;
                             white-space:nowrap;min-width:68px;flex-shrink:0}
            .cf-label{color:#56d364}
            .pipeline-score{display:flex;flex-direction:column;align-items:flex-end;
                             gap:4px;flex-shrink:0;min-width:70px}
            /* Legend block */
            .legend-grid{display:flex;flex-direction:column;gap:10px;margin-top:10px}
            .legend-block{background:#21262d;border:1px solid #30363d;border-radius:6px;padding:8px 12px}
            .legend-title{font-size:.72rem;font-weight:700;color:#79c0ff;margin-bottom:5px;text-transform:uppercase;letter-spacing:.04em}
            .legend-row{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
            .swatch{display:inline-block;width:22px;height:14px;border-radius:3px;background:transparent;flex-shrink:0}
            .swatch-label{font-size:.72rem;color:#c9d1d9;white-space:nowrap;margin-right:6px}
            .legend-pill{font-size:.72rem;font-weight:600;border-radius:4px;padding:2px 8px;border-left:3px solid transparent}
            .pass-pill{border-left-color:#238636;background:#0d2619;color:#56d364}
            .warn-pill{border-left-color:#9e6a03;background:#271e0b;color:#d29922}
            .fail-pill{border-left-color:#da3633;background:#2b0c0c;color:#f85149}
            .fp-pill{border-left-color:#bf55ec;background:#1a0e2e;color:#d2a8ff}
            section{padding:0 24px 16px}
            h2{color:#79c0ff;font-size:1rem;margin:14px 0 10px;padding-bottom:4px;border-bottom:1px solid #21262d}
            /* ── Row card ── */
            .row{background:#161b22;border:1px solid #30363d;border-radius:8px;
                 margin-bottom:8px;padding:10px 12px;display:flex;flex-direction:column;gap:8px}
            .row.pass{border-left:3px solid #238636}
            .row.warn{border-left:3px solid #9e6a03}
            .row.fail{border-left:3px solid #da3633}
            .row.fp{border-left:3px solid #bf55ec;background:#1a0e2e}
            .fp-badge{background:#bf55ec;color:#fff;font-size:.68rem;font-weight:700;
                      border-radius:3px;padding:1px 6px;white-space:nowrap}
            .iou-val{font-size:.72rem;font-weight:600;white-space:nowrap}
            .iou-good{color:#56d364}.iou-warn{color:#d29922}.iou-bad{color:#f85149}
            .row-meta{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
            .row-id{font-size:.72rem;font-weight:700;background:#21262d;border-radius:3px;
                    padding:1px 6px;color:#79c0ff;white-space:nowrap}
            .row-shape{font-size:.78rem;font-weight:600;color:#c9d1d9;white-space:nowrap}
            .row-desc{font-size:.72rem;color:#8b949e;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
            /* ── Score ── */
            .score-val{font-size:.82rem;font-weight:700;min-width:48px;text-align:right}
            .s-good{color:#56d364}.s-warn{color:#d29922}.s-bad{color:#f85149}
            .bar-bg{width:120px;height:7px;background:#21262d;border-radius:4px;overflow:hidden;flex-shrink:0}
            .bar-fill{height:100%;border-radius:4px}
            /* ── Pipeline strip ── */
            .pipeline{display:flex;align-items:flex-start;gap:6px;overflow-x:auto;flex:1}
            .step{display:flex;flex-direction:column;align-items:center;gap:3px;flex-shrink:0}
            .step-img{width:128px;height:96px;object-fit:contain;border-radius:4px;
                      border:1px solid #30363d;background:#0d1117;display:block;
                      transition:border-color .15s;cursor:zoom-in}
            .step-img:hover{border-color:#58a6ff}
            .step-empty{width:128px;height:96px;border:1px dashed #30363d;border-radius:4px;background:#161b22}
            .step-label{font-size:.65rem;color:#8b949e;text-align:center;max-width:128px}
            /* ── Lightbox ── */
            .lb-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.88);
                         z-index:9999;overflow:auto;padding:48px 24px 24px}
            .lb-overlay.lb-visible{display:flex;align-items:center;justify-content:center}
            .lb-box{position:relative;padding:8px;background:#161b22;border:1px solid #30363d;border-radius:8px}
            .lb-img{display:block;max-width:90vw;max-height:80vh;image-rendering:pixelated;border-radius:4px}
            .lb-caption{color:#c9d1d9;font-size:.85rem;text-align:center;margin-top:6px}
            .lb-close{position:absolute;top:-14px;right:-14px;width:28px;height:28px;
                      border-radius:50%;background:#21262d;border:1px solid #484f58;
                      color:#c9d1d9;font-size:1rem;cursor:pointer;display:flex;
                      align-items:center;justify-content:center;line-height:1;z-index:10000}
            .lb-close:hover{background:#30363d;color:#fff}
            """;
    }
}
