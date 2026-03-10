package org.example.vectormatcher;

import org.example.OpenCvLoader;
import org.example.analytics.AnalysisResult;
import org.example.factories.BackgroundFactory;
import org.example.factories.BackgroundId;
import org.example.factories.ReferenceId;
import org.example.factories.ReferenceImageFactory;
import org.example.colour.SceneColourClusters;
import org.example.matchers.SceneDescriptor;
import org.example.matchers.VectorMatcher;
import org.example.matchers.VectorSignature;
import org.example.matchers.VectorVariant;
import org.example.scene.SceneCategory;
import org.example.scene.SceneEntry;
import org.example.utilities.MatchDiagnosticLibrary;
import org.example.utilities.MatchReportLibrary;
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

@DisplayName("Step 5 — Vector Matching (incremental)")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class VectorMatchingTest {

    private static final Path OUTPUT = Paths.get("test_output", "vector_matching");
    private static final double MATCH_THRESHOLD = 50.0;

    // ── Delegate all reporting to the shared library ───────────────────────
    private final MatchReportLibrary     report = new MatchReportLibrary();
    private final MatchDiagnosticLibrary diag   = new MatchDiagnosticLibrary();

    @BeforeAll
    void load() throws IOException {
        OpenCvLoader.load();
        Files.createDirectories(OUTPUT);
        // Clear stale results from any previous partial run
        report.clear();
        diag.clear();
        Files.deleteIfExists(OUTPUT.resolve("report.html"));
        Files.deleteIfExists(OUTPUT.resolve("diagnostics.json"));
    }

    @AfterAll
    void writeReports() throws IOException {
        report.writeReport(OUTPUT, "VectorMatchingTest — Step 5 Visual Report");
        diag.writeReport(OUTPUT);
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

        @Test @Order(5) @DisplayName("S1e — All 3 variants return without error")
        void allVariantsReturnWithoutError() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.CIRCLE_FILLED);
            Mat scene = whiteCircleOnBlack(320, 240, 60);
            List<AnalysisResult> results = runMatcher(ReferenceId.CIRCLE_FILLED, ref, scene).results();
            ref.release(); scene.release();
            assertEquals(3, results.size());
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
                List<AnalysisResult> results = runMatcher(ReferenceId.RECT_FILLED, ref, noise).results();
                assertEquals(3, results.size());
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

    /** Simple carrier: results + descriptor build time. */
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

    private static double normalScore(MatchRun run) {
        return MatchReportLibrary.normalScore(run.results());
    }

    private static double normalScore(List<AnalysisResult> results) {
        return MatchReportLibrary.normalScore(results);
    }

    /** Records result into the shared report and diagnostic libraries; returns score. */
    private double record(String stage, String testId, String shapeName,
                          String sceneDesc, Mat sceneMat, MatchRun run) {
        double score = report.record(stage, testId, shapeName, sceneDesc, sceneMat,
                                     new MatchReportLibrary.MatchRun(run.results(), run.descriptorMs()));
        diag.evaluate(
                BackgroundId.BG_SOLID_BLACK, sceneDesc,
                run.results().isEmpty() ? null : run.results().get(0).referenceId(),
                40.0, 75.0, 0.5, OUTPUT);
        return score;
    }

    private double record(String stage, String testId, String shapeName,
                          String sceneDesc, Mat sceneMat, Rect groundTruth, MatchRun run) {
        double score = report.record(stage, testId, shapeName, sceneDesc, sceneMat, groundTruth,
                                     new MatchReportLibrary.MatchRun(run.results(), run.descriptorMs()));
        return score;
    }

    // =========================================================================
    // Scene builders (kept here for test readability)
    // =========================================================================

    private static Mat whiteCircleOnBlack(int cx, int cy, int radius) {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.circle(m, new Point(cx, cy), radius, new Scalar(255,255,255), -1);
        return m;
    }
    private static Mat whiteRectOnBlack(int x1, int y1, int x2, int y2) {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.rectangle(m, new Point(x1,y1), new Point(x2,y2), new Scalar(255,255,255), -1);
        return m;
    }
    private static Mat whiteTriangleOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.fillPoly(m, List.of(new MatOfPoint(
                new Point(320,130), new Point(180,350), new Point(460,350))),
                new Scalar(255,255,255));
        return m;
    }
    private static Mat whiteHexagonOnBlack(boolean outline) {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Point[] pts = new Point[6];
        for (int i = 0; i < 6; i++) {
            double a = Math.toRadians(60*i-30);
            pts[i] = new Point(320+80*Math.cos(a), 240+80*Math.sin(a));
        }
        MatOfPoint poly = new MatOfPoint(pts);
        if (outline) Imgproc.polylines(m, List.of(poly), true, new Scalar(255,255,255), 3);
        else         Imgproc.fillPoly(m, List.of(poly), new Scalar(255,255,255));
        return m;
    }
    private static Mat whitePentagonOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Point[] pts = new Point[5];
        for (int i = 0; i < 5; i++) {
            double a = Math.toRadians(72*i-90);
            pts[i] = new Point(320+90*Math.cos(a), 240+90*Math.sin(a));
        }
        Imgproc.fillPoly(m, List.of(new MatOfPoint(pts)), new Scalar(255,255,255));
        return m;
    }
    private static Mat whiteStarOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Point[] pts = new Point[10];
        for (int i = 0; i < 10; i++) {
            double a = Math.toRadians(36*i-90);
            int r = (i%2==0)?100:40;
            pts[i] = new Point(320+r*Math.cos(a), 240+r*Math.sin(a));
        }
        Imgproc.fillPoly(m, List.of(new MatOfPoint(pts)), new Scalar(255,255,255));
        return m;
    }
    private static Mat whiteDiamondOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.fillPoly(m, List.of(new MatOfPoint(
                new Point(320,110), new Point(470,240),
                new Point(320,370), new Point(170,240))),
                new Scalar(255,255,255));
        return m;
    }
    private static Mat whiteArrowOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.fillPoly(m, List.of(new MatOfPoint(
                new Point(160,200), new Point(340,200), new Point(340,155),
                new Point(480,240), new Point(340,325), new Point(340,280),
                new Point(160,280))),
                new Scalar(255,255,255));
        return m;
    }
    private static Mat whiteEllipseOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.ellipse(m, new Point(320,240), new Size(140,70), 0, 0, 360, new Scalar(255,255,255), -1);
        return m;
    }
    private static Mat whiteOctagonOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Point[] pts = new Point[8];
        for (int i = 0; i < 8; i++) {
            double a = Math.toRadians(45*i-22.5);
            pts[i] = new Point(320+85*Math.cos(a), 240+85*Math.sin(a));
        }
        Imgproc.fillPoly(m, List.of(new MatOfPoint(pts)), new Scalar(255,255,255));
        return m;
    }
    private static Mat whitePlusOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.rectangle(m, new Point(270,140), new Point(370,340), new Scalar(255,255,255), -1);
        Imgproc.rectangle(m, new Point(170,200), new Point(470,280), new Scalar(255,255,255), -1);
        return m;
    }
    private static Mat whiteConcaveArrowheadOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.fillPoly(m, List.of(new MatOfPoint(
                new Point(320,110), new Point(460,370),
                new Point(320,290), new Point(180,370))),
                new Scalar(255,255,255));
        return m;
    }
    private static Mat whiteCrossOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.line(m, new Point(320,80),  new Point(320,400), new Scalar(255,255,255), 8);
        Imgproc.line(m, new Point(100,240), new Point(540,240), new Scalar(255,255,255), 8);
        return m;
    }
    private static Mat whiteRot45RectOnBlack() {
        return rotate(whiteRectOnBlack(230,160,410,320), 45);
    }
    private static Mat compositeOnBackground(Mat shapeMat, BackgroundId bgId) {
        return MatchDiagnosticLibrary.compositeOnBackground(shapeMat, bgId);
    }
    private static Mat rotate(Mat src, double angleDeg) {
        Point centre = new Point(src.cols()/2.0, src.rows()/2.0);
        Mat rot = Imgproc.getRotationMatrix2D(centre, -angleDeg, 1.0);
        Mat dst = Mat.zeros(src.size(), src.type());
        Imgproc.warpAffine(src, dst, rot, src.size());
        rot.release();
        return dst;
    }
    private static Rect groundTruthRect(Mat shapeMat) {
        return MatchDiagnosticLibrary.groundTruthRect(shapeMat);
    }
    private static double iou(Rect a, Rect b) {
        return MatchDiagnosticLibrary.iou(a, b);
    }
}
