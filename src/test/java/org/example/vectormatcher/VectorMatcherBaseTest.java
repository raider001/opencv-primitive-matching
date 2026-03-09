package org.example.vectormatcher;

import org.example.MatcherVariant;
import org.example.analytics.AnalysisResult;
import org.example.OpenCvLoader;
import org.example.factories.BackgroundId;
import org.example.factories.ReferenceId;
import org.example.factories.ReferenceImageFactory;
import org.example.matchers.VectorMatcher;
import org.example.matchers.VectorSignature;
import org.example.matchers.VectorVariant;
import org.example.scene.SceneCategory;
import org.example.scene.SceneEntry;
import org.junit.jupiter.api.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Step 3 — VectorMatcher base pipeline tests (VECTOR_NORMAL, no CF).
 *
 * <p>Tests the end-to-end {@link VectorMatcher#match} pipeline using synthetic
 * scenes constructed directly in the test — no scene catalogue needed.
 * Only the VECTOR_NORMAL/NONE variant is exercised here; CF variants are
 * tested in Step 4.
 */
@DisplayName("Vector Step 3 — VectorMatcher base pipeline")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class VectorMatcherBaseTest {

    private static final Path OUTPUT = Paths.get("test_output", "vector_matcher_step3");

    @BeforeAll
    static void load() { OpenCvLoader.load(); }

    // -------------------------------------------------------------------------
    // Helpers: build minimal SceneEntry wrappers
    // -------------------------------------------------------------------------

    /** Build a 640x480 scene with a white shape on black background. */
    private static SceneEntry sceneFromMat(Mat bgr, ReferenceId refId) {
        return new SceneEntry(refId, SceneCategory.A_CLEAN, "test_synthetic",
                BackgroundId.BG_SOLID_BLACK, Collections.emptyList(), bgr);
    }

    /** Scene carrying no reference (negative). */
    private static SceneEntry negativeScene(Mat bgr) {
        return new SceneEntry(null, SceneCategory.D_NEGATIVE, "test_negative",
                BackgroundId.BG_SOLID_BLACK, Collections.emptyList(), bgr);
    }

    /** Build a 640x480 black BGR scene with a white filled circle at centre. */
    private static Mat circleScene() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.circle(m, new Point(320, 240), 60, new Scalar(255, 255, 255), -1);
        return m;
    }

    /** Build a 640x480 black BGR scene with a white filled rectangle at centre. */
    private static Mat rectScene() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.rectangle(m, new Point(230, 160), new Point(410, 320),
                new Scalar(255, 255, 255), -1);
        return m;
    }

    /** Build a plain black scene (no shape — should produce low scores). */
    private static Mat blackScene() {
        return Mat.zeros(480, 640, CvType.CV_8UC3);
    }

    // -------------------------------------------------------------------------
    // Tests
    // -------------------------------------------------------------------------

    @Test @Order(1)
    @DisplayName("3a — match() returns exactly 9 results (one per variant)")
    void matchReturnsNineResults() {
        Mat scene = circleScene();
        Mat ref   = ReferenceImageFactory.build(ReferenceId.CIRCLE_OUTLINE);
        List<AnalysisResult> results = VectorMatcher.match(
                ReferenceId.CIRCLE_OUTLINE, ref, sceneFromMat(scene, ReferenceId.CIRCLE_OUTLINE),
                Collections.emptySet(), OUTPUT);
        ref.release(); scene.release();
        assertEquals(9, results.size(), "match() must return exactly 9 results");
    }

    @Test @Order(2)
    @DisplayName("3b — No result is an error")
    void noResultIsError() {
        Mat scene = circleScene();
        Mat ref   = ReferenceImageFactory.build(ReferenceId.CIRCLE_OUTLINE);
        List<AnalysisResult> results = VectorMatcher.match(
                ReferenceId.CIRCLE_OUTLINE, ref, sceneFromMat(scene, ReferenceId.CIRCLE_OUTLINE),
                Collections.emptySet(), OUTPUT);
        ref.release(); scene.release();
        for (AnalysisResult r : results) {
            assertFalse(r.isError(), "Unexpected error in variant " + r.methodName()
                    + ": " + r.errorMessage());
        }
    }

    @Test @Order(3)
    @DisplayName("3c — Scores are in range [0, 100]")
    void scoresInRange() {
        Mat scene = circleScene();
        Mat ref   = ReferenceImageFactory.build(ReferenceId.CIRCLE_OUTLINE);
        List<AnalysisResult> results = VectorMatcher.match(
                ReferenceId.CIRCLE_OUTLINE, ref, sceneFromMat(scene, ReferenceId.CIRCLE_OUTLINE),
                Collections.emptySet(), OUTPUT);
        ref.release(); scene.release();
        for (AnalysisResult r : results) {
            assertTrue(r.matchScorePercent() >= 0 && r.matchScorePercent() <= 100,
                    "Score out of range for " + r.methodName() + ": " + r.matchScorePercent());
        }
    }

    @Test @Order(4)
    @DisplayName("3d — Circle reference scores higher on circle scene than on blank scene")
    void circleSceneScoresHigherThanBlank() {
        Mat circScene = circleScene();
        Mat blankScene = blackScene();
        Mat ref = ReferenceImageFactory.build(ReferenceId.CIRCLE_OUTLINE);

        List<AnalysisResult> onCircle = VectorMatcher.match(
                ReferenceId.CIRCLE_OUTLINE, ref,
                sceneFromMat(circScene, ReferenceId.CIRCLE_OUTLINE),
                Collections.emptySet(), OUTPUT);
        List<AnalysisResult> onBlank = VectorMatcher.match(
                ReferenceId.CIRCLE_OUTLINE, ref,
                negativeScene(blankScene),
                Collections.emptySet(), OUTPUT);
        ref.release(); circScene.release(); blankScene.release();

        double circScore  = normalScore(onCircle);
        double blankScore = normalScore(onBlank);
        System.out.printf("[Step3] Circle-on-circle=%.1f  circle-on-blank=%.1f%n",
                circScore, blankScore);
        assertTrue(circScore > blankScore,
                "Circle should score higher on circle scene than blank scene");
    }

    @Test @Order(5)
    @DisplayName("3e — buildRefSignature produces a non-UNKNOWN signature for CIRCLE_OUTLINE")
    void refSignatureIsNotUnknown() {
        Mat ref = ReferenceImageFactory.build(ReferenceId.CIRCLE_OUTLINE);
        VectorSignature sig = VectorMatcher.buildRefSignature(ref,
                VectorVariant.VECTOR_NORMAL.epsilonFactor());
        ref.release();
        System.out.println("[Step3] Ref signature: " + sig);
        assertNotEquals(VectorSignature.ShapeType.UNKNOWN, sig.type,
                "Reference signature must not be UNKNOWN");
    }

    @Test @Order(6)
    @DisplayName("3f — extractContours finds at least one contour on circle scene")
    void extractContoursFindsShapes() {
        Mat scene = circleScene();
        List<MatOfPoint> contours = VectorMatcher.extractContours(scene);
        scene.release();
        System.out.printf("[Step3] Contours found on circle scene: %d%n", contours.size());
        assertFalse(contours.isEmpty(), "Should find at least one contour on circle scene");
    }

    @Test @Order(7)
    @DisplayName("3g — Each result has the correct variant name")
    void resultVariantNamesMatch() {
        Mat scene = circleScene();
        Mat ref   = ReferenceImageFactory.build(ReferenceId.CIRCLE_OUTLINE);
        List<AnalysisResult> results = VectorMatcher.match(
                ReferenceId.CIRCLE_OUTLINE, ref,
                sceneFromMat(scene, ReferenceId.CIRCLE_OUTLINE),
                Collections.emptySet(), OUTPUT);
        ref.release(); scene.release();

        Set<String> expected = org.example.MatcherVariant.allNamesOf(VectorVariant.class);
        for (AnalysisResult r : results) {
            assertTrue(expected.contains(r.methodName()),
                    "Unexpected variant name: " + r.methodName());
        }
    }

    @Test @Order(8)
    @DisplayName("3h — Rectangle reference scores higher on rect scene than on circle scene")
    void rectScoredCorrectly() {
        Mat rScene = rectScene();
        Mat cScene = circleScene();
        Mat ref    = ReferenceImageFactory.build(ReferenceId.RECT_FILLED);

        List<AnalysisResult> onRect   = VectorMatcher.match(ReferenceId.RECT_FILLED, ref,
                sceneFromMat(rScene, ReferenceId.RECT_FILLED), Collections.emptySet(), OUTPUT);
        List<AnalysisResult> onCircle = VectorMatcher.match(ReferenceId.RECT_FILLED, ref,
                sceneFromMat(cScene, ReferenceId.CIRCLE_FILLED), Collections.emptySet(), OUTPUT);
        ref.release(); rScene.release(); cScene.release();

        double rectScore   = normalScore(onRect);
        double circleScore = normalScore(onCircle);
        System.out.printf("[Step3] Rect ref: on-rect=%.1f  on-circle=%.1f%n",
                rectScore, circleScore);
        assertTrue(rectScore > circleScore,
                "Rect reference should score higher on rect scene than circle scene");
    }

    // -------------------------------------------------------------------------
    // Helper
    // -------------------------------------------------------------------------

    /** Extract the VECTOR_NORMAL score from a result list. */
    private static double normalScore(List<AnalysisResult> results) {
        return results.stream()
                .filter(r -> r.methodName().equals(VectorVariant.VECTOR_NORMAL.variantName()))
                .findFirst()
                .map(AnalysisResult::matchScorePercent)
                .orElse(0.0);
    }
}


