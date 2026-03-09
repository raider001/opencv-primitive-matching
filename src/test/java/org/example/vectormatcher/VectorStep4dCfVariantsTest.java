package org.example.vectormatcher;

import org.example.OpenCvLoader;
import org.example.analytics.AnalysisResult;
import org.example.factories.BackgroundId;
import org.example.factories.ReferenceId;
import org.example.factories.ReferenceImageFactory;
import org.example.matchers.VectorMatcher;
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

import static org.junit.jupiter.api.Assertions.*;

/**
 * Step 4d — End-to-end matching with CF (colour-filter) variants.
 *
 * <p><b>Part A</b>: verifies that CF variants produce valid results on scenes
 * that contain the correct shape in the correct colour.
 *
 * <p><b>Part B</b>: verifies that CF variants behave correctly on negative
 * (wrong colour or no shape) scenes and that all 9 variants produce consistent
 * structural results.
 */
@DisplayName("Vector Step 4d — End-to-end with CF variants")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class VectorStep4dCfVariantsTest {

    private static final Path OUTPUT = Paths.get("test_output", "vector_matcher_step4d");

    @BeforeAll
    static void load() { OpenCvLoader.load(); }

    // ========================================================
    // PART A — Shape present, colour matches (positive)
    // ========================================================

    @Nested
    @DisplayName("Part A — Correct shape and colour present")
    @TestMethodOrder(MethodOrderer.OrderAnnotation.class)
    class PartA {

        @Test @Order(1)
        @DisplayName("A1 — All 9 variants return without error on a circle scene")
        void allVariantsNoError() {
            Mat ref   = ReferenceImageFactory.build(ReferenceId.CIRCLE_FILLED);
            Mat scene = buildColourCircleScene(ref);  // matches reference colour
            List<AnalysisResult> results = VectorMatcher.match(
                    ReferenceId.CIRCLE_FILLED, ref,
                    wrapScene(scene, ReferenceId.CIRCLE_FILLED),
                    Collections.emptySet(), OUTPUT);
            ref.release(); scene.release();
            assertEquals(9, results.size());
            for (AnalysisResult r : results) {
                assertFalse(r.isError(),
                        "Variant " + r.methodName() + " returned an error: " + r.errorMessage());
            }
        }

        @Test @Order(2)
        @DisplayName("A2 — All 9 variant names are distinct in results")
        void allVariantNamesDistinct() {
            Mat ref   = ReferenceImageFactory.build(ReferenceId.RECT_FILLED);
            Mat scene = buildColourRectScene(ref);
            List<AnalysisResult> results = VectorMatcher.match(
                    ReferenceId.RECT_FILLED, ref,
                    wrapScene(scene, ReferenceId.RECT_FILLED),
                    Collections.emptySet(), OUTPUT);
            ref.release(); scene.release();
            long distinct = results.stream().map(AnalysisResult::methodName).distinct().count();
            assertEquals(9, distinct, "All 9 results must have distinct variant names");
        }

        @Test @Order(3)
        @DisplayName("A3 — STRICT variant scores within 30 points of NORMAL on clean scene")
        void strictWithinRangeOfNormal() {
            Mat ref   = ReferenceImageFactory.build(ReferenceId.CIRCLE_FILLED);
            Mat scene = buildColourCircleScene(ref);
            List<AnalysisResult> results = VectorMatcher.match(
                    ReferenceId.CIRCLE_FILLED, ref,
                    wrapScene(scene, ReferenceId.CIRCLE_FILLED),
                    Collections.emptySet(), OUTPUT);
            ref.release(); scene.release();
            double strict = scoreOf(results, VectorVariant.VECTOR_STRICT);
            double normal = scoreOf(results, VectorVariant.VECTOR_NORMAL);
            System.out.printf("[4d-A3] STRICT=%.1f  NORMAL=%.1f%n", strict, normal);
            assertTrue(Math.abs(strict - normal) <= 30,
                    "STRICT and NORMAL should score within 30 points on clean scene");
        }

        @Test @Order(4)
        @DisplayName("A4 — CF_LOOSE variant score is non-zero on colour-matched scene")
        void cfLooseNonZeroOnMatch() {
            Mat ref   = ReferenceImageFactory.build(ReferenceId.RECT_FILLED);
            Mat scene = buildColourRectScene(ref);
            List<AnalysisResult> results = VectorMatcher.match(
                    ReferenceId.RECT_FILLED, ref,
                    wrapScene(scene, ReferenceId.RECT_FILLED),
                    Collections.emptySet(), OUTPUT);
            ref.release(); scene.release();
            double cfLoose = scoreOf(results, VectorVariant.VECTOR_NORMAL_CF_LOOSE);
            System.out.printf("[4d-A4] NORMAL_CF_LOOSE=%.1f%n", cfLoose);
            assertTrue(cfLoose > 0, "CF_LOOSE should not be zero on colour-matched scene");
        }

        @Test @Order(5)
        @DisplayName("A5 — All scores are in [0, 100] range")
        void allScoresInRange() {
            Mat ref   = ReferenceImageFactory.build(ReferenceId.HEXAGON_FILLED);
            Mat scene = buildColourCircleScene(ref);   // intentionally wrong shape
            List<AnalysisResult> results = VectorMatcher.match(
                    ReferenceId.HEXAGON_FILLED, ref,
                    wrapScene(scene, ReferenceId.HEXAGON_FILLED),
                    Collections.emptySet(), OUTPUT);
            ref.release(); scene.release();
            for (AnalysisResult r : results) {
                assertTrue(r.matchScorePercent() >= 0 && r.matchScorePercent() <= 100,
                        "Score out of [0,100] for " + r.methodName()
                                + ": " + r.matchScorePercent());
            }
        }
    }

    // ========================================================
    // PART B — Negative scenes (wrong colour or no shape)
    // ========================================================

    @Nested
    @DisplayName("Part B — Negative / wrong scenes")
    @TestMethodOrder(MethodOrderer.OrderAnnotation.class)
    class PartB {

        @Test @Order(1)
        @DisplayName("B1 — CF_TIGHT on wrong-colour scene: does not throw, score in [0,100]")
        void cfTightWrongColourNoThrow() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.CIRCLE_FILLED);
            // Build scene with deliberately wrong colour (blue circle, reference is typically white)
            Mat scene = Mat.zeros(480, 640, CvType.CV_8UC3);
            Imgproc.circle(scene, new Point(320, 240), 60, new Scalar(200, 10, 10), -1);

            assertDoesNotThrow(() -> {
                List<AnalysisResult> results = VectorMatcher.match(
                        ReferenceId.CIRCLE_FILLED, ref,
                        wrapScene(scene, ReferenceId.CIRCLE_FILLED),
                        Collections.emptySet(), OUTPUT);
                for (AnalysisResult r : results) {
                    assertTrue(r.matchScorePercent() >= 0 && r.matchScorePercent() <= 100);
                }
            });
            ref.release(); scene.release();
        }

        @Test @Order(2)
        @DisplayName("B2 — All variants return results on blank (negative) scene")
        void allVariantsReturnOnBlankScene() {
            Mat ref   = ReferenceImageFactory.build(ReferenceId.CIRCLE_FILLED);
            Mat scene = Mat.zeros(480, 640, CvType.CV_8UC3);
            List<AnalysisResult> results = VectorMatcher.match(
                    ReferenceId.CIRCLE_FILLED, ref,
                    new SceneEntry(null, SceneCategory.D_NEGATIVE, "negative_blank",
                            BackgroundId.BG_SOLID_BLACK, Collections.emptyList(), scene),
                    Collections.emptySet(), OUTPUT);
            ref.release(); scene.release();
            assertEquals(9, results.size(), "Must return 9 results even on blank negative scene");
            for (AnalysisResult r : results) {
                assertFalse(r.isError(), r.methodName() + " errored on blank scene");
            }
        }

        @Test @Order(3)
        @DisplayName("B3 — CIRCLE_FILLED reference scores lower on triangle scene than on circle scene")
        void circleScoresLowerOnTriangle() {
            Mat ref         = ReferenceImageFactory.build(ReferenceId.CIRCLE_FILLED);
            Mat circScene   = Mat.zeros(480, 640, CvType.CV_8UC3);
            Mat triScene    = Mat.zeros(480, 640, CvType.CV_8UC3);
            Imgproc.circle(circScene, new Point(320, 240), 60,
                    new Scalar(255, 255, 255), -1);
            MatOfPoint pts = new MatOfPoint(
                    new Point(320, 130), new Point(180, 350), new Point(460, 350));
            Imgproc.fillPoly(triScene, List.of(pts), new Scalar(255, 255, 255));

            List<AnalysisResult> onCircle = VectorMatcher.match(ReferenceId.CIRCLE_FILLED, ref,
                    wrapScene(circScene, ReferenceId.CIRCLE_FILLED),
                    Collections.emptySet(), OUTPUT);
            List<AnalysisResult> onTri = VectorMatcher.match(ReferenceId.CIRCLE_FILLED, ref,
                    wrapScene(triScene, ReferenceId.TRIANGLE_FILLED),
                    Collections.emptySet(), OUTPUT);
            ref.release(); circScene.release(); triScene.release();

            double circScore = scoreOf(onCircle, VectorVariant.VECTOR_NORMAL);
            double triScore  = scoreOf(onTri,    VectorVariant.VECTOR_NORMAL);
            System.out.printf("[4d-B3] circle-on-circle=%.1f  circle-on-tri=%.1f%n",
                    circScore, triScore);
            assertTrue(circScore > triScore,
                    "Circle ref should score higher on circle scene than triangle scene");
        }

        @Test @Order(4)
        @DisplayName("B4 — No variant throws on a heavily noisy negative scene")
        void noThrowOnNoisyNegativeScene() {
            Mat ref   = ReferenceImageFactory.build(ReferenceId.RECT_FILLED);
            Mat noise = new Mat(480, 640, CvType.CV_8UC3);
            Core.randu(noise, 0.0, 255.0);
            assertDoesNotThrow(() -> {
                List<AnalysisResult> results = VectorMatcher.match(
                        ReferenceId.RECT_FILLED, ref,
                        new SceneEntry(null, SceneCategory.D_NEGATIVE, "noisy",
                                BackgroundId.BG_COLOURED_NOISE, Collections.emptyList(), noise),
                        Collections.emptySet(), OUTPUT);
                assertEquals(9, results.size());
            });
            ref.release(); noise.release();
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /**
     * Build a 640x480 scene containing a circle drawn in the same colour as
     * the reference foreground (extracted from the centre of the ref image).
     */
    private static Mat buildColourCircleScene(Mat ref) {
        double[] px  = ref.get(64, 64);
        Scalar colour = px != null ? new Scalar(px[0], px[1], px[2])
                                   : new Scalar(255, 255, 255);
        Mat scene = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.circle(scene, new Point(320, 240), 60, colour, -1);
        return scene;
    }

    private static Mat buildColourRectScene(Mat ref) {
        double[] px  = ref.get(64, 64);
        Scalar colour = px != null ? new Scalar(px[0], px[1], px[2])
                                   : new Scalar(255, 255, 255);
        Mat scene = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.rectangle(scene, new Point(230, 160), new Point(410, 320), colour, -1);
        return scene;
    }

    private static SceneEntry wrapScene(Mat bgr, ReferenceId refId) {
        return new SceneEntry(refId, SceneCategory.A_CLEAN, "test",
                BackgroundId.BG_SOLID_BLACK, Collections.emptyList(), bgr);
    }

    private static double scoreOf(List<AnalysisResult> results, VectorVariant v) {
        return results.stream()
                .filter(r -> r.methodName().equals(v.variantName()))
                .findFirst()
                .map(AnalysisResult::matchScorePercent)
                .orElse(0.0);
    }
}


