package org.example.vectormatcher;

import org.example.OpenCvLoader;
import org.example.factories.ReferenceId;
import org.example.factories.ReferenceImageFactory;
import org.example.matchers.VectorMatcher;
import org.example.matchers.VectorVariant;
import org.junit.jupiter.api.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Step 4a — Contour Extraction.
 *
 * <p><b>Part A</b> (positive scenes only): verifies that contours are correctly found
 * on scenes that definitely contain shapes.
 *
 * <p><b>Part B</b> (positive + negative scenes): confirms that contour extraction
 * behaves appropriately on blank/noise scenes and does not crash.
 */
@DisplayName("Vector Step 4a — Contour Extraction")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class VectorStep4aContourExtractionTest {

    @BeforeAll
    static void load() { OpenCvLoader.load(); }

    // ========================================================
    // PART A — Positive scenes only
    // ========================================================

    @Nested
    @DisplayName("Part A — Positive scenes (shape present)")
    @TestMethodOrder(MethodOrderer.OrderAnnotation.class)
    class PartA {

        @Test @Order(1)
        @DisplayName("A1 — Single circle on black background: at least 1 contour found")
        void singleCircleContourFound() {
            Mat scene = makeSolidCircleScene(255, 255, 255);
            List<MatOfPoint> c = VectorMatcher.extractContours(scene);
            scene.release();
            System.out.printf("[4a-A1] Circle scene contours: %d%n", c.size());
            assertFalse(c.isEmpty(), "Should find at least one contour on circle scene");
        }

        @Test @Order(2)
        @DisplayName("A2 — Rectangle on black background: at least 1 contour found")
        void rectangleContourFound() {
            Mat scene = makeRectScene(255, 255, 255);
            List<MatOfPoint> c = VectorMatcher.extractContours(scene);
            scene.release();
            System.out.printf("[4a-A2] Rect scene contours: %d%n", c.size());
            assertFalse(c.isEmpty(), "Should find at least one contour on rect scene");
        }

        @Test @Order(3)
        @DisplayName("A3 — All contours have area >= 64 px² (noise filtered)")
        void contoursAboveAreaThreshold() {
            Mat scene = makeSolidCircleScene(255, 255, 255);
            List<MatOfPoint> c = VectorMatcher.extractContours(scene);
            scene.release();
            for (MatOfPoint contour : c) {
                double area = Imgproc.contourArea(contour);
                assertTrue(area >= 64,
                        "Contour area should be >= 64 px², got " + area);
            }
        }

        @Test @Order(4)
        @DisplayName("A4 — Triangle scene produces at least one contour")
        void triangleContourFound() {
            Mat scene = makeTriangleScene();
            List<MatOfPoint> c = VectorMatcher.extractContours(scene);
            scene.release();
            System.out.printf("[4a-A4] Triangle scene contours: %d%n", c.size());
            assertFalse(c.isEmpty());
        }

        @Test @Order(5)
        @DisplayName("A5 — Real reference image (CIRCLE_OUTLINE) produces contours after binarise")
        void refImageProducesContours() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.CIRCLE_OUTLINE);
            List<MatOfPoint> c = VectorMatcher.extractContours(ref);
            ref.release();
            System.out.printf("[4a-A5] CIRCLE_OUTLINE ref contours: %d%n", c.size());
            assertFalse(c.isEmpty(), "Reference image should yield contours");
        }
    }

    // ========================================================
    // PART B — Positive + negative scenes
    // ========================================================

    @Nested
    @DisplayName("Part B — Positive and negative scenes")
    @TestMethodOrder(MethodOrderer.OrderAnnotation.class)
    class PartB {

        @Test @Order(1)
        @DisplayName("B1 — Blank black scene returns zero contours (nothing to detect)")
        void blankSceneZeroContours() {
            Mat scene = Mat.zeros(480, 640, CvType.CV_8UC3);
            List<MatOfPoint> c = VectorMatcher.extractContours(scene);
            scene.release();
            System.out.printf("[4a-B1] Blank scene contours: %d%n", c.size());
            assertEquals(0, c.size(), "Blank scene should produce 0 significant contours");
        }

        @Test @Order(2)
        @DisplayName("B2 — Circle scene produces more contours than blank scene")
        void circleMoreThanBlank() {
            Mat circle = makeSolidCircleScene(255, 255, 255);
            Mat blank  = Mat.zeros(480, 640, CvType.CV_8UC3);
            int circCount = VectorMatcher.extractContours(circle).size();
            int blankCount = VectorMatcher.extractContours(blank).size();
            circle.release(); blank.release();
            System.out.printf("[4a-B2] circle=%d  blank=%d%n", circCount, blankCount);
            assertTrue(circCount > blankCount,
                    "Circle scene should have more contours than blank scene");
        }

        @Test @Order(3)
        @DisplayName("B3 — Does not throw on pure noise scene")
        void noThrowOnNoiseScene() {
            Mat noise = new Mat(480, 640, CvType.CV_8UC3);
            Core.randu(noise, 0.0, 30.0);
            assertDoesNotThrow(() -> {
                List<MatOfPoint> c = VectorMatcher.extractContours(noise);
                System.out.printf("[4a-B3] Noise scene contours: %d%n", c.size());
            });
            noise.release();
        }

        @Test @Order(4)
        @DisplayName("B4 — Bright single pixel does not survive area filter")
        void singlePixelFiltered() {
            Mat scene = Mat.zeros(480, 640, CvType.CV_8UC3);
            scene.put(240, 320, new byte[]{(byte)255, (byte)255, (byte)255});
            List<MatOfPoint> c = VectorMatcher.extractContours(scene);
            scene.release();
            // Single pixel area = 1 px² which is below the 64 px² threshold
            for (MatOfPoint contour : c) {
                assertTrue(Imgproc.contourArea(contour) >= 64,
                        "Single-pixel contours must be filtered out");
            }
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private static Mat makeSolidCircleScene(int r, int g, int b) {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.circle(m, new Point(320, 240), 60, new Scalar(b, g, r), -1);
        return m;
    }

    private static Mat makeRectScene(int r, int g, int b) {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.rectangle(m, new Point(230, 160), new Point(410, 320),
                new Scalar(b, g, r), -1);
        return m;
    }

    private static Mat makeTriangleScene() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        MatOfPoint pts = new MatOfPoint(
                new Point(320, 130), new Point(180, 350), new Point(460, 350));
        Imgproc.fillPoly(m, List.of(pts), new Scalar(255, 255, 255));
        return m;
    }
}


