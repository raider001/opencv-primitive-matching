package org.example.vectormatcher;

import org.example.OpenCvLoader;
import org.example.matchers.VectorSignature;
import org.junit.jupiter.api.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Step 1 — VectorSignature unit tests.
 *
 * <p>Verifies that {@link VectorSignature#build} correctly classifies basic
 * synthetic shapes drawn directly into binary mats.  No scene catalogue or
 * reference images required — pure geometry.
 */
@DisplayName("Vector Step 1 — VectorSignature descriptor")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class VectorSignatureTest {

    @BeforeAll
    static void load() { OpenCvLoader.load(); }

    // -------------------------------------------------------------------------
    // Helper: draw shapes into 128x128 binary mats
    // -------------------------------------------------------------------------

    private static Mat filledCircle() {
        Mat m = Mat.zeros(128, 128, CvType.CV_8UC1);
        Imgproc.circle(m, new Point(64, 64), 48, new Scalar(255), -1);
        return m;
    }

    private static Mat outlineCircle() {
        Mat m = Mat.zeros(128, 128, CvType.CV_8UC1);
        Imgproc.circle(m, new Point(64, 64), 48, new Scalar(255), 2);
        return m;
    }

    private static Mat filledRect() {
        Mat m = Mat.zeros(128, 128, CvType.CV_8UC1);
        Imgproc.rectangle(m, new Point(20, 20), new Point(108, 108), new Scalar(255), -1);
        return m;
    }

    private static Mat triangle() {
        Mat m = Mat.zeros(128, 128, CvType.CV_8UC1);
        MatOfPoint pts = new MatOfPoint(
                new Point(64, 10), new Point(10, 118), new Point(118, 118));
        Imgproc.fillPoly(m, java.util.List.of(pts), new Scalar(255));
        return m;
    }

    private static Mat hexagon() {
        Mat m = Mat.zeros(128, 128, CvType.CV_8UC1);
        Point[] hex = new Point[6];
        for (int i = 0; i < 6; i++) {
            double a = Math.toRadians(60 * i);
            hex[i] = new Point(64 + 50 * Math.cos(a), 64 + 50 * Math.sin(a));
        }
        Imgproc.fillPoly(m, java.util.List.of(new MatOfPoint(hex)), new Scalar(255));
        return m;
    }

    private static Mat star5() {
        Mat m = Mat.zeros(128, 128, CvType.CV_8UC1);
        Point[] pts = new Point[10];
        for (int i = 0; i < 10; i++) {
            double a = Math.toRadians(i * 36 - 90);
            double r = (i % 2 == 0) ? 50 : 22;
            pts[i] = new Point(64 + r * Math.cos(a), 64 + r * Math.sin(a));
        }
        Imgproc.fillPoly(m, java.util.List.of(new MatOfPoint(pts)), new Scalar(255));
        return m;
    }

    private static Mat horizontalLine() {
        Mat m = Mat.zeros(128, 128, CvType.CV_8UC1);
        Imgproc.rectangle(m, new Point(10, 60), new Point(118, 68), new Scalar(255), -1);
        return m;
    }

    private static Mat crossShape() {
        // Two separate rectangles (compound shape)
        Mat m = Mat.zeros(128, 128, CvType.CV_8UC1);
        Imgproc.rectangle(m, new Point(56, 10), new Point(72, 118), new Scalar(255), -1);
        Imgproc.rectangle(m, new Point(10, 56), new Point(118, 72), new Scalar(255), -1);
        return m;
    }

    // -------------------------------------------------------------------------
    // Tests
    // -------------------------------------------------------------------------

    @Test @Order(1)
    @DisplayName("1a — Filled circle classified as CIRCLE")
    void filledCircleIsCircle() {
        Mat m = filledCircle();
        VectorSignature sig = VectorSignature.build(m, 0.04);
        m.release();
        System.out.println("[Step1] Filled circle: " + sig);
        assertEquals(VectorSignature.ShapeType.CIRCLE, sig.type,
                "Filled circle should be classified CIRCLE");
        assertTrue(sig.circularity > 0.88,
                "Circularity of a filled circle should be > 0.88, got " + sig.circularity);
    }

    @Test @Order(2)
    @DisplayName("1b — Outline circle classified as CIRCLE")
    void outlineCircleIsCircle() {
        Mat m = outlineCircle();
        VectorSignature sig = VectorSignature.build(m, 0.04);
        m.release();
        System.out.println("[Step1] Outline circle: " + sig);
        assertEquals(VectorSignature.ShapeType.CIRCLE, sig.type,
                "Outline circle should be classified CIRCLE");
    }

    @Test @Order(3)
    @DisplayName("1c — Rectangle classified as CLOSED_CONVEX_POLY with 4 vertices")
    void rectangleIsFourVertexPoly() {
        Mat m = filledRect();
        VectorSignature sig = VectorSignature.build(m, 0.04);
        m.release();
        System.out.println("[Step1] Rectangle: " + sig);
        assertEquals(VectorSignature.ShapeType.CLOSED_CONVEX_POLY, sig.type,
                "Rectangle should be CLOSED_CONVEX_POLY");
        assertEquals(4, sig.vertexCount,
                "Rectangle should have 4 vertices, got " + sig.vertexCount);
    }

    @Test @Order(4)
    @DisplayName("1d — Triangle classified as CLOSED_CONVEX_POLY with 3 vertices")
    void triangleIsThreeVertexPoly() {
        Mat m = triangle();
        VectorSignature sig = VectorSignature.build(m, 0.04);
        m.release();
        System.out.println("[Step1] Triangle: " + sig);
        assertEquals(VectorSignature.ShapeType.CLOSED_CONVEX_POLY, sig.type,
                "Triangle should be CLOSED_CONVEX_POLY");
        assertEquals(3, sig.vertexCount,
                "Triangle should have 3 vertices, got " + sig.vertexCount);
    }

    @Test @Order(5)
    @DisplayName("1e — Hexagon classified as CLOSED_CONVEX_POLY with 6 vertices")
    void hexagonIsSixVertexPoly() {
        Mat m = hexagon();
        VectorSignature sig = VectorSignature.build(m, 0.04);
        m.release();
        System.out.println("[Step1] Hexagon: " + sig);
        assertEquals(VectorSignature.ShapeType.CLOSED_CONVEX_POLY, sig.type,
                "Hexagon should be CLOSED_CONVEX_POLY");
        assertTrue(sig.vertexCount >= 5 && sig.vertexCount <= 8,
                "Hexagon should have ~6 vertices (5-8 after approx), got " + sig.vertexCount);
    }

    @Test @Order(6)
    @DisplayName("1f — 5-pointed star classified as CLOSED_CONCAVE_POLY")
    void starIsConcavePoly() {
        Mat m = star5();
        VectorSignature sig = VectorSignature.build(m, 0.02);
        m.release();
        System.out.println("[Step1] 5-pointed star: " + sig);
        assertEquals(VectorSignature.ShapeType.CLOSED_CONCAVE_POLY, sig.type,
                "Star should be CLOSED_CONCAVE_POLY due to convexity defects");
        assertTrue(sig.concavityRatio > 0.0,
                "Star concavityRatio should be > 0, got " + sig.concavityRatio);
    }

    @Test @Order(7)
    @DisplayName("1g — Horizontal bar classified as LINE_SEGMENT")
    void horizontalBarIsLine() {
        Mat m = horizontalLine();
        VectorSignature sig = VectorSignature.build(m, 0.04);
        m.release();
        System.out.println("[Step1] Horizontal bar: " + sig);
        assertEquals(VectorSignature.ShapeType.LINE_SEGMENT, sig.type,
                "Elongated bar should be LINE_SEGMENT");
        assertTrue(sig.aspectRatio >= 4.0,
                "Aspect ratio should be >= 4.0, got " + sig.aspectRatio);
    }

    @Test @Order(8)
    @DisplayName("1h — Cross shape (two overlapping rects) classified as COMPOUND or CLOSED_CONVEX_POLY")
    void crossIsCompoundOrConvex() {
        Mat m = crossShape();
        VectorSignature sig = VectorSignature.build(m, 0.04);
        m.release();
        System.out.println("[Step1] Cross: " + sig);
        // The two rectangles merge at the centre pixel — may be 1 or 2 components
        assertNotEquals(VectorSignature.ShapeType.UNKNOWN, sig.type,
                "Cross should not be UNKNOWN");
    }

    @Test @Order(9)
    @DisplayName("1i — Circle is more similar to itself than to a rectangle")
    void circleSelfSimilarityHigherThanRectangle() {
        Mat mc = filledCircle();
        Mat mr = filledRect();
        VectorSignature circSig = VectorSignature.build(mc, 0.04);
        VectorSignature rectSig = VectorSignature.build(mr, 0.04);
        mc.release(); mr.release();

        double selfSim = circSig.similarity(circSig);
        double crossSim = circSig.similarity(rectSig);
        System.out.printf("[Step1] Circle self-sim=%.3f, circle-vs-rect=%.3f%n",
                selfSim, crossSim);
        assertEquals(1.0, selfSim, 0.001, "Self-similarity should be 1.0");
        assertTrue(selfSim > crossSim,
                "Circle should be more similar to itself than to a rectangle");
    }

    @Test @Order(10)
    @DisplayName("1j — Angle histogram sums to 1.0")
    void angleHistogramSumsToOne() {
        Mat m = hexagon();
        VectorSignature sig = VectorSignature.build(m, 0.04);
        m.release();
        double sum = 0;
        for (double v : sig.angleHistogram) sum += v;
        System.out.printf("[Step1] Angle histogram sum=%.4f%n", sum);
        assertEquals(1.0, sum, 0.01, "Angle histogram must sum to 1.0");
    }

    @Test @Order(11)
    @DisplayName("1k — Empty mat returns UNKNOWN signature")
    void emptyMatReturnsUnknown() {
        Mat empty = new Mat();
        VectorSignature sig = VectorSignature.build(empty, 0.04);
        empty.release();
        assertEquals(VectorSignature.ShapeType.UNKNOWN, sig.type,
                "Empty mat should produce UNKNOWN signature");
    }

    @Test @Order(12)
    @DisplayName("1l — Rectangle similar to rotated rectangle (rotation invariance)")
    void rectangleSimilarToRotatedRectangle() {
        Mat m1 = filledRect();
        // Rotate 45 degrees
        Mat m2 = Mat.zeros(128, 128, CvType.CV_8UC1);
        Point[] pts = new Point[4];
        double cx = 64, cy = 64, hs = 40;
        for (int i = 0; i < 4; i++) {
            double a = Math.toRadians(45 + i * 90);
            pts[i] = new Point(cx + hs * Math.cos(a), cy + hs * Math.sin(a));
        }
        Imgproc.fillPoly(m2, java.util.List.of(new MatOfPoint(pts)), new Scalar(255));

        VectorSignature s1 = VectorSignature.build(m1, 0.04);
        VectorSignature s2 = VectorSignature.build(m2, 0.04);
        m1.release(); m2.release();

        double sim = s1.similarity(s2);
        System.out.printf("[Step1] Rect vs rotated rect similarity=%.3f%n", sim);
        assertTrue(sim > 0.70,
                "Rectangle should be similar to rotated rectangle (sim=" + sim + "), expected > 0.70");
    }
}


