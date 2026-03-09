package org.example.vectormatcher;

import org.example.OpenCvLoader;
import org.example.matchers.VectorSignature;
import org.junit.jupiter.api.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Step 4c — Similarity Scoring.
 *
 * <p><b>Part A</b>: verifies the scoring logic on pairs of shapes where the
 * correct shape is always present — confirming high similarity for correct
 * matches and meaningful separation between different shape types.
 *
 * <p><b>Part B</b>: verifies that scoring correctly discriminates against
 * negative/wrong-shape comparisons and handles edge cases gracefully.
 */
@DisplayName("Vector Step 4c — Similarity Scoring")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class VectorStep4cScoringTest {

    @BeforeAll
    static void load() { OpenCvLoader.load(); }

    // ========================================================
    // PART A — Correct matches score high
    // ========================================================

    @Nested
    @DisplayName("Part A — Correct shape pairs score high")
    @TestMethodOrder(MethodOrderer.OrderAnnotation.class)
    class PartA {

        @Test @Order(1)
        @DisplayName("A1 — Circle vs circle: self-similarity = 1.0")
        void circleSelfSim() {
            VectorSignature s = build(filledCircle(), 0.04);
            assertEquals(1.0, s.similarity(s), 0.001);
        }

        @Test @Order(2)
        @DisplayName("A2 — Rectangle vs rectangle: similarity > 0.80")
        void rectSelfSim() {
            VectorSignature s = build(filledRect(), 0.04);
            double sim = s.similarity(s);
            System.out.printf("[4c-A2] rect self-sim=%.3f%n", sim);
            assertTrue(sim > 0.80, "Rect vs itself should be > 0.80, got " + sim);
        }

        @Test @Order(3)
        @DisplayName("A3 — Triangle vs triangle: similarity > 0.80")
        void triangleSelfSim() {
            VectorSignature s = build(filledTriangle(), 0.04);
            double sim = s.similarity(s);
            System.out.printf("[4c-A3] triangle self-sim=%.3f%n", sim);
            assertTrue(sim > 0.80);
        }

        @Test @Order(4)
        @DisplayName("A4 — Hexagon vs hexagon: similarity > 0.75")
        void hexagonSelfSim() {
            VectorSignature s = build(filledHexagon(), 0.04);
            double sim = s.similarity(s);
            System.out.printf("[4c-A4] hexagon self-sim=%.3f%n", sim);
            assertTrue(sim > 0.75);
        }

        @Test @Order(5)
        @DisplayName("A5 — Circle vs scaled circle: similarity > 0.70 (scale invariance)")
        void circleScaleInvariant() {
            Mat small = Mat.zeros(128, 128, CvType.CV_8UC1);
            Mat large = Mat.zeros(128, 128, CvType.CV_8UC1);
            Imgproc.circle(small, new Point(64, 64), 20, new Scalar(255), -1);
            Imgproc.circle(large, new Point(64, 64), 55, new Scalar(255), -1);
            VectorSignature s1 = VectorSignature.build(small, 0.04);
            VectorSignature s2 = VectorSignature.build(large, 0.04);
            small.release(); large.release();
            double sim = s1.similarity(s2);
            System.out.printf("[4c-A5] small-circle vs large-circle sim=%.3f%n", sim);
            assertTrue(sim > 0.70, "Circles at different scales should be similar, got " + sim);
        }

        @Test @Order(6)
        @DisplayName("A6 — Rectangle vs rotated 45° rectangle: similarity > 0.65 (rotation invariance)")
        void rectRotationInvariant() {
            Mat m1 = filledRect();
            Mat m2 = Mat.zeros(128, 128, CvType.CV_8UC1);
            Point[] pts = new Point[4];
            for (int i = 0; i < 4; i++) {
                double a = Math.toRadians(45 + i * 90);
                pts[i] = new Point(64 + 40 * Math.cos(a), 64 + 40 * Math.sin(a));
            }
            Imgproc.fillPoly(m2, java.util.List.of(new MatOfPoint(pts)), new Scalar(255));
            VectorSignature s1 = VectorSignature.build(m1, 0.04);
            VectorSignature s2 = VectorSignature.build(m2, 0.04);
            m1.release(); m2.release();
            double sim = s1.similarity(s2);
            System.out.printf("[4c-A6] rect vs rotated-rect sim=%.3f%n", sim);
            assertTrue(sim > 0.65, "Rect vs rotated rect should be > 0.65, got " + sim);
        }
    }

    // ========================================================
    // PART B — Wrong shape pairs score low / graceful handling
    // ========================================================

    @Nested
    @DisplayName("Part B — Wrong shapes score lower, edge cases handled")
    @TestMethodOrder(MethodOrderer.OrderAnnotation.class)
    class PartB {

        @Test @Order(1)
        @DisplayName("B1 — Circle vs rectangle: similarity lower than circle vs circle")
        void circleVsRectLowerThanSelf() {
            VectorSignature circ = build(filledCircle(), 0.04);
            VectorSignature rect = build(filledRect(), 0.04);
            double selfSim  = circ.similarity(circ);
            double crossSim = circ.similarity(rect);
            System.out.printf("[4c-B1] circle-self=%.3f  circle-rect=%.3f%n", selfSim, crossSim);
            assertTrue(selfSim > crossSim,
                    "Circle self-similarity must be higher than circle-vs-rect");
        }

        @Test @Order(2)
        @DisplayName("B2 — Triangle vs rectangle: similarity < 0.82")
        void triangleVsRectLow() {
            VectorSignature tri  = build(filledTriangle(), 0.04);
            VectorSignature rect = build(filledRect(), 0.04);
            double sim = tri.similarity(rect);
            System.out.printf("[4c-B2] triangle vs rect sim=%.3f%n", sim);
            assertTrue(sim < 0.82,
                    "Triangle vs rect similarity should be < 0.82, got " + sim);
        }

        @Test @Order(3)
        @DisplayName("B3 — Circle vs rectangle similarity < 0.60")
        void circleVsRectLow() {
            VectorSignature circ = build(filledCircle(), 0.04);
            VectorSignature rect = build(filledRect(), 0.04);
            double sim = circ.similarity(rect);
            System.out.printf("[4c-B3] circle vs rect sim=%.3f%n", sim);
            assertTrue(sim < 0.60,
                    "Circle vs rect similarity should be < 0.60, got " + sim);
        }

        @Test @Order(4)
        @DisplayName("B4 — Similarity against null returns 0.0")
        void nullReturnsZero() {
            VectorSignature s = build(filledCircle(), 0.04);
            assertEquals(0.0, s.similarity(null), 0.001,
                    "similarity(null) must return 0.0");
        }

        @Test @Order(5)
        @DisplayName("B5 — UNKNOWN signature vs UNKNOWN: similarity not NaN")
        void unknownVsUnknownNotNaN() {
            Mat empty = new Mat();
            VectorSignature u1 = VectorSignature.build(empty, 0.04);
            VectorSignature u2 = VectorSignature.build(empty, 0.04);
            empty.release();
            double sim = u1.similarity(u2);
            System.out.printf("[4c-B5] unknown vs unknown sim=%.3f%n", sim);
            assertFalse(Double.isNaN(sim), "Similarity must never be NaN");
            assertTrue(sim >= 0.0 && sim <= 1.0, "Must be in [0,1], got " + sim);
        }

        @Test @Order(6)
        @DisplayName("B6 — Similarity is symmetric: sim(A,B) == sim(B,A)")
        void symmetry() {
            VectorSignature circ = build(filledCircle(), 0.04);
            VectorSignature rect = build(filledRect(), 0.04);
            double ab = circ.similarity(rect);
            double ba = rect.similarity(circ);
            System.out.printf("[4c-B6] sim(circle,rect)=%.4f  sim(rect,circle)=%.4f%n", ab, ba);
            assertEquals(ab, ba, 0.0001, "Similarity must be symmetric");
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private static VectorSignature build(Mat binary, double eps) {
        VectorSignature sig = VectorSignature.build(binary, eps);
        binary.release();
        return sig;
    }

    private static Mat filledCircle() {
        Mat m = Mat.zeros(128, 128, CvType.CV_8UC1);
        Imgproc.circle(m, new Point(64, 64), 48, new Scalar(255), -1);
        return m;
    }

    private static Mat filledRect() {
        Mat m = Mat.zeros(128, 128, CvType.CV_8UC1);
        Imgproc.rectangle(m, new Point(20, 20), new Point(108, 108), new Scalar(255), -1);
        return m;
    }

    private static Mat filledTriangle() {
        Mat m = Mat.zeros(128, 128, CvType.CV_8UC1);
        MatOfPoint pts = new MatOfPoint(
                new Point(64, 10), new Point(10, 118), new Point(118, 118));
        Imgproc.fillPoly(m, java.util.List.of(pts), new Scalar(255));
        return m;
    }

    private static Mat filledHexagon() {
        Mat m = Mat.zeros(128, 128, CvType.CV_8UC1);
        Point[] hex = new Point[6];
        for (int i = 0; i < 6; i++) {
            double a = Math.toRadians(60 * i);
            hex[i] = new Point(64 + 50 * Math.cos(a), 64 + 50 * Math.sin(a));
        }
        Imgproc.fillPoly(m, java.util.List.of(new MatOfPoint(hex)), new Scalar(255));
        return m;
    }
}

