package org.example.vectormatcher;

import org.example.OpenCvLoader;
import org.example.factories.ReferenceId;
import org.example.factories.ReferenceImageFactory;
import org.example.matchers.VectorMatcher;
import org.example.matchers.VectorSignature;
import org.example.matchers.VectorVariant;
import org.junit.jupiter.api.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Step 4b — Reference Signature Building.
 *
 * <p><b>Part A</b>: verifies that reference signatures for shapes that definitely exist
 * are built correctly (non-UNKNOWN, plausible type, non-zero fields).
 *
 * <p><b>Part B</b>: verifies signature building against degenerate/negative inputs
 * (empty mat, all-black mat, etc.) and confirms graceful handling.
 */
@DisplayName("Vector Step 4b — Reference Signature Building")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class VectorStep4bRefSignatureTest {

    @BeforeAll
    static void load() { OpenCvLoader.load(); }

    // ========================================================
    // PART A — Known shapes produce valid signatures
    // ========================================================

    @Nested
    @DisplayName("Part A — Known reference shapes")
    @TestMethodOrder(MethodOrderer.OrderAnnotation.class)
    class PartA {

        @Test @Order(1)
        @DisplayName("A1 — CIRCLE_OUTLINE signature is CIRCLE type")
        void circleOutlineSignatureIsCircle() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.CIRCLE_OUTLINE);
            VectorSignature sig = VectorMatcher.buildRefSignature(
                    ref, VectorVariant.VECTOR_NORMAL.epsilonFactor());
            ref.release();
            System.out.println("[4b-A1] " + sig);
            assertEquals(VectorSignature.ShapeType.CIRCLE, sig.type,
                    "CIRCLE_OUTLINE reference should produce a CIRCLE signature");
        }

        @Test @Order(2)
        @DisplayName("A2 — CIRCLE_FILLED signature is CIRCLE type")
        void circleFilledSignatureIsCircle() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.CIRCLE_FILLED);
            VectorSignature sig = VectorMatcher.buildRefSignature(
                    ref, VectorVariant.VECTOR_NORMAL.epsilonFactor());
            ref.release();
            System.out.println("[4b-A2] " + sig);
            assertEquals(VectorSignature.ShapeType.CIRCLE, sig.type);
        }

        @Test @Order(3)
        @DisplayName("A3 — RECT_FILLED signature has 4 vertices")
        void rectFilledHasFourVertices() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.RECT_FILLED);
            VectorSignature sig = VectorMatcher.buildRefSignature(
                    ref, VectorVariant.VECTOR_NORMAL.epsilonFactor());
            ref.release();
            System.out.println("[4b-A3] " + sig);
            assertEquals(4, sig.vertexCount,
                    "RECT_FILLED should have 4 vertices, got " + sig.vertexCount);
        }

        @Test @Order(4)
        @DisplayName("A4 — TRIANGLE_FILLED signature has 3 vertices")
        void triangleFilledHasThreeVertices() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.TRIANGLE_FILLED);
            VectorSignature sig = VectorMatcher.buildRefSignature(
                    ref, VectorVariant.VECTOR_NORMAL.epsilonFactor());
            ref.release();
            System.out.println("[4b-A4] " + sig);
            assertEquals(3, sig.vertexCount,
                    "TRIANGLE_FILLED should have 3 vertices, got " + sig.vertexCount);
        }

        @Test @Order(5)
        @DisplayName("A5 — HEXAGON_OUTLINE signature has 5-8 vertices (approx tolerance)")
        void hexagonHasSixish() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.HEXAGON_OUTLINE);
            VectorSignature sig = VectorMatcher.buildRefSignature(
                    ref, VectorVariant.VECTOR_NORMAL.epsilonFactor());
            ref.release();
            System.out.println("[4b-A5] " + sig);
            assertTrue(sig.vertexCount >= 5 && sig.vertexCount <= 8,
                    "Hexagon should have 5-8 vertices after approximation, got " + sig.vertexCount);
        }

        @Test @Order(6)
        @DisplayName("A6 — STAR_5_FILLED has CLOSED_CONCAVE_POLY type (concavity from points)")
        void starIsConcave() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.STAR_5_FILLED);
            VectorSignature sig = VectorMatcher.buildRefSignature(
                    ref, VectorVariant.VECTOR_STRICT.epsilonFactor());
            ref.release();
            System.out.println("[4b-A6] " + sig);
            assertEquals(VectorSignature.ShapeType.CLOSED_CONCAVE_POLY, sig.type,
                    "STAR_5_FILLED should be CLOSED_CONCAVE_POLY");
        }

        @Test @Order(7)
        @DisplayName("A7 — Signature circularity is > 0.80 for CIRCLE_FILLED")
        void circleHighCircularity() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.CIRCLE_FILLED);
            VectorSignature sig = VectorMatcher.buildRefSignature(
                    ref, VectorVariant.VECTOR_NORMAL.epsilonFactor());
            ref.release();
            assertTrue(sig.circularity > 0.80,
                    "CIRCLE_FILLED circularity should be > 0.80, got " + sig.circularity);
        }

        @Test @Order(8)
        @DisplayName("A8 — LINE_H signature is LINE_SEGMENT type")
        void lineHIsLineSegment() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.LINE_H);
            VectorSignature sig = VectorMatcher.buildRefSignature(
                    ref, VectorVariant.VECTOR_NORMAL.epsilonFactor());
            ref.release();
            System.out.println("[4b-A8] " + sig);
            assertEquals(VectorSignature.ShapeType.LINE_SEGMENT, sig.type,
                    "LINE_H should produce a LINE_SEGMENT signature");
        }
    }

    // ========================================================
    // PART B — Degenerate and negative inputs
    // ========================================================

    @Nested
    @DisplayName("Part B — Degenerate/negative inputs")
    @TestMethodOrder(MethodOrderer.OrderAnnotation.class)
    class PartB {

        @Test @Order(1)
        @DisplayName("B1 — All-black reference returns UNKNOWN signature (no contours)")
        void allBlackRefReturnsUnknown() {
            Mat allBlack = Mat.zeros(128, 128, CvType.CV_8UC3);
            VectorSignature sig = VectorMatcher.buildRefSignature(
                    allBlack, VectorVariant.VECTOR_NORMAL.epsilonFactor());
            allBlack.release();
            System.out.println("[4b-B1] " + sig);
            assertEquals(VectorSignature.ShapeType.UNKNOWN, sig.type,
                    "All-black reference should produce UNKNOWN signature");
        }

        @Test @Order(2)
        @DisplayName("B2 — CIRCLE and RECT signatures are not equal in type")
        void circleAndRectSignaturesDiffer() {
            Mat refCircle = ReferenceImageFactory.build(ReferenceId.CIRCLE_FILLED);
            Mat refRect   = ReferenceImageFactory.build(ReferenceId.RECT_FILLED);
            VectorSignature sigCircle = VectorMatcher.buildRefSignature(
                    refCircle, VectorVariant.VECTOR_NORMAL.epsilonFactor());
            VectorSignature sigRect = VectorMatcher.buildRefSignature(
                    refRect, VectorVariant.VECTOR_NORMAL.epsilonFactor());
            refCircle.release(); refRect.release();
            System.out.printf("[4b-B2] circle=%s  rect=%s%n", sigCircle.type, sigRect.type);
            assertNotEquals(sigCircle.type, sigRect.type,
                    "Circle and rect should have different ShapeTypes");
        }

        @Test @Order(3)
        @DisplayName("B3 — STRICT epsilon yields more vertices than LOOSE for hexagon")
        void strictMoreVerticesThanLoose() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.HEXAGON_OUTLINE);
            VectorSignature strict = VectorMatcher.buildRefSignature(
                    ref, VectorVariant.VECTOR_STRICT.epsilonFactor());
            VectorSignature loose = VectorMatcher.buildRefSignature(
                    ref, VectorVariant.VECTOR_LOOSE.epsilonFactor());
            ref.release();
            System.out.printf("[4b-B3] strict vertices=%d  loose vertices=%d%n",
                    strict.vertexCount, loose.vertexCount);
            assertTrue(strict.vertexCount >= loose.vertexCount,
                    "STRICT approximation should yield >= as many vertices as LOOSE");
        }

        @Test @Order(4)
        @DisplayName("B4 — buildRefSignature does not throw on tiny 1x1 black mat")
        void tinyMatDoesNotThrow() {
            Mat tiny = Mat.zeros(1, 1, CvType.CV_8UC3);
            assertDoesNotThrow(() -> {
                VectorSignature sig = VectorMatcher.buildRefSignature(
                        tiny, VectorVariant.VECTOR_NORMAL.epsilonFactor());
                System.out.println("[4b-B4] " + sig);
            });
            tiny.release();
        }

        @Test @Order(5)
        @DisplayName("B5 — Circle self-similarity = 1.0 via buildRefSignature")
        void circleSelfSimilarityViaRefBuilder() {
            Mat ref = ReferenceImageFactory.build(ReferenceId.CIRCLE_FILLED);
            VectorSignature sig = VectorMatcher.buildRefSignature(
                    ref, VectorVariant.VECTOR_NORMAL.epsilonFactor());
            ref.release();
            double sim = sig.similarity(sig);
            System.out.printf("[4b-B5] circle self-sim=%.3f%n", sim);
            assertEquals(1.0, sim, 0.001, "Self-similarity must be 1.0");
        }
    }
}

