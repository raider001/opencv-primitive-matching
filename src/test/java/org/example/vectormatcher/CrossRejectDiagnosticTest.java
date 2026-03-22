package org.example.vectormatcher;

import org.example.OpenCvLoader;
import org.example.analytics.AnalysisResult;
import org.example.factories.BackgroundId;
import org.example.factories.ReferenceId;
import org.example.factories.ReferenceImageFactory;
import org.example.matchers.vectormatcher.VectorMatcher;
import org.example.scene.SceneCategory;
import org.example.scene.SceneEntry;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;

/**
 * Focused diagnostic for cross-rejection test failures.
 * Measures current scores for shape pairs that should NOT match.
 */
public class CrossRejectDiagnosticTest {

    private static final Path OUTPUT = Paths.get("test_output", "cross_reject_diag");

    @BeforeAll
    static void setup() throws IOException {
        OpenCvLoader.load();
        Files.createDirectories(OUTPUT);
    }

    @Test
    void diagnosticDiamondVsRotatedRect() {
        testPair(ReferenceId.POLYLINE_DIAMOND, ReferenceId.RECT_ROTATED_45, "DIAMOND→RECT_ROTATED_45");
    }

    @Test
    void diagnosticHexagonVsOctagon() {
        testPair(ReferenceId.HEXAGON_OUTLINE, ReferenceId.OCTAGON_FILLED, "HEXAGON→OCTAGON");
    }

    @Test
    void diagnosticTriangleVsPentagon() {
        testPair(ReferenceId.TRIANGLE_FILLED, ReferenceId.PENTAGON_FILLED, "TRIANGLE→PENTAGON");
    }

    @Test
    void diagnosticEllipseVsArrow() {
        testPair(ReferenceId.ELLIPSE_H, ReferenceId.CONCAVE_ARROW_HEAD, "ELLIPSE→ARROW");
    }

    @Test
    void diagnosticRectVsStar() {
        testPair(ReferenceId.RECT_FILLED, ReferenceId.STAR_5_FILLED, "RECT→STAR");
    }

    @Test
    void diagnosticCrossVsPlus() {
        testPair(ReferenceId.LINE_CROSS, ReferenceId.POLYLINE_PLUS_SHAPE, "CROSS→PLUS");
    }

    @Test
    void diagnosticTriangleVsCross() {
        testPair(ReferenceId.TRIANGLE_FILLED, ReferenceId.LINE_CROSS, "TRIANGLE→CROSS");
    }

    private void testPair(ReferenceId queryRef, ReferenceId sceneRef, String label) {
        // Build scene: 3× scaled sceneRef on black background
        Mat sceneMat = buildCrossRejectScene(sceneRef);
        Mat queryMat = ReferenceImageFactory.build(queryRef);
        
        try {
            SceneEntry scene = new SceneEntry(
                    sceneRef, SceneCategory.A_CLEAN, "cross_diag",
                    BackgroundId.BG_SOLID_BLACK, Collections.emptyList(), sceneMat);
            
            List<AnalysisResult> results = VectorMatcher.match(
                    queryRef, queryMat, scene, Collections.emptySet(), OUTPUT);
            
            double score = results.isEmpty() ? 0.0 : results.get(0).matchScorePercent();
            
            System.out.printf("%-30s  score=%.1f%%  %s%n", 
                label, score, 
                score < 60.0 ? "✓ PASS (rejected)" : "✗ FAIL (should reject)");
            
        } finally {
            queryMat.release();
            sceneMat.release();
        }
    }

    private Mat buildCrossRejectScene(ReferenceId shapeRef) {
        // Create 640×480 black canvas
        Mat canvas = new Mat(480, 640, CvType.CV_8UC3, new Scalar(0, 0, 0));
        
        // Load and scale reference 3×
        Mat refImg = ReferenceImageFactory.build(shapeRef);
        Mat scaled = new Mat();
        Imgproc.resize(refImg, scaled, new org.opencv.core.Size(
            refImg.cols() * 3, refImg.rows() * 3), 0, 0, Imgproc.INTER_NEAREST);
        refImg.release();
        
        // Center it on canvas
        int x = (canvas.cols() - scaled.cols()) / 2;
        int y = (canvas.rows() - scaled.rows()) / 2;
        Mat roi = canvas.submat(y, y + scaled.rows(), x, x + scaled.cols());
        scaled.copyTo(roi);
        
        roi.release();
        scaled.release();
        
        return canvas;
    }
}

