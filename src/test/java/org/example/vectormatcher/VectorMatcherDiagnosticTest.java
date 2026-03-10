package org.example.vectormatcher;
import org.example.OpenCvLoader;
import org.example.analytics.AnalysisResult;
import org.example.colour.SceneColourClusters;
import org.example.factories.BackgroundId;
import org.example.factories.ReferenceId;
import org.example.factories.ReferenceImageFactory;
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
import java.nio.file.*;
import java.util.*;
/**
 * Diagnostic test - evaluates every shape x background and writes:
 *   test_output/vector_matching/diagnostics.json
 *   test_output/vector_matching/report.html
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@DisplayName("VectorMatcher - Diagnostic Suite")
class VectorMatcherDiagnosticTest {
    private static final Path   OUTPUT      = Paths.get("test_output", "vector_matching");
    private static final double PASS_THRESH = 40.0;
    private static final double TARGET      = 75.0;
    private static final double GOOD_IOU    = 0.5;
    private static final ReferenceId[] ALL_SHAPES = {
        ReferenceId.CIRCLE_FILLED, ReferenceId.RECT_FILLED, ReferenceId.TRIANGLE_FILLED,
        ReferenceId.HEXAGON_OUTLINE, ReferenceId.PENTAGON_FILLED, ReferenceId.STAR_5_FILLED,
        ReferenceId.POLYLINE_DIAMOND, ReferenceId.POLYLINE_ARROW_RIGHT, ReferenceId.ELLIPSE_H,
        ReferenceId.OCTAGON_FILLED, ReferenceId.POLYLINE_PLUS_SHAPE,
        ReferenceId.CONCAVE_ARROW_HEAD, ReferenceId.LINE_CROSS, ReferenceId.RECT_ROTATED_45,
    };
    private enum BgSpec {
        SOLID_WHITE (BackgroundId.BG_SOLID_WHITE,       "solid-white"),
        NOISE_LIGHT (BackgroundId.BG_NOISE_LIGHT,       "noise-light"),
        GRADIENT_H  (BackgroundId.BG_GRADIENT_H_COLOUR, "gradient-colour"),
        RAND_CIRCLES(BackgroundId.BG_RANDOM_CIRCLES,    "random-circles"),
        RAND_LINES  (BackgroundId.BG_RANDOM_LINES,      "random-lines"),
        RAND_MIXED  (BackgroundId.BG_RANDOM_MIXED,      "random-mixed");
        final BackgroundId id; final String label;
        BgSpec(BackgroundId id, String label) { this.id = id; this.label = label; }
    }
    private final MatchDiagnosticLibrary diag   = new MatchDiagnosticLibrary();
    private final MatchReportLibrary     report = new MatchReportLibrary();
    @BeforeAll
    void setup() throws IOException {
        OpenCvLoader.load();
        Files.createDirectories(OUTPUT);
        diag.clear();
        report.clear();
        Files.deleteIfExists(OUTPUT.resolve("diagnostics.json"));
        Files.deleteIfExists(OUTPUT.resolve("report.html"));
    }
    @AfterAll
    void writeReports() throws IOException {
        diag.writeReport(OUTPUT);
        report.writeReport(OUTPUT, "VectorMatcher Diagnostic Report");
    }
    @Test
    @DisplayName("Full shape x background matrix")
    void runDiagnostics() {
        for (BgSpec bg : BgSpec.values()) {
            for (ReferenceId refId : ALL_SHAPES) {
                Mat shapeMat = MatchDiagnosticLibrary.buildShapeMat(refId);
                Mat ref      = ReferenceImageFactory.build(refId);
                Rect gt      = MatchDiagnosticLibrary.groundTruthRect(shapeMat);
                Mat scene    = MatchDiagnosticLibrary.compositeOnBackground(shapeMat, bg.id);
                SceneEntry se = new SceneEntry(refId, SceneCategory.A_CLEAN,
                        bg.label, bg.id, Collections.emptyList(), scene);
                List<AnalysisResult> results = VectorMatcher.match(
                        refId, ref, se, Collections.emptySet(), OUTPUT);
                diag.evaluate(bg.id, bg.label, refId, PASS_THRESH, TARGET, GOOD_IOU, OUTPUT);
                report.record(bg.label, bg.label + "/" + refId.name(), refId.name(),
                        bg.label, scene, gt, results, 0L);
                se.release(); shapeMat.release(); ref.release(); scene.release();
            }
        }
    }
    @Test
    @DisplayName("Focused: RECT_FILLED on random-circles")
    void focusedRectOnRandomCircles() {
        runFocused(ReferenceId.RECT_FILLED, BackgroundId.BG_RANDOM_CIRCLES, "random-circles");
    }
    private void runFocused(ReferenceId refId, BackgroundId bgId, String bgLabel) {
        Mat shapeMat = MatchDiagnosticLibrary.buildShapeMat(refId);
        Mat ref      = ReferenceImageFactory.build(refId);
        Rect gt      = MatchDiagnosticLibrary.groundTruthRect(shapeMat);
        Mat scene    = MatchDiagnosticLibrary.compositeOnBackground(shapeMat, bgId);
        SceneEntry se = new SceneEntry(refId, SceneCategory.A_CLEAN,
                bgLabel, bgId, Collections.emptyList(), scene);
        List<AnalysisResult> results = VectorMatcher.match(
                refId, ref, se, Collections.emptySet(), OUTPUT);
        AnalysisResult nr = results.stream()
                .filter(r -> r.methodName().equals(VectorVariant.VECTOR_NORMAL.variantName()))
                .findFirst().orElse(null);
        double score = nr != null ? nr.matchScorePercent() : 0;
        Rect   bbox  = nr != null ? nr.boundingRect() : null;
        double iouV  = bbox != null && gt != null ? MatchDiagnosticLibrary.iou(bbox, gt) : Double.NaN;
        System.out.printf("%n=== FOCUSED: %s on %s ===%n", refId.name(), bgLabel);
        System.out.printf("score=%.1f%%  iou=%.3f  -> %s%n%n", score, iouV,
            iouV >= 0.5 ? "CORRECT HIT" : iouV >= 0.3 ? "BAD IoU" : "FALSE POSITIVE");
        double eps = VectorVariant.VECTOR_NORMAL.epsilonFactor();
        double sa  = (double) scene.rows() * scene.cols();
        VectorSignature refSig = VectorMatcher.buildRefSignature(ref, eps);
        List<SceneDescriptor.ClusterContours> clusters = se.descriptor().clusters();
        System.out.printf("Clusters: %d%n", clusters.size());
        for (int ci = 0; ci < clusters.size(); ci++) {
            SceneDescriptor.ClusterContours cc = clusters.get(ci);
            double maxA = cc.contours.stream()
                    .mapToDouble(c -> { Rect r = Imgproc.boundingRect(c); return (double)r.width*r.height; })
                    .max().orElse(1);
            long sigN = cc.contours.stream()
                    .filter(c -> { Rect r = Imgproc.boundingRect(c); return (double)r.width*r.height >= maxA*0.20; })
                    .count();
            double pen = sigN > 1 ? 1.0/(Math.log(sigN+1)/Math.log(2)) : 1.0;
            System.out.printf("  Cluster %d  hue=%.0f  achromatic=%b  n=%d  penalty=%.3f%n",
                ci, cc.hue, cc.achromatic, cc.contours.size(), pen);
            for (int ki = 0; ki < cc.contours.size(); ki++) {
                MatOfPoint c = cc.contours.get(ki);
                Rect bb = Imgproc.boundingRect(c);
                VectorSignature s = VectorSignature.buildFromContour(c, eps, sa);
                double raw = refSig.similarity(s);
                double iouC = gt != null ? MatchDiagnosticLibrary.iou(bb, gt) : Double.NaN;
                System.out.printf("    [%d]%s (%d,%d %dx%d) raw=%.3f pen=%.3f %s v=%d iou=%.2f%n",
                    ki, bbox != null && bbox.equals(bb) ? " ***WINNER***" : "",
                    bb.x, bb.y, bb.width, bb.height,
                    raw, raw*pen, s.type.name(), s.vertexCount, iouC);
            }
        }
        diag.evaluate(bgId, bgLabel, refId, PASS_THRESH, TARGET, GOOD_IOU, OUTPUT);
        report.record(bgLabel, bgLabel+"/"+refId.name(), refId.name(), bgLabel,
                scene, gt, results, 0L);
        se.release(); shapeMat.release(); ref.release(); scene.release();
    }
    @Test
    @DisplayName("Focused: BICOLOUR_RECT_HALVES on own scene")
    void focusedBicolourRectHalves() {
        runFocusedMultiColour(ReferenceId.BICOLOUR_RECT_HALVES);
    }

    @Test
    @DisplayName("Focused: TRICOLOUR_TRIANGLE on own scene")
    void focusedTricolourTriangle() {
        runFocusedMultiColour(ReferenceId.TRICOLOUR_TRIANGLE);
    }

    @Test
    @DisplayName("Focused: BICOLOUR_CIRCLE_RING on own scene")
    void focusedBicolourCircleRing() {
        runFocusedMultiColour(ReferenceId.BICOLOUR_CIRCLE_RING);
    }

    @Test
    @DisplayName("Focused: BICOLOUR_CROSSHAIR_RING on own scene")
    void focusedBicolourCrosshairRing() {
        runFocusedMultiColour(ReferenceId.BICOLOUR_CROSSHAIR_RING);
    }

    @Test
    @DisplayName("Focused: COMPOUND_CROSS_IN_CIRCLE on own scene")
    void focusedCompoundCrossInCircle() {
        runFocusedMultiColour(ReferenceId.COMPOUND_CROSS_IN_CIRCLE);
    }

    @Test
    @DisplayName("Focused: BICOLOUR_CHEVRON_FILLED on own scene")
    void focusedBicolourChevronFilled() {
        runFocusedMultiColour(ReferenceId.BICOLOUR_CHEVRON_FILLED);
    }

    /** Builds a 640x480 scene with the ref scaled 3x centred on black. */
    private static Mat buildMultiColourScene(ReferenceId id) {
        Mat ref = ReferenceImageFactory.build(id);
        Mat scaled = new Mat();
        Imgproc.resize(ref, scaled, new Size(ref.cols() * 3, ref.rows() * 3), 0, 0, Imgproc.INTER_NEAREST);
        ref.release();
        Mat canvas = Mat.zeros(480, 640, CvType.CV_8UC3);
        int x = (canvas.cols() - scaled.cols()) / 2;
        int y = (canvas.rows() - scaled.rows()) / 2;
        scaled.copyTo(canvas.submat(new Rect(x, y, scaled.cols(), scaled.rows())));
        scaled.release();
        return canvas;
    }

    private void runFocusedMultiColour(ReferenceId refId) {
        Mat ref   = ReferenceImageFactory.build(refId);
        Mat scene = buildMultiColourScene(refId);

        // Dump scene to disk so we can inspect it
        Path sceneOut = OUTPUT.resolve("debug_scene_" + refId.name() + ".png");
        Imgcodecs.imwrite(sceneOut.toString(), scene);

        // Show what clusters the ref image has
        System.out.printf("%n=== REF CLUSTERS: %s ===%n", refId.name());
        List<SceneColourClusters.Cluster> refClusters = SceneColourClusters.extract(ref);
        for (int i = 0; i < refClusters.size(); i++) {
            SceneColourClusters.Cluster c = refClusters.get(i);
            List<MatOfPoint> cnts = SceneDescriptor.contoursFromMask(c.mask);
            System.out.printf("  ref cluster %d: hue=%.0f achromatic=%b px=%d contours=%d%n",
                i, c.hue, c.achromatic, Core.countNonZero(c.mask), cnts.size());
            for (MatOfPoint cnt : cnts) {
                Rect bb = Imgproc.boundingRect(cnt);
                System.out.printf("    contour (%d,%d %dx%d) area=%.0f%n",
                    bb.x, bb.y, bb.width, bb.height, Imgproc.contourArea(cnt));
            }
            c.release();
        }

        // Show what clusters the scene has
        System.out.printf("%n=== SCENE CLUSTERS: %s ===%n", refId.name());
        SceneEntry se = new SceneEntry(refId, SceneCategory.A_CLEAN,
                "own-scene", null, Collections.emptyList(), scene);
        List<SceneDescriptor.ClusterContours> clusters = se.descriptor().clusters();
        double eps = VectorVariant.VECTOR_NORMAL.epsilonFactor();
        double sa  = (double) scene.rows() * scene.cols();
        List<VectorSignature> refSigs = VectorMatcher.buildRefSignatures(ref, eps);
        System.out.printf("  refSigs count: %d%n", refSigs.size());
        for (int ci = 0; ci < clusters.size(); ci++) {
            SceneDescriptor.ClusterContours cc = clusters.get(ci);
            System.out.printf("  scene cluster %d: hue=%.0f achromatic=%b contours=%d%n",
                ci, cc.hue, cc.achromatic, cc.contours.size());
            for (MatOfPoint cnt : cc.contours) {
                Rect bb = Imgproc.boundingRect(cnt);
                VectorSignature vs = VectorSignature.buildFromContour(cnt, eps, sa);
                double bestSim = refSigs.stream().mapToDouble(r -> r.similarity(vs)).max().orElse(0);
                System.out.printf("    contour (%d,%d %dx%d) bestSim=%.3f type=%s v=%d%n",
                    bb.x, bb.y, bb.width, bb.height, bestSim, vs.type.name(), vs.vertexCount);
            }
        }

        // Run matcher and show result
        List<AnalysisResult> results = VectorMatcher.match(refId, ref, se, Collections.emptySet(), OUTPUT);
        AnalysisResult nr = results.stream()
                .filter(r -> r.methodName().equals(VectorVariant.VECTOR_NORMAL.variantName()))
                .findFirst().orElse(null);
        double score = nr != null ? nr.matchScorePercent() : 0;
        Rect   bbox  = nr != null ? nr.boundingRect() : null;
        System.out.printf("%n  RESULT: score=%.1f%%  bbox=%s%n", score,
            bbox != null ? String.format("(%d,%d %dx%d)", bbox.x, bbox.y, bbox.width, bbox.height) : "null");

        report.record("Multi-Colour Debug", refId.name(), refId.name(), "own-scene",
                scene, null, results, 0L);

        se.release(); ref.release(); scene.release();
    }
}

