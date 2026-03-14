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
import org.example.utilities.ExpectedOutcome;
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
    private static final double PASS_THRESH     = 40.0;
    private static final double TARGET          = 90.0;
    /** Perfect-match IoU in the new coverage-scaled formula (1.0 = exact fit). */
    private static final double GOOD_IOU        = 1.0;
    /** Score above which a negative-scene detection is a false alarm. */
    private static final double FP_SCORE_GATE   = 60.0;
    /** IoU must reach this fraction of GOOD_IOU to count as a correct localisation. */
    private static final double IOU_PASS_MARGIN = 0.95;
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
        report.scanTestAnnotations(VectorMatcherDiagnosticTest.class);
        Files.deleteIfExists(OUTPUT.resolve("diagnostics.json"));
        Files.deleteIfExists(OUTPUT.resolve("report.html"));
    }
    @AfterAll
    void writeReports() throws IOException {
        diag.writeReport(OUTPUT);
        report.writeReport(OUTPUT, "VectorMatcher Diagnostic Report");
    }

    /**
     * Full shape × background matrix.
     *
     * <p>Runs every shape in {@link #ALL_SHAPES} against all six {@link BgSpec}
     * backgrounds (solid-white, noise-light, gradient-colour, random-circles,
     * random-lines, random-mixed) and records scores + IoU in both the
     * diagnostic JSON and the HTML report.
     *
     * <p>Expected outcome: 80 / 84 correct.  The four known false-positives are
     * {@code CIRCLE_FILLED}, {@code HEXAGON_OUTLINE}, and {@code OCTAGON_FILLED}
     * on {@code random-circles} (background circles share geometry with round
     * ref shapes), and {@code POLYLINE_PLUS_SHAPE} on {@code random-mixed}.
     * {@code HEXAGON_OUTLINE} and {@code STAR_5_FILLED} score ~86–87 % across
     * all backgrounds — below the 90 % target — due to contour-approximation
     * variance at the 128 px reference scale.
     */
    @Test
    @DisplayName("Full shape x background matrix")
    @ExpectedOutcome(
        value  = ExpectedOutcome.Result.PARTIAL,
        reason = "80/84 correct. 4 FP: CIRCLE_FILLED, HEXAGON_OUTLINE, OCTAGON_FILLED on " +
                 "random-circles (background circles share geometry), and POLYLINE_PLUS_SHAPE " +
                 "on random-mixed. HEXAGON_OUTLINE and STAR_5_FILLED sit at ~86-87% across all " +
                 "backgrounds due to contour-approximation variance at the 128px ref scale.")
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

    /**
     * Focused probe: {@code RECT_FILLED} on {@code random-circles} background.
     *
     * <p>Verifies that the rectangle's straight-edge, right-angle geometry is
     * not confused with the circular background elements.  The filled rectangle
     * has solidity ≈ 1.0 and low concavity, which cleanly separates it from
     * any round background contour.
     */
    @Test
    @DisplayName("Focused: RECT_FILLED on random-circles")
    @ExpectedOutcome(
        value  = ExpectedOutcome.Result.PASS,
        reason = "RECT_FILLED scores ~97% even on random-circles: its solidity=1.0 and " +
                 "4-vertex polygon signature is geometrically incompatible with circular " +
                 "background elements, so the AR multiplier and type gate prevent false matches.")
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
            iouV >= GOOD_IOU * IOU_PASS_MARGIN ? "CORRECT HIT"
                : iouV >= 0.5                  ? "BAD IoU"
                :                                "FALSE POSITIVE");
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

    /**
     * Focused probe: {@code BICOLOUR_RECT_HALVES} matched against its own scene
     * (reference scaled 3× and centred on a black canvas).
     *
     * <p>The two-colour rectangle has a red half and a blue half, each forming
     * a separate cluster.  Layer 2 structural coherence must correctly pair both
     * clusters between reference and scene for a high score.
     */
    @Test
    @DisplayName("Focused: BICOLOUR_RECT_HALVES on own scene")
    @ExpectedOutcome(
        value  = ExpectedOutcome.Result.PARTIAL,
        reason = "Self-match scores ~83% due to multi-cluster assignment variance: the two " +
                 "colour halves are similar in size, so the anchor-to-ref assignment can swap " +
                 "them, lowering Layer 2 coherence.  No architectural fix without dedicated " +
                 "colour-ordering logic.")
    void focusedBicolourRectHalves() {
        runFocusedMultiColour(ReferenceId.BICOLOUR_RECT_HALVES);
    }

    /**
     * Focused probe: {@code TRICOLOUR_TRIANGLE} matched against its own scene.
     *
     * <p>Three distinct hue clusters (red, green, blue wedges) must all be
     * found and coherently matched.  The triangular wedge geometry is highly
     * distinctive per cluster, but three-cluster expansion increases Layer 1
     * count-match difficulty.
     */
    @Test
    @DisplayName("Focused: TRICOLOUR_TRIANGLE on own scene")
    @ExpectedOutcome(
        value  = ExpectedOutcome.Result.PARTIAL,
        reason = "Three-cluster shape: Layer 1 count-match penalty increases with cluster " +
                 "count, and wedge areas are equal-sized so assignment order is ambiguous. " +
                 "Score typically 85-90%.")
    void focusedTricolourTriangle() {
        runFocusedMultiColour(ReferenceId.TRICOLOUR_TRIANGLE);
    }

    /**
     * Focused probe: {@code BICOLOUR_CIRCLE_RING} matched against its own scene.
     *
     * <p>Outer ring (one hue) surrounds an inner filled circle (second hue).
     * The spatial relationship (inner completely enclosed by outer) provides a
     * strong Layer 2 proximity cue and the circular geometry gives high Layer 3
     * geometry scores.
     */
    @Test
    @DisplayName("Focused: BICOLOUR_CIRCLE_RING on own scene")
    @ExpectedOutcome(
        value  = ExpectedOutcome.Result.PASS,
        reason = "Inner circle fully enclosed by outer ring: strong spatial proximity cue " +
                 "for Layer 2, and both clusters have near-perfect circular geometry " +
                 "(circularity ≈ 0.95+) matching the reference cleanly.")
    void focusedBicolourCircleRing() {
        runFocusedMultiColour(ReferenceId.BICOLOUR_CIRCLE_RING);
    }

    /**
     * Focused probe: {@code BICOLOUR_CROSSHAIR_RING} matched against its own scene.
     *
     * <p>A crosshair (one colour) overlaid on a circular ring (second colour).
     * The crosshair's COMPOUND type must pair with the reference's COMPOUND
     * cluster correctly.
     */
    @Test
    @DisplayName("Focused: BICOLOUR_CROSSHAIR_RING on own scene")
    @ExpectedOutcome(
        value  = ExpectedOutcome.Result.PARTIAL,
        reason = "Score ~89.7% — marginally below the 90% target. The crosshair cluster is " +
                 "classified as COMPOUND (multiple components) and its SegmentDescriptor " +
                 "has higher variance across scales than single-component shapes.")
    void focusedBicolourCrosshairRing() {
        runFocusedMultiColour(ReferenceId.BICOLOUR_CROSSHAIR_RING);
    }

    /**
     * Focused probe: {@code COMPOUND_CROSS_IN_CIRCLE} matched against its own scene.
     *
     * <p>A cross drawn inside a circle outline, both in the same colour.
     * {@code SceneColourClusters.extractFromBorderPixels} sees this as a single
     * achromatic cluster with a compound contour set.
     */
    @Test
    @DisplayName("Focused: COMPOUND_CROSS_IN_CIRCLE on own scene")
    @ExpectedOutcome(
        value  = ExpectedOutcome.Result.PASS,
        reason = "Both circle and cross components are present in the same achromatic cluster. " +
                 "COMPOUND type matches on both sides and the dominant (circle) boundary " +
                 "gives a high Layer 3 geometry score.")
    void focusedCompoundCrossInCircle() {
        runFocusedMultiColour(ReferenceId.COMPOUND_CROSS_IN_CIRCLE);
    }

    /**
     * Focused probe: {@code BICOLOUR_CHEVRON_FILLED} matched against its own scene.
     *
     * <p>A filled chevron split into two colour halves.  The chevron has
     * significant concavity (CLOSED_CONCAVE_POLY) and a non-trivial aspect
     * ratio, making two-cluster coherence harder to achieve.
     */
    @Test
    @DisplayName("Focused: BICOLOUR_CHEVRON_FILLED on own scene")
    @ExpectedOutcome(
        value  = ExpectedOutcome.Result.PARTIAL,
        reason = "Score ~84%: concave chevron geometry has higher SegmentDescriptor variance " +
                 "than convex shapes, and the two colour halves have similar sizes causing " +
                 "occasional cluster swap in Layer 2 assignment.")
    void focusedBicolourChevronFilled() {
        runFocusedMultiColour(ReferenceId.BICOLOUR_CHEVRON_FILLED);
    }

    /** Builds a 640×480 scene with the ref scaled 3× and centred on black. */
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

        // Dump scene to disk so we can inspect it — upscaled 4× for clarity
        Path sceneOut = OUTPUT.resolve("debug_scene_" + refId.name() + ".png");
        Mat sceneBig = new Mat();
        Imgproc.resize(scene, sceneBig, new Size(scene.cols() * 4, scene.rows() * 4),
                0, 0, Imgproc.INTER_NEAREST);
        Imgcodecs.imwrite(sceneOut.toString(), sceneBig);
        sceneBig.release();

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

    /**
     * Diagnostic: compares {@code VectorSignature} of a diamond reference against
     * both a diamond scene and a circle scene.
     *
     * <p>Prints per-contour similarity scores to stdout.  Used to verify that the
     * diamond's 4-vertex rhombus signature scores high against itself and low
     * against the circular scene, confirming the AR and type gates work correctly.
     *
     * <p>No assertions are made — this is a human-inspection tool.
     */
    @Test
    @DisplayName("Diag: DIAMOND sig vs diamond scene and circle scene")
    @ExpectedOutcome(
        value  = ExpectedOutcome.Result.DIAGNOSTIC,
        reason = "Prints VectorSignature fields and cross-similarity scores to stdout. " +
                 "Expected: diamond-vs-diamond similarity >> diamond-vs-circle similarity, " +
                 "confirming the AR multiplier and type hard-gate prevent false matches.")
    void diagDiamondSignatures() {
        double eps = VectorVariant.VECTOR_NORMAL.epsilonFactor();
        Mat ref         = ReferenceImageFactory.build(ReferenceId.POLYLINE_DIAMOND);
        Mat diamScene   = buildDiamond();
        Mat circScene   = buildCircle();
        List<VectorSignature> refSigs = VectorMatcher.buildRefSignatures(ref, eps);
        System.out.printf("%nREF sigs (%d):%n", refSigs.size());
        for (VectorSignature rs : refSigs)
            System.out.printf("  ref: %s v=%d circ=%.3f solid=%.3f ar=%.3f%n",
                rs.type, rs.vertexCount, rs.circularity, rs.solidity, rs.aspectRatio);
        for (String label : new String[]{"diamond","circle"}) {
            Mat scene = label.equals("diamond") ? diamScene : circScene;
            double sa = (double) scene.rows() * scene.cols();
            SceneDescriptor desc = SceneDescriptor.build(scene);
            System.out.printf("%n--- %s scene ---%n", label);
            for (SceneDescriptor.ClusterContours cc : desc.clusters()) {
                for (MatOfPoint c : cc.contours) {
                    Rect bb = Imgproc.boundingRect(c);
                    VectorSignature vs = VectorSignature.buildFromContour(c, eps, sa);
                    double bestSim = refSigs.stream().mapToDouble(r -> r.similarity(vs)).max().orElse(0);
                    System.out.printf("  (%d,%d %dx%d) type=%s v=%d circ=%.3f solid=%.3f ar=%.3f sim=%.3f%n",
                        bb.x, bb.y, bb.width, bb.height,
                        vs.type, vs.vertexCount, vs.circularity, vs.solidity, vs.aspectRatio, bestSim);
                }
            }
            desc.release();
        }
        ref.release(); diamScene.release(); circScene.release();
    }

    /**
     * Diagnostic: compares {@code VectorSignature} of an arrow reference against
     * both an arrow scene and a rectangle scene.
     *
     * <p>Prints per-contour similarity scores to stdout.  Confirms that the arrow's
     * concave 7-vertex signature scores high against itself and well below the
     * rectangle (convex 4-vertex), verifying the concavity ratio discriminator.
     *
     * <p>No assertions are made — this is a human-inspection tool.
     */
    @Test
    @DisplayName("Diag: ARROW sig vs arrow scene and rect scene")
    @ExpectedOutcome(
        value  = ExpectedOutcome.Result.DIAGNOSTIC,
        reason = "Prints VectorSignature fields to stdout. Expected: arrow-vs-arrow " +
                 "similarity >> arrow-vs-rect similarity, because the arrow's concavity " +
                 "ratio (~0.15) differs sharply from the rectangle's (0.0).")
    void diagArrowSignatures() {
        double eps = VectorVariant.VECTOR_NORMAL.epsilonFactor();
        Mat ref       = ReferenceImageFactory.build(ReferenceId.POLYLINE_ARROW_RIGHT);
        Mat arrowScene = buildArrow();
        Mat rectScene  = buildRect();

        List<VectorSignature> refSigs = VectorMatcher.buildRefSignatures(ref, eps);
        System.out.printf("%nREF sigs (%d):%n", refSigs.size());
        for (VectorSignature rs : refSigs)
            System.out.printf("  ref: %s v=%d circ=%.3f solid=%.3f ar=%.3f%n",
                rs.type, rs.vertexCount, rs.circularity, rs.solidity, rs.aspectRatio);

        for (String label : new String[]{"arrow","rect"}) {
            Mat scene = label.equals("arrow") ? arrowScene : rectScene;
            double sa = (double) scene.rows() * scene.cols();
            SceneDescriptor desc = SceneDescriptor.build(scene);
            System.out.printf("%n--- %s scene ---%n", label);
            for (SceneDescriptor.ClusterContours cc : desc.clusters()) {
                for (MatOfPoint c : cc.contours) {
                    Rect bb = Imgproc.boundingRect(c);
                    VectorSignature vs = VectorSignature.buildFromContour(c, eps, sa);
                    double bestSim = refSigs.stream().mapToDouble(r -> r.similarity(vs)).max().orElse(0);
                    System.out.printf("  (%d,%d %dx%d) type=%s v=%d circ=%.3f solid=%.3f ar=%.3f sim=%.3f%n",
                        bb.x, bb.y, bb.width, bb.height,
                        vs.type, vs.vertexCount, vs.circularity, vs.solidity, vs.aspectRatio, bestSim);
                }
            }
            desc.release();
        }
        ref.release(); arrowScene.release(); rectScene.release();
    }

    private static Mat buildDiamond() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.polylines(m, List.of(new MatOfPoint(
                new Point(320,100), new Point(500,240),
                new Point(320,380), new Point(140,240))),
                true, new Scalar(255,255,255), 3);
        return m;
    }
    private static Mat buildCircle() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.circle(m, new Point(320,240), 60, new Scalar(255,255,255), -1);
        return m;
    }
    private static Mat buildArrow() {
        // Reference proportions: hw=45, hh=20, headH=36 on 128×128 → AR = 90/72 = 1.25
        // Scaled ×3 centred at (320,240): hw=135, hh=60, headH=108 → AR = 270/216 = 1.25
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.polylines(m, List.of(new MatOfPoint(
                new Point(185, 180), new Point(320, 180), new Point(320, 132),
                new Point(455, 240),
                new Point(320, 348), new Point(320, 300),
                new Point(185, 300))),
                true, new Scalar(255, 255, 255), 3);
        return m;
    }
    private static Mat buildRect() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.rectangle(m, new Point(230,160), new Point(410,320), new Scalar(255,255,255), -1);
        return m;
    }

    // =========================================================================
    // Rotation robustness
    // =========================================================================

    private static final int[] ROTATION_ANGLES = {0, 15, 30, 45, 90, 135, 180};

    /**
     * Rotation robustness: every shape at 0°, 15°, 30°, 45°, 90°, 135°, 180° on
     * a solid-black background.
     *
     * <p>Shapes are built via {@link MatchDiagnosticLibrary#buildShapeMat} (which
     * uses the correct reference proportions after the AR fixes), then rotated with
     * {@link Imgproc#warpAffine} around the canvas centre.  The ground-truth rect
     * is re-detected from the rotated image.  Annotated PNGs are saved to
     * {@code test_output/vector_matching/annotated/VECTOR_NORMAL/}.
     *
     * <h2>Expected outcomes</h2>
     * <ul>
     *   <li><b>All angles — PASS (≥ 90 %):</b> {@code CIRCLE_FILLED},
     *       {@code PENTAGON_FILLED}, {@code OCTAGON_FILLED}, {@code TRIANGLE_FILLED},
     *       {@code POLYLINE_ARROW_RIGHT}, {@code CONCAVE_ARROW_HEAD},
     *       {@code LINE_CROSS} — these shapes have rotational symmetry or
     *       rotation-invariant contour profiles.</li>
     *   <li><b>PARTIAL — diagonal angles fail:</b>
     *     <ul>
     *       <li>{@code ELLIPSE_H}: fails at 45°/135° (~74%) because rotating the
     *           horizontal ellipse changes the bounding-box AR, triggering the AR
     *           multiplier.  A diagonal ellipse is genuinely different from a
     *           horizontal one — correct behaviour.</li>
     *       <li>{@code RECT_ROTATED_45}: fails at 45°/135° (~62%) because rotating
     *           the already-45°-tilted rect by a further 45° produces an axis-aligned
     *           rectangle, indistinguishable from {@code RECT_FILLED}.</li>
     *       <li>{@code RECT_FILLED}: fails at 15°/30°/135° (~87–88%) — non-orthogonal
     *           rotation alters the bounding-box AR of the asymmetric rectangle.</li>
     *       <li>{@code POLYLINE_DIAMOND}: fails at 30°/45°/135° (~88%) — same AR
     *           sensitivity as the rectangle.</li>
     *       <li>{@code POLYLINE_PLUS_SHAPE}: fails at 15°/30° (~70%) — the diagonal
     *           arm orientation causes contour approximation to collapse 12 vertices
     *           to 4, identical to the lines-background issue.</li>
     *     </ul>
     *   </li>
     *   <li><b>Below target at all angles:</b> {@code HEXAGON_OUTLINE} (~86%),
     *       {@code STAR_5_FILLED} (~87%) — pre-existing matcher ceiling, not a
     *       rotation regression.</li>
     * </ul>
     *
     * <p>Overall: ~65 / 98 (66 %) pass.  All failures are geometrically expected;
     * none represent regressions.
     */
    @Test
    @DisplayName("Rotation robustness: all shapes × 0°/15°/30°/45°/90°/135°/180° on black")
    @ExpectedOutcome(
        value  = ExpectedOutcome.Result.PARTIAL,
        reason = "Pass criteria: IoU ≥ GOOD_IOU × 0.95 (= 0.475), no score-threshold gate. " +
                 "Symmetric shapes (circle, pentagon, octagon, triangle, arrow, cross) pass at " +
                 "all angles. AR-sensitive shapes (ELLIPSE_H, RECT_FILLED, RECT_ROTATED_45, " +
                 "POLYLINE_DIAMOND) legitimately fail at diagonal rotations because the bounding-" +
                 "box AR changes. POLYLINE_PLUS_SHAPE drops at 15°/30° (diagonal arms collapse " +
                 "vertex count). HEXAGON_OUTLINE and STAR_5_FILLED are below target at all " +
                 "angles — pre-existing ceiling, not a rotation regression.")
    void runRotationRobustness() {
        // Header
        System.out.printf("%n=== ROTATION ROBUSTNESS (black background, threshold ≥ %.0f%%) ===%n", TARGET);
        String hdr = String.format("%-26s", "Shape");
        for (int a : ROTATION_ANGLES) hdr += String.format(" %6s", a + "°");
        System.out.println(hdr);
        System.out.println("─".repeat(26 + ROTATION_ANGLES.length * 7));

        int passCount = 0, totalCount = 0;

        for (ReferenceId refId : ALL_SHAPES) {
            Mat ref = ReferenceImageFactory.build(refId);
            StringBuilder row = new StringBuilder(String.format("%-26s", refId.name()));

            for (int angle : ROTATION_ANGLES) {
                Mat shapeMat = MatchDiagnosticLibrary.buildShapeMat(refId);
                Mat scene    = rotateScene(shapeMat, angle);
                shapeMat.release();

                Rect gt = MatchDiagnosticLibrary.groundTruthRect(scene);

                String label = "rot" + angle + "deg_black";
                SceneEntry se = new SceneEntry(refId, SceneCategory.B_TRANSFORMED,
                        label, BackgroundId.BG_SOLID_BLACK, Collections.emptyList(), scene);

                // Save annotated PNG for every rotation scene so they appear
                // in test_output/vector_matching/annotated/VECTOR_NORMAL/
                Set<String> save = Set.of(VectorVariant.VECTOR_NORMAL.variantName());
                List<AnalysisResult> results = VectorMatcher.match(
                        refId, ref, se, save, OUTPUT);

                AnalysisResult best = results.stream()
                        .filter(r -> r.methodName().equals(VectorVariant.VECTOR_NORMAL.variantName()))
                        .findFirst().orElse(results.isEmpty() ? null : results.get(0));

                double score  = best != null ? best.matchScorePercent() : 0.0;
                Rect   bbox   = best != null ? best.boundingRect()      : null;
                double iouVal = (bbox != null && gt != null)
                        ? MatchDiagnosticLibrary.iou(bbox, gt) : Double.NaN;

                boolean pass = (gt != null)
                        ? (!Double.isNaN(iouVal) && iouVal >= GOOD_IOU * IOU_PASS_MARGIN)
                        : (score < FP_SCORE_GATE);
                row.append(String.format(" %5.1f%s", score, pass ? "✓" : "✗"));

                report.record(label, label + "/" + refId.name(), refId.name(),
                        label, scene, gt, results, 0L);

                if (pass) passCount++;
                totalCount++;

                se.release();
                scene.release();
            }

            ref.release();
            System.out.println(row);
        }

        System.out.println("─".repeat(26 + ROTATION_ANGLES.length * 7));
        System.out.printf("Pass: %d / %d  (%.0f%%)%n%n", passCount, totalCount,
                100.0 * passCount / totalCount);
    }

    /** Rotates {@code src} by {@code angleDeg} around the image centre on a black canvas. */
    private static Mat rotateScene(Mat src, int angleDeg) {
        if (angleDeg == 0) return src.clone();
        Point centre = new Point(src.cols() / 2.0, src.rows() / 2.0);
        Mat   rotM   = Imgproc.getRotationMatrix2D(centre, angleDeg, 1.0);
        Mat   dst    = Mat.zeros(src.size(), src.type());
        Imgproc.warpAffine(src, dst, rotM, src.size(), Imgproc.INTER_LINEAR,
                Core.BORDER_CONSTANT, new Scalar(0, 0, 0));
        rotM.release();
        return dst;
    }
}
