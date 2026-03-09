package org.example.vectormatcher;

import org.example.OpenCvLoader;
import org.example.colour.SceneColourClusters;
import org.example.factories.BackgroundFactory;
import org.example.factories.BackgroundId;
import org.example.factories.ReferenceId;
import org.example.factories.ReferenceImageFactory;
import org.example.matchers.SceneDescriptor;
import org.example.matchers.VectorMatcher;
import org.example.matchers.VectorSignature;
import org.example.matchers.VectorVariant;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Standalone diagnostic test — writes diagnostics.json with details of every
 * case (FP, bad IoU, correct-but-low-score, and passing).
 *
 * Run with: mvn test -Dtest=VectorMatcherDiagnosticTest
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class VectorMatcherDiagnosticTest {

    private static final Path OUTPUT = Paths.get("test_output", "vector_matching");
    private static final double PASS_THRESHOLD  = 40.0;   // min score to count as a detection
    private static final double TARGET_SCORE    = 75.0;   // score we want correct detections to reach
    private static final double GOOD_IOU        = 0.5;    // IoU above which location is "correct"

    // ── All shapes under test ────────────────────────────────────────────────
    private static final ReferenceId[] ALL_SHAPES = {
        ReferenceId.CIRCLE_FILLED,
        ReferenceId.RECT_FILLED,
        ReferenceId.TRIANGLE_FILLED,
        ReferenceId.HEXAGON_OUTLINE,
        ReferenceId.PENTAGON_FILLED,
        ReferenceId.STAR_5_FILLED,
        ReferenceId.POLYLINE_DIAMOND,
        ReferenceId.POLYLINE_ARROW_RIGHT,
        ReferenceId.ELLIPSE_H,
        ReferenceId.OCTAGON_FILLED,
        ReferenceId.POLYLINE_PLUS_SHAPE,
        ReferenceId.CONCAVE_ARROW_HEAD,
        ReferenceId.LINE_CROSS,
        ReferenceId.RECT_ROTATED_45,
    };

    // ── Backgrounds under test ───────────────────────────────────────────────
    private enum BgSpec {
        SOLID_WHITE   (BackgroundId.BG_SOLID_WHITE,       "solid-white",    40.0),
        NOISE_LIGHT   (BackgroundId.BG_NOISE_LIGHT,       "noise-light",    40.0),
        GRADIENT_H    (BackgroundId.BG_GRADIENT_H_COLOUR, "gradient-colour", 40.0),
        RAND_CIRCLES  (BackgroundId.BG_RANDOM_CIRCLES,    "random-circles", 20.0),
        RAND_LINES    (BackgroundId.BG_RANDOM_LINES,      "random-lines",   20.0),
        RAND_MIXED    (BackgroundId.BG_RANDOM_MIXED,      "random-mixed",   20.0);

        final BackgroundId id; final String label; final double threshold;
        BgSpec(BackgroundId id, String label, double threshold) {
            this.id = id; this.label = label; this.threshold = threshold;
        }
    }

    // ── Result row ───────────────────────────────────────────────────────────
    record DiagRow(
        String bg, String shape,
        double score, double iou,
        // classification flags
        boolean falsePositive,   // score >= PASS_THRESHOLD but IoU < 0.3
        boolean badIou,          // score >= PASS_THRESHOLD, IoU in [0.3,0.5)
        boolean correctHit,      // score >= PASS_THRESHOLD and IoU >= 0.5
        boolean lowScore,        // correctHit but score < TARGET_SCORE
        boolean missed,          // shape IS in scene but score < PASS_THRESHOLD
        // bboxes
        int gtX, int gtY, int gtW, int gtH,
        int detX, int detY, int detW, int detH,
        // signatures
        String refSig, String detSig,
        double circScore, double solidScore, double totalSim,
        int refVertices, int detVertices,
        String refType, String detType
    ) {}

    private final List<DiagRow> rows = new ArrayList<>();
    private final List<String> debugLines = new ArrayList<>();

    @BeforeAll
    void setup() throws IOException {
        OpenCvLoader.load();
        Files.createDirectories(OUTPUT);
    }

    @Test
    void runDiagnostics() throws IOException {
        for (BgSpec bg : BgSpec.values()) {
            for (ReferenceId refId : ALL_SHAPES) {
                runOne(bg, refId);
            }
        }
        writeJson();
    }

    // ─────────────────────────────────────────────────────────────────────────

    private void runOne(BgSpec bg, ReferenceId refId) {
        Mat shapeMat = buildShapeMat(refId);
        Mat ref      = ReferenceImageFactory.build(refId);
        Rect gt      = groundTruthRect(shapeMat);
        Mat scene    = compositeOnBackground(shapeMat, bg.id);

        double eps       = VectorVariant.VECTOR_NORMAL.epsilonFactor();
        double sceneArea = (double) scene.rows() * scene.cols();

        SceneDescriptor descriptor = SceneDescriptor.build(scene);
        VectorSignature refSig     = VectorMatcher.buildRefSignature(ref, eps);

        double bestScore = 0.0;
        Rect   bestBbox  = null;
        VectorSignature bestSceneSig = null;

        for (SceneDescriptor.ClusterContours cc : descriptor.clusters()) {
            for (MatOfPoint c : cc.contours) {
                VectorSignature sceneSig = VectorSignature.buildFromContour(c, eps, sceneArea);
                double sim = refSig.similarity(sceneSig);
                if (sim > bestScore) {
                    bestScore    = sim;
                    bestBbox     = Imgproc.boundingRect(c);
                    bestSceneSig = sceneSig;
                    // Capture for debug
                    if (sim * 100 >= PASS_THRESHOLD) {
                        Rect b = bestBbox;
                        double na = sceneSig.normalisedArea;
                        // Re-run similarity to verify we get same result
                        double simCheck = refSig.similarity(sceneSig);
                        debugLines.add(String.format("HIT %-16s %-24s sim=%.4f simCheck=%.4f normArea=%.4f gateWouldFire=%b bbox=(%d,%d,%d,%d)",
                            bg.label, refId.name(), sim, simCheck, na,
                            !Double.isNaN(na) && na > 0.80,
                            b.x, b.y, b.width, b.height));
                    }
                }
            }
        }

        double  scorePercent  = bestScore * 100.0;
        double  iou           = Double.NaN;
        boolean falsePositive = false;
        boolean badIou        = false;
        boolean correctHit    = false;
        boolean missed        = false;

        // Debug: log normArea for any result with score > threshold
        if (scorePercent >= PASS_THRESHOLD && bestSceneSig != null) {
            debugLines.add(String.format("%-16s %-24s score=%5.1f normArea=%.4f det@(%d,%d,%d,%d)",
                bg.label, refId.name(), scorePercent, bestSceneSig.normalisedArea,
                bestBbox != null ? bestBbox.x : -1, bestBbox != null ? bestBbox.y : -1,
                bestBbox != null ? bestBbox.width : -1, bestBbox != null ? bestBbox.height : -1));
        }

        if (bestBbox != null && gt != null) {
            iou           = iou(bestBbox, gt);
            falsePositive = (scorePercent >= PASS_THRESHOLD) && (iou < 0.3);
            badIou        = (scorePercent >= PASS_THRESHOLD) && (iou >= 0.3) && (iou < GOOD_IOU);
            correctHit    = (scorePercent >= PASS_THRESHOLD) && (iou >= GOOD_IOU);
        }
        // Missed: shape exists in scene but nothing scored above threshold
        if (scorePercent < PASS_THRESHOLD && gt != null) missed = true;

        boolean lowScore = correctHit && (scorePercent < TARGET_SCORE);

        double circScore  = 0, solidScore = 0;
        if (refSig != null && bestSceneSig != null) {
            circScore  = 1.0 - Math.abs(refSig.circularity - bestSceneSig.circularity);
            solidScore = 1.0 - Math.abs(refSig.solidity    - bestSceneSig.solidity);
        }

        rows.add(new DiagRow(
            bg.label, refId.name(),
            scorePercent, iou,
            falsePositive, badIou, correctHit, lowScore, missed,
            gt  != null ? gt.x      : -1, gt  != null ? gt.y       : -1,
            gt  != null ? gt.width  : -1, gt  != null ? gt.height  : -1,
            bestBbox != null ? bestBbox.x     : -1, bestBbox != null ? bestBbox.y      : -1,
            bestBbox != null ? bestBbox.width : -1, bestBbox != null ? bestBbox.height : -1,
            refSig       != null ? refSig.toString()       : "null",
            bestSceneSig != null ? bestSceneSig.toString() : "null",
            circScore, solidScore,
            refSig.similarity(bestSceneSig != null ? bestSceneSig : refSig),
            refSig       != null ? refSig.vertexCount       : -1,
            bestSceneSig != null ? bestSceneSig.vertexCount : -1,
            refSig       != null ? refSig.type.name()       : "null",
            bestSceneSig != null ? bestSceneSig.type.name() : "null"
        ));

        descriptor.release();
        shapeMat.release();
        ref.release();
        scene.release();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // JSON / text writer
    // ─────────────────────────────────────────────────────────────────────────

    private void writeJson() throws IOException {
        long fpCount      = rows.stream().filter(DiagRow::falsePositive).count();
        long biouCount    = rows.stream().filter(DiagRow::badIou).count();
        long correctCount = rows.stream().filter(DiagRow::correctHit).count();
        long lowCount     = rows.stream().filter(DiagRow::lowScore).count();
        long missedCount  = rows.stream().filter(DiagRow::missed).count();
        long total        = rows.size();

        StringBuilder sb = new StringBuilder();
        sb.append("// ====== VectorMatcher Diagnostics ======\n");
        sb.append(String.format("// Total: %d  |  Correct: %d  |  LowScore(<75%%): %d  |  BadIoU: %d  |  FP: %d  |  Missed: %d%n%n",
            total, correctCount, lowCount, biouCount, fpCount, missedCount));

        sb.append("// ---- FALSE POSITIVES ----\n");
        rows.stream().filter(DiagRow::falsePositive).forEach(r -> appendRow(sb, r, "FP "));

        sb.append("\n// ---- BAD IoU (detected but wrong location) ----\n");
        rows.stream().filter(DiagRow::badIou).forEach(r -> appendRow(sb, r, "BIOU"));

        sb.append("\n// ---- MISSED (shape in scene, nothing detected) ----\n");
        rows.stream().filter(DiagRow::missed).forEach(r -> appendRow(sb, r, "MISS"));

        sb.append("\n// ---- CORRECT BUT LOW SCORE (< 75%) ----\n");
        rows.stream().filter(DiagRow::lowScore).forEach(r -> appendRow(sb, r, "LOW "));

        sb.append("\n// ---- ALL CORRECT HITS ----\n");
        rows.stream().filter(DiagRow::correctHit).forEach(r -> appendRow(sb, r, "OK  "));

        // Full JSON array — all rows
        sb.append("\n// ---- FULL DATA ----\n[\n");
        for (int i = 0; i < rows.size(); i++) {
            DiagRow r = rows.get(i);
            String status = r.falsePositive() ? "FP" : r.badIou() ? "BIOU"
                          : r.missed() ? "MISS" : r.lowScore() ? "LOW" : r.correctHit() ? "OK" : "NONE";
            sb.append("  {\"bg\":\"").append(r.bg())
              .append("\",\"shape\":\"").append(r.shape())
              .append("\",\"status\":\"").append(status)
              .append("\",\"score\":").append(fmt(r.score()))
              .append(",\"iou\":").append(Double.isNaN(r.iou()) ? "null" : fmt(r.iou()))
              .append(",\"refType\":\"").append(r.refType())
              .append("\",\"detType\":\"").append(r.detType())
              .append("\",\"vRef\":").append(r.refVertices())
              .append(",\"vDet\":").append(r.detVertices())
              .append(",\"circ\":").append(fmt(r.circScore()))
              .append(",\"solid\":").append(fmt(r.solidScore()))
              .append(",\"gt\":{\"x\":").append(r.gtX()).append(",\"y\":").append(r.gtY())
              .append(",\"w\":").append(r.gtW()).append(",\"h\":").append(r.gtH()).append("}")
              .append(",\"det\":{\"x\":").append(r.detX()).append(",\"y\":").append(r.detY())
              .append(",\"w\":").append(r.detW()).append(",\"h\":").append(r.detH()).append("}")
              .append(",\"refSig\":\"").append(j(r.refSig()))
              .append("\",\"detSig\":\"").append(j(r.detSig())).append("\"}")
              .append(i < rows.size() - 1 ? "," : "").append("\n");
        }
        sb.append("]\n");

        Path out = OUTPUT.resolve("diagnostics.json");
        Files.writeString(out, sb.toString(), StandardCharsets.UTF_8);

        // Write debug info
        Files.writeString(OUTPUT.resolve("debug_normarea.txt"),
            String.join("\n", debugLines), StandardCharsets.UTF_8);

        // ── stdout summary ────────────────────────────────────────────────
        System.out.printf("%n=== SUMMARY  total=%d  correct=%d  lowScore=%d  badIoU=%d  FP=%d  missed=%d ===%n",
            total, correctCount, lowCount, biouCount, fpCount, missedCount);

        printSection("FALSE POSITIVES", rows.stream().filter(DiagRow::falsePositive).collect(Collectors.toList()));
        printSection("BAD IoU", rows.stream().filter(DiagRow::badIou).collect(Collectors.toList()));
        printSection("MISSED", rows.stream().filter(DiagRow::missed).collect(Collectors.toList()));
        printSection("CORRECT BUT LOW SCORE (<75%)", rows.stream().filter(DiagRow::lowScore).collect(Collectors.toList()));
        System.out.println("\n=== CORRECT HITS ===");
        rows.stream().filter(DiagRow::correctHit).forEach(r ->
            System.out.printf("  OK   %-18s %-25s score=%5.1f%% iou=%.2f%n",
                r.bg(), r.shape(), r.score(), Double.isNaN(r.iou()) ? 0 : r.iou()));
    }

    private void printSection(String title, List<DiagRow> list) {
        System.out.println("\n=== " + title + " ===");
        list.forEach(r -> System.out.printf("  %-4s %-18s %-25s score=%5.1f%% iou=%.2f  refType=%-20s detType=%-20s vRef=%d vDet=%d  circ=%.2f solid=%.2f%n",
            r.falsePositive() ? "FP" : r.badIou() ? "BIOU" : r.missed() ? "MISS" : "LOW ",
            r.bg(), r.shape(), r.score(), Double.isNaN(r.iou()) ? 0 : r.iou(),
            r.refType(), r.detType(), r.refVertices(), r.detVertices(),
            r.circScore(), r.solidScore()));
    }

    private void appendRow(StringBuilder sb, DiagRow r, String tag) {
        sb.append(String.format("// %s %-18s %-25s score=%5.1f%% iou=%.2f  det@(%d,%d,%d,%d)  gt@(%d,%d,%d,%d)  refType=%-20s detType=%-20s vRef=%d vDet=%d  circ=%.2f solid=%.2f%n",
            tag, r.bg(), r.shape(), r.score(), Double.isNaN(r.iou()) ? 0 : r.iou(),
            r.detX(), r.detY(), r.detW(), r.detH(),
            r.gtX(), r.gtY(), r.gtW(), r.gtH(),
            r.refType(), r.detType(), r.refVertices(), r.detVertices(),
            r.circScore(), r.solidScore()));
    }

    private static String fmt(double v) { return String.format("%.4f", v); }
    private static String j(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Scene / shape builders
    // ─────────────────────────────────────────────────────────────────────────

    private static Mat buildShapeMat(ReferenceId id) {
        return switch (id) {
            case CIRCLE_FILLED        -> whiteCircleOnBlack(320, 240, 60);
            case RECT_FILLED          -> whiteRectOnBlack(230, 160, 410, 320);
            case TRIANGLE_FILLED      -> whiteTriangleOnBlack();
            case HEXAGON_OUTLINE      -> whiteHexagonOnBlack();
            case PENTAGON_FILLED      -> whitePentagonOnBlack();
            case STAR_5_FILLED        -> whiteStarOnBlack();
            case POLYLINE_DIAMOND     -> whiteDiamondOnBlack();
            case POLYLINE_ARROW_RIGHT -> whiteArrowOnBlack();
            case ELLIPSE_H            -> whiteEllipseOnBlack();
            case OCTAGON_FILLED       -> whiteOctagonOnBlack();
            case POLYLINE_PLUS_SHAPE  -> whitePlusOnBlack();
            case CONCAVE_ARROW_HEAD   -> whiteConcaveArrowheadOnBlack();
            case LINE_CROSS           -> whiteCrossOnBlack();
            case RECT_ROTATED_45      -> whiteRot45RectOnBlack();
            default                   -> Mat.zeros(480, 640, CvType.CV_8UC3);
        };
    }

    private static Mat whiteCircleOnBlack(int cx, int cy, int r) {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.circle(m, new Point(cx, cy), r, new Scalar(255,255,255), -1);
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
            new Point(320,130), new Point(180,350), new Point(460,350))), new Scalar(255,255,255));
        return m;
    }
    private static Mat whiteHexagonOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Point[] pts = new Point[6];
        for (int i = 0; i < 6; i++) {
            double a = Math.toRadians(60*i - 30);
            pts[i] = new Point(320 + 80*Math.cos(a), 240 + 80*Math.sin(a));
        }
        Imgproc.fillPoly(m, List.of(new MatOfPoint(pts)), new Scalar(255,255,255));
        return m;
    }
    private static Mat whitePentagonOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Point[] pts = new Point[5];
        for (int i = 0; i < 5; i++) {
            double a = Math.toRadians(72*i - 90);
            pts[i] = new Point(320 + 90*Math.cos(a), 240 + 90*Math.sin(a));
        }
        Imgproc.fillPoly(m, List.of(new MatOfPoint(pts)), new Scalar(255,255,255));
        return m;
    }
    private static Mat whiteStarOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Point[] pts = new Point[10];
        for (int i = 0; i < 10; i++) {
            double a = Math.toRadians(36*i - 90);
            int r = (i%2==0) ? 100 : 40;
            pts[i] = new Point(320 + r*Math.cos(a), 240 + r*Math.sin(a));
        }
        Imgproc.fillPoly(m, List.of(new MatOfPoint(pts)), new Scalar(255,255,255));
        return m;
    }
    private static Mat whiteDiamondOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.fillPoly(m, List.of(new MatOfPoint(
            new Point(320,110), new Point(470,240),
            new Point(320,370), new Point(170,240))), new Scalar(255,255,255));
        return m;
    }
    private static Mat whiteArrowOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.fillPoly(m, List.of(new MatOfPoint(
            new Point(160,200), new Point(340,200), new Point(340,155),
            new Point(480,240), new Point(340,325), new Point(340,280),
            new Point(160,280))), new Scalar(255,255,255));
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
            double a = Math.toRadians(45*i - 22.5);
            pts[i] = new Point(320 + 85*Math.cos(a), 240 + 85*Math.sin(a));
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
            new Point(320,290), new Point(180,370))), new Scalar(255,255,255));
        return m;
    }
    private static Mat whiteCrossOnBlack() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.line(m, new Point(320,80),  new Point(320,400), new Scalar(255,255,255), 8);
        Imgproc.line(m, new Point(100,240), new Point(540,240), new Scalar(255,255,255), 8);
        return m;
    }
    private static Mat whiteRot45RectOnBlack() {
        Mat base = whiteRectOnBlack(230, 160, 410, 320);
        Point centre = new Point(base.cols() / 2.0, base.rows() / 2.0);
        Mat rot = Imgproc.getRotationMatrix2D(centre, -45, 1.0);
        Mat dst = Mat.zeros(base.size(), base.type());
        Imgproc.warpAffine(base, dst, rot, base.size());
        rot.release(); base.release();
        return dst;
    }

    private static Mat compositeOnBackground(Mat shapeMat, BackgroundId bgId) {
        Mat scene    = BackgroundFactory.build(bgId, shapeMat.cols(), shapeMat.rows());
        Mat grey     = new Mat(); Imgproc.cvtColor(scene, grey, Imgproc.COLOR_BGR2GRAY);
        double bgLuma = Core.mean(grey).val[0]; grey.release();
        Mat foreground = shapeMat.clone();
        if (bgLuma > 100) Core.bitwise_not(foreground, foreground);
        Mat maskGrey = new Mat(); Mat mask = new Mat();
        Imgproc.cvtColor(shapeMat, maskGrey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(maskGrey, mask, 5, 255, Imgproc.THRESH_BINARY);
        maskGrey.release();
        foreground.copyTo(scene, mask);
        foreground.release(); mask.release();
        return scene;
    }

    private static Rect groundTruthRect(Mat shapeMat) {
        Mat grey = new Mat(); Mat bin = new Mat();
        Imgproc.cvtColor(shapeMat, grey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(grey, bin, 5, 255, Imgproc.THRESH_BINARY);
        grey.release();
        List<MatOfPoint> cs = new ArrayList<>();
        Imgproc.findContours(bin, cs, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        bin.release();
        if (cs.isEmpty()) return null;
        Rect r = Imgproc.boundingRect(cs.get(0));
        for (int i = 1; i < cs.size(); i++) {
            Rect b = Imgproc.boundingRect(cs.get(i));
            int x1 = Math.min(r.x, b.x), y1 = Math.min(r.y, b.y);
            int x2 = Math.max(r.x+r.width, b.x+b.width);
            int y2 = Math.max(r.y+r.height, b.y+b.height);
            r = new Rect(x1, y1, x2-x1, y2-y1);
        }
        return r;
    }

    private static double iou(Rect a, Rect b) {
        int ix1 = Math.max(a.x, b.x), iy1 = Math.max(a.y, b.y);
        int ix2 = Math.min(a.x+a.width,  b.x+b.width);
        int iy2 = Math.min(a.y+a.height, b.y+b.height);
        if (ix2 <= ix1 || iy2 <= iy1) return 0.0;
        double inter = (double)(ix2-ix1)*(iy2-iy1);
        double ua = (double)a.width*a.height, ub = (double)b.width*b.height;
        return inter / (ua + ub - inter);
    }
}







