package org.example.utilities;

import org.example.analytics.AnalysisResult;
import org.example.colour.ColourCluster;
import org.example.colour.SceneColourClusters;
import org.example.factories.BackgroundId;
import org.example.factories.ReferenceId;
import org.example.matchers.SceneDescriptor;
import org.example.matchers.VectorSignature;
import org.example.matchers.VectorVariant;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Shared diagnostic library for VectorMatcher tests.
 *
 * <p>Accumulates per-case {@link DiagRow} records produced by
 * {@link #recordResult} then writes a combined JSON + stdout report
 * via {@link #writeReport(Path)}.
 *
 * <p>Usage:
 * <pre>
 *   MatchDiagnosticLibrary diag = new MatchDiagnosticLibrary();
 *   diag.recordResult(BackgroundId.BG_SOLID_WHITE, "solid-white",
 *                     ReferenceId.CIRCLE_FILLED, results, gt, 40.0, 75.0, 1.0);
 *   diag.writeReport(OUTPUT);
 * </pre>
 */
public class MatchDiagnosticLibrary {

    // ── Result row ────────────────────────────────────────────────────────────
    public record DiagRow(
        String bg, String shape,
        double score, double iou,
        boolean falsePositive,
        boolean badIou,
        boolean correctHit,
        boolean lowScore,
        boolean missed,
        int gtX, int gtY, int gtW, int gtH,
        int detX, int detY, int detW, int detH,
        String refSig, String detSig,
        double circScore, double solidScore, double totalSim,
        int refVertices, int detVertices,
        String refType, String detType,
        /* All other contour hits above 1% similarity: each entry [x, y, w, h, scorePct]. */
        List<double[]> otherHits
    ) {}

    private final List<DiagRow> rows = new ArrayList<>();

    /** Clear all accumulated rows (call in @BeforeAll to reset between runs). */
    public void clear() { rows.clear(); }

    /** Read-only view of accumulated rows. */
    public List<DiagRow> rows() { return Collections.unmodifiableList(rows); }


    // ── Report writer ─────────────────────────────────────────────────────────

    /**
     * Writes {@code diagnostics.json} and prints a summary to stdout.
     * Only covers the rows accumulated since the last {@link #clear()}.
     */
    public void writeReport(Path outputDir) throws IOException {
        Files.createDirectories(outputDir);

        long fpCount      = rows.stream().filter(DiagRow::falsePositive).count();
        long biouCount    = rows.stream().filter(DiagRow::badIou).count();
        long correctCount = rows.stream().filter(DiagRow::correctHit).count();
        long lowCount     = rows.stream().filter(DiagRow::lowScore).count();
        long missedCount  = rows.stream().filter(DiagRow::missed).count();
        long total        = rows.size();

        StringBuilder sb = new StringBuilder();
        sb.append("// ====== VectorMatcher Diagnostics ======\n");
        sb.append(String.format(
            "// Total: %d  |  Correct: %d  |  LowScore(<75%%): %d  |  BadIoU: %d  |  FP: %d  |  Missed: %d%n%n",
            total, correctCount, lowCount, biouCount, fpCount, missedCount));

        appendSection(sb, "FALSE POSITIVES",               rows.stream().filter(DiagRow::falsePositive).collect(Collectors.toList()), "FP  ");
        appendSection(sb, "BAD IoU (wrong location)",      rows.stream().filter(DiagRow::badIou).collect(Collectors.toList()),        "BIOU");
        appendSection(sb, "MISSED",                        rows.stream().filter(DiagRow::missed).collect(Collectors.toList()),        "MISS");
        appendSection(sb, "CORRECT BUT LOW SCORE (< 75%)", rows.stream().filter(DiagRow::lowScore).collect(Collectors.toList()),      "LOW ");
        appendSection(sb, "ALL CORRECT HITS",              rows.stream().filter(DiagRow::correctHit).collect(Collectors.toList()),    "OK  ");

        // Full JSON array
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
              .append(",\"vRef\":").append(r.refVertices())
              .append(",\"vDet\":").append(r.detVertices())
              .append(",\"circ\":").append(fmt(r.circScore()))
              .append(",\"solid\":").append(fmt(r.solidScore()))
              .append(",\"gt\":{\"x\":").append(r.gtX()).append(",\"y\":").append(r.gtY())
              .append(",\"w\":").append(r.gtW()).append(",\"h\":").append(r.gtH()).append("}")
              .append(",\"det\":{\"x\":").append(r.detX()).append(",\"y\":").append(r.detY())
              .append(",\"w\":").append(r.detW()).append(",\"h\":").append(r.detH()).append("}")
              .append(",\"refSig\":\"").append(j(r.refSig()))
              .append("\",\"detSig\":\"").append(j(r.detSig())).append("\"")
              .append(",\"otherHits\":[");
            List<double[]> hits = r.otherHits();
            for (int h = 0; h < hits.size(); h++) {
                double[] e = hits.get(h);
                sb.append("{\"x\":").append((int)e[0]).append(",\"y\":").append((int)e[1])
                  .append(",\"w\":").append((int)e[2]).append(",\"h\":").append((int)e[3])
                  .append(",\"score\":").append(fmt(e[4])).append("}");
                if (h < hits.size() - 1) sb.append(",");
            }
            sb.append("]}")
              .append(i < rows.size() - 1 ? "," : "").append("\n");
        }
        sb.append("]\n");

        Files.writeString(outputDir.resolve("diagnostics.json"), sb.toString(), StandardCharsets.UTF_8);

        // ── stdout ────────────────────────────────────────────────────────
        System.out.printf("%n=== DIAGNOSTIC SUMMARY  total=%d  correct=%d  lowScore=%d  badIoU=%d  FP=%d  missed=%d ===%n",
            total, correctCount, lowCount, biouCount, fpCount, missedCount);
        printSection("FALSE POSITIVES",             rows.stream().filter(DiagRow::falsePositive).collect(Collectors.toList()));
        printSection("BAD IoU",                     rows.stream().filter(DiagRow::badIou).collect(Collectors.toList()));
        printSection("MISSED",                      rows.stream().filter(DiagRow::missed).collect(Collectors.toList()));
        printSection("CORRECT BUT LOW SCORE (<75%)",rows.stream().filter(DiagRow::lowScore).collect(Collectors.toList()));
        System.out.println("\n=== CORRECT HITS ===");
        rows.stream().filter(DiagRow::correctHit).forEach(r ->
            System.out.printf("  OK   %-18s %-25s score=%5.1f%% iou=%.2f%n",
                r.bg(), r.shape(), r.score(), Double.isNaN(r.iou()) ? 0 : r.iou()));
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private void appendSection(StringBuilder sb, String title, List<DiagRow> list, String tag) {
        sb.append("\n// ---- ").append(title).append(" ----\n");
        list.forEach(r -> sb.append(String.format(
            "// %s %-18s %-25s score=%5.1f%% iou=%.2f  det@(%d,%d,%d,%d)  gt@(%d,%d,%d,%d)  refType=%-20s detType=%-20s vRef=%d vDet=%d  circ=%.2f solid=%.2f%n",
            tag, r.bg(), r.shape(), r.score(), Double.isNaN(r.iou()) ? 0 : r.iou(),
            r.detX(), r.detY(), r.detW(), r.detH(),
            r.gtX(),  r.gtY(),  r.gtW(),  r.gtH(),
            r.refType(), r.detType(), r.refVertices(), r.detVertices(),
            r.circScore(), r.solidScore())));
    }

    private void printSection(String title, List<DiagRow> list) {
        System.out.println("\n=== " + title + " ===");
        list.forEach(r -> System.out.printf(
            "  %-4s %-18s %-25s score=%5.1f%% iou=%.2f  refType=%-20s detType=%-20s vRef=%d vDet=%d  circ=%.2f solid=%.2f%n",
            r.falsePositive() ? "FP" : r.badIou() ? "BIOU" : r.missed() ? "MISS" : "LOW ",
            r.bg(), r.shape(), r.score(), Double.isNaN(r.iou()) ? 0 : r.iou(),
            r.refType(), r.detType(), r.refVertices(), r.detVertices(),
            r.circScore(), r.solidScore()));
    }

    private static String fmt(double v) { return String.format("%.4f", v); }
    private static String j(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n");
    }

    // ── Static geometry / IoU helpers (reusable) ──────────────────────────────

    /**
     * Coverage-scaled IoU: {@code recall × max(1, detArea / gtArea)}.
     *
     * <p><b>Parameter convention:</b> {@code a} = detection bbox, {@code b} = ground-truth bbox.
     *
     * <ul>
     *   <li><b>1.0</b> — detection fits the GT exactly.</li>
     *   <li><b>&gt; 1.0</b> — detection is larger than GT (e.g. 1.20 = 20 % bigger).
     *       The scale factor only amplifies when the detection fully (or mostly) covers
     *       the GT, because recall is also high in that case.</li>
     *   <li><b>&lt; 1.0</b> — detection only covers part of the GT
     *       (e.g. 0.80 = covers 80 % of GT area).</li>
     *   <li><b>0.0</b> — no overlap at all.</li>
     * </ul>
     */
    public static double iou(Rect a, Rect b) {
        int ix1 = Math.max(a.x, b.x),  iy1 = Math.max(a.y, b.y);
        int ix2 = Math.min(a.x + a.width,  b.x + b.width);
        int iy2 = Math.min(a.y + a.height, b.y + b.height);
        if (ix2 <= ix1 || iy2 <= iy1) return 0.0;
        double inter   = (double)(ix2 - ix1) * (iy2 - iy1);
        double detArea = (double) a.width * a.height;
        double gtArea  = (double) b.width * b.height;
        double recall  = inter   / gtArea;                 // fraction of GT covered [0, 1]
        double scale   = Math.max(1.0, detArea / gtArea);  // ≥ 1.0 when det is bigger than GT
        return recall * scale;
    }

    public static Rect groundTruthRect(Mat shapeMat) {
        Mat grey = new Mat(), bin = new Mat();
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
            r = new Rect(Math.min(r.x,b.x), Math.min(r.y,b.y),
                         Math.max(r.x+r.width,b.x+b.width)   - Math.min(r.x,b.x),
                         Math.max(r.y+r.height,b.y+b.height) - Math.min(r.y,b.y));
        }
        return r;
    }


    public static Mat buildShapeMat(ReferenceId id) {
        return switch (id) {
            case CIRCLE_FILLED        -> circle();
            case RECT_FILLED          -> rect();
            case TRIANGLE_FILLED      -> triangle();
            case HEXAGON_OUTLINE      -> hexagon();
            case PENTAGON_FILLED      -> pentagon();
            case STAR_5_FILLED        -> star();
            case POLYLINE_DIAMOND     -> diamond();
            case POLYLINE_ARROW_RIGHT -> arrow();
            case ELLIPSE_H            -> ellipse();
            case OCTAGON_FILLED       -> octagon();
            case POLYLINE_PLUS_SHAPE  -> plus();
            case CONCAVE_ARROW_HEAD   -> concaveArrowhead();
            case LINE_CROSS           -> cross();
            case RECT_ROTATED_45      -> rot45Rect();
            // ── Extended 20 ───────────────────────────────────────────────
            case LINE_H                       -> lineH();
            case LINE_V                       -> lineV();
            case LINE_X                       -> lineX();
            case CIRCLE_OUTLINE               -> circleOutline();
            case ELLIPSE_V                    -> ellipseV();
            case RECT_OUTLINE                 -> rectOutline();
            case RECT_SQUARE                  -> rectSquare();
            case HEXAGON_FILLED               -> hexagonFilled();
            case STAR_5_OUTLINE               -> star5Outline();
            case HEPTAGON_OUTLINE             -> heptagonOutline();
            case POLYLINE_ARROW_LEFT          -> arrowLeft();
            case POLYLINE_CHEVRON             -> chevron();
            case POLYLINE_T_SHAPE             -> tShape();
            case ARC_HALF                     -> arcHalf();
            case ARC_QUARTER                  -> arcQuarter();
            case CONCAVE_MOON                 -> concaveMoon();
            case IRREGULAR_QUAD               -> irregularQuad();
            case COMPOUND_RECT_IN_CIRCLE      -> compoundRectInCircle();
            case COMPOUND_TRIANGLE_IN_CIRCLE  -> compoundTriangleInCircle();
            case CROSSHAIR                    -> crosshair();
            default                   -> Mat.zeros(480, 640, CvType.CV_8UC3);
        };
    }

    // ── Shape-mat helpers — all sized to match ReferenceImageFactory proportions ──
    // ReferenceImageFactory uses SIZE=128.  Shapes here are centred at (320,240)
    // and scaled ×3 from the 128×128 reference so the actual composited shape has
    // the same segment-length ratios as the reference, making it score higher than
    // background elements whose proportions differ.

    private static Mat circle() {
        // Ref: circle(64,64, SIZE/2-8=56).  ×3 → radius=168.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.circle(m, new Point(320, 240), 168, new Scalar(255,255,255), -1);
        return m;
    }
    private static Mat rect() {
        // Ref: rect((12,24)→(115,103)) = 103×79 (AR=1.304).  ×3 → 309×237.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.rectangle(m, new Point(166, 122), new Point(475, 359), new Scalar(255,255,255), -1);
        return m;
    }
    private static Mat triangle() {
        // Ref: trianglePoints(pad=10) → tip(64,10) br(118,118) bl(10,118).
        // Relative to centre (64,64): tip(0,-54) br(54,54) bl(-54,54).  ×3.5 centred at (320,240).
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.fillPoly(m, List.of(new MatOfPoint(
            new Point(320, 51), new Point(509, 429), new Point(131, 429))),
            new Scalar(255,255,255));
        return m;
    }
    private static Mat hexagon() {
        // Ref: drawRegularPolygon(6, r=54, start=-90°).  ×3 → r=162.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Point[] pts = new Point[6];
        for (int i = 0; i < 6; i++) {
            double a = Math.toRadians(-90 + 60.0 * i);
            pts[i] = new Point(320 + 162 * Math.cos(a), 240 + 162 * Math.sin(a));
        }
        Imgproc.polylines(m, List.of(new MatOfPoint(pts)), true, new Scalar(255,255,255), 3);
        return m;
    }
    private static Mat pentagon() {
        // Ref: drawRegularPolygon(5, r=54, filled, start=-90°).  ×3 → r=162.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Point[] pts = new Point[5];
        for (int i = 0; i < 5; i++) {
            double a = Math.toRadians(-90 + 72.0 * i);
            pts[i] = new Point(320 + 162 * Math.cos(a), 240 + 162 * Math.sin(a));
        }
        Imgproc.fillPoly(m, List.of(new MatOfPoint(pts)), new Scalar(255,255,255));
        return m;
    }
    private static Mat star() {
        // Ref: drawStar(5, outerR=54, innerR=54×0.4=21.6, filled, start=-90°).  ×3.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Point[] pts = new Point[10];
        for (int i = 0; i < 10; i++) {
            double a = Math.toRadians(-90 + 36.0 * i);
            double r = (i % 2 == 0) ? 162 : 65;
            pts[i] = new Point(320 + r * Math.cos(a), 240 + r * Math.sin(a));
        }
        Imgproc.fillPoly(m, List.of(new MatOfPoint(pts)), new Scalar(255,255,255));
        return m;
    }
    private static Mat diamond() {
        // Ref: drawDiamond(cx=64, r=54) — equal diagonals (square diamond).
        // Relative to (64,64): top(0,-54) right(54,0) bot(0,54) left(-54,0).  ×3.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.polylines(m, List.of(new MatOfPoint(
            new Point(320,  78), new Point(482, 240),
            new Point(320, 402), new Point(158, 240))),
            true, new Scalar(255, 255, 255), 3);
        return m;
    }
    private static Mat arrow() {
        // Reference proportions: hw=45, hh=20, headH=36 on 128×128 → AR = 90/72 = 1.25
        // Scaled ×3 centred at (320,240): hw=135, hh=60, headH=108 → AR = 270/216 = 1.25
        // Drawn as polylines (outline) to match ReferenceImageFactory.drawArrow()
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.polylines(m, List.of(new MatOfPoint(
                new Point(185, 180), new Point(320, 180), new Point(320, 132),
                new Point(455, 240),
                new Point(320, 348), new Point(320, 300),
                new Point(185, 300))),
                true, new Scalar(255, 255, 255), 3);
        return m;
    }
    private static Mat ellipse() {
        // Ref: ELLIPSE_H → ellipse(cx, Size(SIZE/2-8, SIZE/4-4)) = Size(56,28) outline.  ×3 → axes (168,84).
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.ellipse(m, new Point(320, 240), new Size(168, 84), 0, 0, 360,
                new Scalar(255, 255, 255), -1);
        return m;
    }
    private static Mat octagon() {
        // Ref: drawRegularPolygon(8, r=54, filled, start=-90+22.5=-67.5°).  ×3 → r=162.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Point[] pts = new Point[8];
        for (int i = 0; i < 8; i++) {
            double a = Math.toRadians(-90 + 45.0 * i);
            pts[i] = new Point(320 + 162 * Math.cos(a), 240 + 162 * Math.sin(a));
        }
        Imgproc.fillPoly(m, List.of(new MatOfPoint(pts)), new Scalar(255,255,255));
        return m;
    }
    private static Mat plus() {
        // Reference proportions: SIZE=128, ctr=44 → shape spans [16,112]×[16,112] = 96×96 (AR=1.0)
        //   half_total=48, half_arm=20, arm_width=40 (symmetric both axes)
        // Scaled ×2.5 centred at (320,240): half_total=120, half_arm=50 → 240×240 (AR=1.0)
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.fillPoly(m, List.of(new MatOfPoint(
                new Point(270, 120), new Point(370, 120),
                new Point(370, 190), new Point(440, 190),
                new Point(440, 290), new Point(370, 290),
                new Point(370, 360), new Point(270, 360),
                new Point(270, 290), new Point(200, 290),
                new Point(200, 190), new Point(270, 190))),
                new Scalar(255, 255, 255));
        return m;
    }
    private static Mat concaveArrowhead() {
        // Ref: drawArrowHead → tip(64,10) br(114,114) notch(64,92) bl(14,114).
        // Relative to centre (64,64): tip(0,-54) br(50,50) notch(0,28) bl(-50,50).  ×3.5.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.fillPoly(m, List.of(new MatOfPoint(
            new Point(320,  51), new Point(495, 415),
            new Point(320, 338), new Point(145, 415))),
            new Scalar(255,255,255));
        return m;
    }
    private static Mat cross() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.line(m, new Point(320,80),  new Point(320,400), new Scalar(255,255,255), 8);
        Imgproc.line(m, new Point(100,240), new Point(540,240), new Scalar(255,255,255), 8);
        return m;
    }
    private static Mat rot45Rect() {
        // Reference proportions: hw=48, hh=28 on 128×128 → AR = 96/56 = 1.714
        // Scaled ×2.5 centred at (320,240): hw=120, hh=70 → rect 240×140 (AR=1.714)
        // Drawn as polylines (outline) to match ReferenceImageFactory.drawRotatedRect().
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.polylines(m, List.of(new MatOfPoint(
                new Point(200, 170), new Point(440, 170),
                new Point(440, 310), new Point(200, 310))),
                true, new Scalar(255, 255, 255), 3);
        Point centre = new Point(m.cols() / 2.0, m.rows() / 2.0);
        Mat rotM = Imgproc.getRotationMatrix2D(centre, -45, 1.0);
        Mat dst  = Mat.zeros(m.size(), m.type());
        Imgproc.warpAffine(m, dst, rotM, m.size());
        rotM.release(); m.release();
        return dst;
    }

    // ── Extended shape-mat helpers (×3 from 128×128 ref, centred at 320,240) ─
    // Formula: screen(x,y) = (128 + 3*ref_x , 48 + 3*ref_y)

    private static Mat lineH() {
        // Ref: line (8,64)→(119,64), th=2.  ×3.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.line(m, new Point(152, 240), new Point(485, 240), new Scalar(255,255,255), 6);
        return m;
    }
    private static Mat lineV() {
        // Ref: line (64,8)→(64,119), th=2.  ×3.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.line(m, new Point(320, 72), new Point(320, 405), new Scalar(255,255,255), 6);
        return m;
    }
    private static Mat lineX() {
        // Ref: two diagonals (8,8)→(119,119) and (119,8)→(8,119), th=2.  ×3.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.line(m, new Point(152, 72),  new Point(485, 405), new Scalar(255,255,255), 6);
        Imgproc.line(m, new Point(485, 72),  new Point(152, 405), new Scalar(255,255,255), 6);
        return m;
    }
    private static Mat circleOutline() {
        // Ref: circle(centre, r=56, th=2).  ×3 → r=168.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.circle(m, new Point(320, 240), 168, new Scalar(255,255,255), 6);
        return m;
    }
    private static Mat ellipseV() {
        // Ref: ellipse(centre, Size(28,56), th=2).  ×3 → axes(84,168).
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.ellipse(m, new Point(320, 240), new Size(84, 168), 0, 0, 360,
                new Scalar(255,255,255), 6);
        return m;
    }
    private static Mat rectOutline() {
        // Ref: rect (12,24)→(115,103), th=2.  ×3 → (164,120)→(473,357).
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.rectangle(m, new Point(164, 120), new Point(473, 357),
                new Scalar(255,255,255), 6);
        return m;
    }
    private static Mat rectSquare() {
        // Ref: square (16,16)→(111,111), th=2.  ×3 → (176,96)→(461,381).
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.rectangle(m, new Point(176, 96), new Point(461, 381),
                new Scalar(255,255,255), 6);
        return m;
    }
    private static Mat hexagonFilled() {
        // Ref: drawRegularPolygon(6, r=54, filled, start=-90°).  ×3 → r=162.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Point[] pts = new Point[6];
        for (int i = 0; i < 6; i++) {
            double a = Math.toRadians(-90 + 60.0 * i);
            pts[i] = new Point(320 + 162 * Math.cos(a), 240 + 162 * Math.sin(a));
        }
        Imgproc.fillPoly(m, List.of(new MatOfPoint(pts)), new Scalar(255,255,255));
        return m;
    }
    private static Mat star5Outline() {
        // Ref: drawStar(5, outerR=54, innerR=21.6, outline, start=-90°).  ×3.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Point[] pts = new Point[10];
        for (int i = 0; i < 10; i++) {
            double a = Math.toRadians(-90 + 36.0 * i);
            double r = (i % 2 == 0) ? 162 : 65;
            pts[i] = new Point(320 + r * Math.cos(a), 240 + r * Math.sin(a));
        }
        Imgproc.polylines(m, List.of(new MatOfPoint(pts)), true, new Scalar(255,255,255), 3);
        return m;
    }
    private static Mat heptagonOutline() {
        // Ref: drawRegularPolygon(7, r=54, outline, start=-90°).  ×3 → r=162.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Point[] pts = new Point[7];
        for (int i = 0; i < 7; i++) {
            double a = Math.toRadians(-90 + 360.0 / 7 * i);
            pts[i] = new Point(320 + 162 * Math.cos(a), 240 + 162 * Math.sin(a));
        }
        Imgproc.polylines(m, List.of(new MatOfPoint(pts)), true, new Scalar(255,255,255), 3);
        return m;
    }
    private static Mat arrowLeft() {
        // Ref: drawArrow(false): cx=64,cy=64, hw=45,hh=20,headH=36.  ×3.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.polylines(m, List.of(new MatOfPoint(
                new Point(455, 180), new Point(320, 180), new Point(320, 132),
                new Point(185, 240),
                new Point(320, 348), new Point(320, 300),
                new Point(455, 300))),
                true, new Scalar(255,255,255), 3);
        return m;
    }
    private static Mat chevron() {
        // Ref: drawChevron pts (16,64),(64,16),(112,64),(112,112),(64,84),(16,112).  ×3.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.polylines(m, List.of(new MatOfPoint(
                new Point(176, 240), new Point(320,  96), new Point(464, 240),
                new Point(464, 384), new Point(320, 300), new Point(176, 384))),
                true, new Scalar(255,255,255), 3);
        return m;
    }
    private static Mat tShape() {
        // Ref: drawTShape pts (16,16),(112,16),(112,44),(76,44),(76,112),(52,112),(52,44),(16,44). ×3.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.polylines(m, List.of(new MatOfPoint(
                new Point(176,  96), new Point(464,  96), new Point(464, 180),
                new Point(356, 180), new Point(356, 384), new Point(284, 384),
                new Point(284, 180), new Point(176, 180))),
                true, new Scalar(255,255,255), 3);
        return m;
    }
    private static Mat arcHalf() {
        // Ref: ellipse(centre, Size(54,54), 0, 0°→180°, th=2).  ×3 → r=162.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.ellipse(m, new Point(320, 240), new Size(162, 162), 0, 0, 180,
                new Scalar(255,255,255), 6);
        return m;
    }
    private static Mat arcQuarter() {
        // Ref: ellipse(centre, Size(54,54), 0, 0°→90°, th=2).  ×3 → r=162.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.ellipse(m, new Point(320, 240), new Size(162, 162), 0, 0, 90,
                new Scalar(255,255,255), 6);
        return m;
    }
    private static Mat concaveMoon() {
        // Ref: filled circle(r=54) then black-erase offset circle(centre=(79,64), r=50). ×3.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.circle(m, new Point(320, 240), 162, new Scalar(255,255,255), -1);
        Imgproc.circle(m, new Point(365, 240), 150, new Scalar(0,0,0),       -1);
        return m;
    }
    private static Mat irregularQuad() {
        // Ref: drawIrregularQuad pts (18,22),(102,14),(112,96),(30,110).  ×3.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.polylines(m, List.of(new MatOfPoint(
                new Point(182, 114), new Point(434,  90),
                new Point(464, 336), new Point(218, 378))),
                true, new Scalar(255,255,255), 3);
        return m;
    }
    private static Mat compoundRectInCircle() {
        // Ref: circle(r=54) + rect((28,28)→(99,99)).  ×3.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.circle(m, new Point(320, 240), 162, new Scalar(255,255,255), 3);
        Imgproc.rectangle(m, new Point(212, 132), new Point(425, 345),
                new Scalar(255,255,255), 3);
        return m;
    }
    private static Mat compoundTriangleInCircle() {
        // Ref: circle(r=54) + triangle outline (tip(0,-54), br(54,54), bl(-54,54)).  ×3.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.circle(m, new Point(320, 240), 162, new Scalar(255,255,255), 3);
        Imgproc.polylines(m, List.of(new MatOfPoint(
                new Point(320, 78), new Point(482, 402), new Point(158, 402))),
                true, new Scalar(255,255,255), 3);
        return m;
    }
    private static Mat crosshair() {
        // Ref: H+V lines th=1, filled circle r=3.  ×3 → th=3, r=9.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.line(m, new Point(152, 240), new Point(485, 240), new Scalar(255,255,255), 3);
        Imgproc.line(m, new Point(320,  72), new Point(320, 405), new Scalar(255,255,255), 3);
        Imgproc.circle(m, new Point(320, 240), 9, new Scalar(255,255,255), -1);
        return m;
    }

    // ── Fast-path recorder — uses pre-computed results ────────────────────────

    /**
     * Records a diagnostic row from <em>pre-computed</em> matcher results.
     * Does NOT rebuild the scene or re-run the matcher.
     *
     * @param bgId          background identifier
     * @param bgLabel       human-readable label for the background / test stage
     * @param refId         reference shape that was matched
     * @param results       pre-computed matcher results
     * @param gt            ground-truth bounding rect derived from a clean (black-BG)
     *                      version of the scene; may be {@code null} for negative scenes
     * @param passThreshold min score (%) to count as a detection
     * @param targetScore   score (%) above which a correct hit is "good"
     * @param goodIou       perfect-match IoU value (1.0 with coverage-scaled formula)
     * @return the recorded {@link DiagRow}
     */
    public DiagRow recordResult(BackgroundId bgId, String bgLabel, ReferenceId refId,
                                List<AnalysisResult> results, Rect gt,
                                double passThreshold, double targetScore, double goodIou) {
        AnalysisResult result = results.stream()
                .filter(r -> r.methodName().equals(VectorVariant.VECTOR_NORMAL.variantName()))
                .findFirst().orElse(results.isEmpty() ? null : results.get(0));

        double scorePercent = result != null ? result.matchScorePercent() : 0.0;
        Rect   bestBbox     = result != null ? result.boundingRect()      : null;

        double  iouVal        = Double.NaN;
        boolean falsePositive = false;
        boolean badIou        = false;
        boolean correctHit    = false;
        boolean missed        = false;

        final double iouThreshold = goodIou * 0.95;
        final double fpGate       = 60.0;

        if (gt != null) {
            if (bestBbox != null) {
                iouVal        = iou(bestBbox, gt);
                falsePositive = (scorePercent >= fpGate)        && (iouVal < 0.3);
                badIou        = (scorePercent >= passThreshold) && (iouVal >= 0.3) && (iouVal < iouThreshold);
                correctHit    = (scorePercent >= passThreshold) && (iouVal >= iouThreshold);
            }
            if (scorePercent < passThreshold) missed = true;
        } else {
            falsePositive = (scorePercent >= fpGate);
        }
        boolean lowScore = correctHit && (scorePercent < targetScore);

        DiagRow row = new DiagRow(
            bgLabel, refId != null ? refId.name() : "?",
            scorePercent, iouVal,
            falsePositive, badIou, correctHit, lowScore, missed,
            gt != null ? gt.x      : -1, gt != null ? gt.y       : -1,
            gt != null ? gt.width  : -1, gt != null ? gt.height  : -1,
            bestBbox != null ? bestBbox.x      : -1, bestBbox != null ? bestBbox.y       : -1,
            bestBbox != null ? bestBbox.width  : -1, bestBbox != null ? bestBbox.height  : -1,
            "n/a", "n/a",   // refSig / detSig — not recomputed in fast path
            0.0, 0.0, 0.0,  // circScore / solidScore / totalSim
            -1, -1,         // refVertices / detVertices
            "n/a", "n/a",   // refType / detType
            List.of()       // otherHits
        );
        rows.add(row);
        return row;
    }

    /** Scores every colour-cluster contour against the given ref signatures.
     *  Each entry: {@code [x, scoreFraction, y, w, h]} (indexed for sort by [1]). */
    public static List<double[]> allScoredBboxes(Mat scene, List<VectorSignature> refSigs) {
        List<double[]> hits = new ArrayList<>();
        if (refSigs == null || refSigs.isEmpty()) return hits;
        double area = (double) scene.rows() * scene.cols();
        double eps  = VectorVariant.VECTOR_NORMAL.epsilonFactor();
        List<ColourCluster> clusters = SceneColourClusters.extract(scene);
        for (ColourCluster c : clusters) {
            List<MatOfPoint> contours = SceneDescriptor.contoursFromMask(c.mask);
            double maxA = contours.stream()
                    .mapToDouble(cnt -> { Rect r = Imgproc.boundingRect(cnt); return (double)r.width*r.height; })
                    .max().orElse(1);
            long sig = contours.stream()
                    .filter(cnt -> { Rect r = Imgproc.boundingRect(cnt); return (double)r.width*r.height >= maxA*0.20; })
                    .count();
            double penalty = sig > 1 ? 1.0 / (Math.log(sig + 1) / Math.log(2)) : 1.0;
            for (MatOfPoint cnt : contours) {
                VectorSignature vs = VectorSignature.buildFromContour(cnt, eps, area);
                // Score against every ref sig, take best
                double bestSim = 0.0;
                for (VectorSignature refSig : refSigs) {
                    double s = refSig.similarity(vs);
                    if (s > bestSim) bestSim = s;
                }
                double penalised = bestSim * penalty;
                if (penalised > 0.01) {
                    Rect bb = Imgproc.boundingRect(cnt);
                    hits.add(new double[]{bb.x, penalised, bb.y, bb.width, bb.height});
                }
            }
            c.release();
        }
        return hits;
    }
}


