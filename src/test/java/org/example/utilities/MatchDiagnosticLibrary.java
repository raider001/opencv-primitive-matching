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
import java.util.concurrent.CopyOnWriteArrayList;
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

    private final List<DiagRow> rows = new CopyOnWriteArrayList<>();

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
              .append(",\"layer1_boundary\":").append(fmt(r.circScore()))
              .append(",\"layer2_structural\":").append(fmt(r.solidScore()))
              .append(",\"layer3_geometry\":").append(fmt(r.totalSim()))
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
            "// %s %-18s %-25s score=%5.1f%% iou=%.2f  det@(%d,%d,%d,%d)  gt@(%d,%d,%d,%d)  refType=%-20s detType=%-20s vRef=%d vDet=%d  L1=%.1f L2=%.1f L3=%.1f%n",
            tag, r.bg(), r.shape(), r.score(), Double.isNaN(r.iou()) ? 0 : r.iou(),
            r.detX(), r.detY(), r.detW(), r.detH(),
            r.gtX(),  r.gtY(),  r.gtW(),  r.gtH(),
            r.refType(), r.detType(), r.refVertices(), r.detVertices(),
            r.circScore(), r.solidScore(), r.totalSim())));
    }

    private void printSection(String title, List<DiagRow> list) {
        System.out.println("\n=== " + title + " ===");
        list.forEach(r -> System.out.printf(
            "  %-4s %-18s %-25s score=%5.1f%% iou=%.2f  refType=%-20s detType=%-20s vRef=%d vDet=%d  L1=%.1f L2=%.1f L3=%.1f%n",
            r.falsePositive() ? "FP" : r.badIou() ? "BIOU" : r.missed() ? "MISS" : "LOW ",
            r.bg(), r.shape(), r.score(), Double.isNaN(r.iou()) ? 0 : r.iou(),
            r.refType(), r.detType(), r.refVertices(), r.detVertices(),
            r.circScore(), r.solidScore(), r.totalSim()));
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
            // ── New 20 shapes ─────────────────────────────────────────────
            case TRIANGLE_RIGHT               -> triangleRight();
            case TRAPEZOID                    -> trapezoid();
            case KITE                         -> kite();
            case STAR_8_OUTLINE               -> star8Outline();
            case NONAGON_OUTLINE              -> nonagonOutline();
            case DECAGON_OUTLINE              -> decagonOutline();
            case POLYLINE_U_SHAPE             -> uShape();
            case POLYLINE_H_SHAPE             -> hShape();
            case POLYLINE_Z_SHAPE             -> zShape();
            case POLYLINE_CARET               -> caret();
            case POLYLINE_S_CURVE             -> sCurve();
            case ARC_NEAR_FULL                -> arcNearFull();
            case CONCAVE_LIGHTNING            -> lightning();
            case RECT_THIN_TALL               -> rectThinTall();
            case CIRCLE_DONUT                 -> donut();
            case COMPOUND_STAR_IN_CIRCLE      -> compoundStarInCircle();
            case COMPOUND_DIAMOND_IN_RECT     -> compoundDiamondInRect();
            case GRID_3X3                     -> grid3x3();
            case GRID_DIAGONAL                -> gridDiagonal();
            case RECT_ROTATED_75              -> rectRotated75();
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

    // ── New shape helpers (20 shapes, all ×3 from 128×128 ref, centred at 320,240) ──

    private static Mat triangleRight() {
        // Ref: (12,116)→(12,12)→(116,116), th=2.  ×3 → (164,396)→(164,84)→(476,396).
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.polylines(m, List.of(new MatOfPoint(
                new Point(164, 396), new Point(164, 84), new Point(476, 396))),
                true, new Scalar(255,255,255), 6);
        return m;
    }
    private static Mat trapezoid() {
        // Ref: (32,20)→(95,20)→(114,116)→(14,116), th=2.  ×3.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.polylines(m, List.of(new MatOfPoint(
                new Point(224, 108), new Point(413, 108),
                new Point(470, 396), new Point(170, 396))),
                true, new Scalar(255,255,255), 6);
        return m;
    }
    private static Mat kite() {
        // Ref: top(64,10) right(114,80) bot(64,116) left(14,80), th=2.  ×3.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.polylines(m, List.of(new MatOfPoint(
                new Point(320, 78), new Point(470, 288),
                new Point(320, 396), new Point(170, 288))),
                true, new Scalar(255,255,255), 6);
        return m;
    }
    private static Mat star8Outline() {
        // Ref: drawStar(8, outerR=54, innerR=21.6, outline, start=-90°).  ×3.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        double outerR = 162, innerR = 65;
        Point[] pts = new Point[16];
        for (int i = 0; i < 16; i++) {
            double a = Math.toRadians(-90 + (180.0 / 8) * i);
            double r = (i % 2 == 0) ? outerR : innerR;
            pts[i] = new Point(320 + r * Math.cos(a), 240 + r * Math.sin(a));
        }
        Imgproc.polylines(m, List.of(new MatOfPoint(pts)), true, new Scalar(255,255,255), 6);
        return m;
    }
    private static Mat nonagonOutline() {
        // Ref: drawRegularPolygon(9, r=54, outline, start=-90°).  ×3 → r=162.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Point[] pts = new Point[9];
        for (int i = 0; i < 9; i++) {
            double a = Math.toRadians(-90 + 360.0 / 9 * i);
            pts[i] = new Point(320 + 162 * Math.cos(a), 240 + 162 * Math.sin(a));
        }
        Imgproc.polylines(m, List.of(new MatOfPoint(pts)), true, new Scalar(255,255,255), 6);
        return m;
    }
    private static Mat decagonOutline() {
        // Ref: drawRegularPolygon(10, r=54, outline, start=-90°).  ×3 → r=162.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Point[] pts = new Point[10];
        for (int i = 0; i < 10; i++) {
            double a = Math.toRadians(-90 + 36.0 * i);
            pts[i] = new Point(320 + 162 * Math.cos(a), 240 + 162 * Math.sin(a));
        }
        Imgproc.polylines(m, List.of(new MatOfPoint(pts)), true, new Scalar(255,255,255), 6);
        return m;
    }
    private static Mat uShape() {
        // Ref: U-shape (16,16)→(44,16)→(44,80)→(84,80)→(84,16)→(112,16)→(112,112)→(16,112), th=2.  ×3.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.polylines(m, List.of(new MatOfPoint(
                new Point(176,  96), new Point(260,  96), new Point(260, 288),
                new Point(380, 288), new Point(380,  96), new Point(464,  96),
                new Point(464, 384), new Point(176, 384))),
                true, new Scalar(255,255,255), 6);
        return m;
    }
    private static Mat hShape() {
        // Ref: H-shape 12 vertices, th=2.  ×3.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.polylines(m, List.of(new MatOfPoint(
                new Point(176,  96), new Point(260,  96), new Point(260, 198),
                new Point(380, 198), new Point(380,  96), new Point(464,  96),
                new Point(464, 384), new Point(380, 384), new Point(380, 282),
                new Point(260, 282), new Point(260, 384), new Point(176, 384))),
                true, new Scalar(255,255,255), 6);
        return m;
    }
    private static Mat zShape() {
        // Ref: Z-shape 10 vertices, th=2.  ×3.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.polylines(m, List.of(new MatOfPoint(
                new Point(176,  96), new Point(464,  96), new Point(464, 180),
                new Point(248, 300), new Point(464, 300), new Point(464, 384),
                new Point(176, 384), new Point(176, 300), new Point(392, 180),
                new Point(176, 180))),
                true, new Scalar(255,255,255), 6);
        return m;
    }
    private static Mat caret() {
        // Ref: caret (16,108)→(16,82)→(64,16)→(112,82)→(112,108), th=2.  ×3.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.polylines(m, List.of(new MatOfPoint(
                new Point(176, 372), new Point(176, 294),
                new Point(320,  96),
                new Point(464, 294), new Point(464, 372))),
                true, new Scalar(255,255,255), 6);
        return m;
    }
    private static Mat sCurve() {
        // Ref: S-curve ~33 points, open polyline, th=2.  ×3 → th=6.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        int r = 90, steps = 16;
        double cx1 = 275, cy1 = 162, cx2 = 365, cy2 = 318;
        List<Point> pts = new ArrayList<>();
        for (int i = 0; i <= steps; i++) {
            double a = Math.toRadians(180.0 + 180.0 * i / steps);
            pts.add(new Point(cx1 + r * Math.cos(a), cy1 + r * Math.sin(a)));
        }
        for (int i = 0; i <= steps; i++) {
            double a = Math.toRadians(180.0 * i / steps);
            pts.add(new Point(cx2 + r * Math.cos(a), cy2 + r * Math.sin(a)));
        }
        MatOfPoint mop = new MatOfPoint();
        mop.fromList(pts);
        Imgproc.polylines(m, List.of(mop), false, new Scalar(255,255,255), 6);
        return m;
    }
    private static Mat arcNearFull() {
        // Ref: ellipse(centre, Size(54,54), 0, 20°→350°, th=2).  ×3 → r=162, th=6.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.ellipse(m, new Point(320, 240), new Size(162, 162), 0, 20, 350,
                new Scalar(255,255,255), 6);
        return m;
    }
    private static Mat lightning() {
        // Ref: lightning filled 6-vertex polygon, th=2 outline.  ×3.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.fillPoly(m, List.of(new MatOfPoint(
                new Point(356,  78), new Point(236, 246),
                new Point(314, 228), new Point(212, 402),
                new Point(380, 234), new Point(302, 252))),
                new Scalar(255,255,255));
        return m;
    }
    private static Mat rectThinTall() {
        // Ref: (48,8)→(79,119), th=2.  ×3 → (272,72)→(365,405), th=6.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.rectangle(m, new Point(272, 72), new Point(365, 405),
                new Scalar(255,255,255), 6);
        return m;
    }
    private static Mat donut() {
        // Ref: circle(r=56) filled, then circle(r=36) erased black.  ×3 → outer=168, inner=108.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.circle(m, new Point(320, 240), 168, new Scalar(255,255,255), -1);
        Imgproc.circle(m, new Point(320, 240), 108, new Scalar(0,0,0),       -1);
        return m;
    }
    private static Mat compoundStarInCircle() {
        // Ref: circle(r=54, th=2) + star(5, outerR=54, innerR=21.6, outline, th=2).  ×3.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.circle(m, new Point(320, 240), 162, new Scalar(255,255,255), 6);
        Point[] pts = new Point[10];
        for (int i = 0; i < 10; i++) {
            double a = Math.toRadians(-90 + 36.0 * i);
            double r = (i % 2 == 0) ? 162 : 65;
            pts[i] = new Point(320 + r * Math.cos(a), 240 + r * Math.sin(a));
        }
        Imgproc.polylines(m, List.of(new MatOfPoint(pts)), true, new Scalar(255,255,255), 6);
        return m;
    }
    private static Mat compoundDiamondInRect() {
        // Ref: rect((12,12)→(115,115), th=2) + diamond(r=54, th=2).  ×3.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.rectangle(m, new Point(164, 164), new Point(473, 473),
                new Scalar(255,255,255), 6);
        Imgproc.polylines(m, List.of(new MatOfPoint(
                new Point(320,  78), new Point(482, 240),
                new Point(320, 402), new Point(158, 240))),
                true, new Scalar(255,255,255), 6);
        return m;
    }
    private static Mat grid3x3() {
        // Ref: 3×3 grid (4H + 4V lines), th=1.  ×3 → th=3, line spacing ×3.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Scalar w = new Scalar(255,255,255);
        int[] xs = {128, 254, 380, 509};
        int[] ys = { 48, 174, 300, 429};
        for (int x : xs) Imgproc.line(m, new Point(x,  48), new Point(x, 429), w, 3);
        for (int y : ys) Imgproc.line(m, new Point(128, y), new Point(509, y), w, 3);
        return m;
    }
    private static Mat gridDiagonal() {
        // Ref: diagonal grid (step=22), th=1.  ×3 → step=66, th=3.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Scalar w = new Scalar(255,255,255);
        // 45° lines
        for (int c = -479; c < 640; c += 66) {
            int x0, y0, x1, y1;
            if (c >= 0) { x0 = 0; y0 = c; x1 = Math.min(639, 639-c+479); y1 = 479; }
            else        { x0 = -c; y0 = 0; x1 = 639; y1 = Math.min(479, 639+c); }
            if (x0 < 640 && y0 < 480 && x1 >= 0 && y1 >= 0)
                Imgproc.line(m, new Point(x0, y0), new Point(x1, y1), w, 3);
        }
        // 135° lines
        for (int c = 0; c < 1119; c += 66) {
            int x0, y0, x1, y1;
            if (c <= 479) { x0 = 0; y0 = c; x1 = Math.min(c, 639); y1 = Math.max(0, c-639); }
            else          { x0 = c-479; y0 = 479; x1 = 639; y1 = Math.max(0, c-639); }
            if (x0 < 640 && y0 >= 0 && x1 >= 0 && y1 < 480)
                Imgproc.line(m, new Point(x0, y0), new Point(x1, y1), w, 3);
        }
        return m;
    }
    private static Mat rectRotated75() {
        // Ref: drawRotatedRect(75, hw=48, hh=28), th=2.  ×3 → hw=144, hh=84, th=6.
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.polylines(m, List.of(new MatOfPoint(
                new Point(176, 156), new Point(464, 156),
                new Point(464, 324), new Point(176, 324))),
                true, new Scalar(255,255,255), 6);
        Point centre = new Point(m.cols() / 2.0, m.rows() / 2.0);
        Mat rotM = Imgproc.getRotationMatrix2D(centre, -75, 1.0);
        Mat dst  = Mat.zeros(m.size(), m.type());
        Imgproc.warpAffine(m, dst, rotM, m.size());
        rotM.release(); m.release();
        return dst;
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
        final double iouUpperCap  = 1.3;   // detection area must not exceed 1.3× GT area
        final double fpGate       = 60.0;

        if (gt != null) {
            if (bestBbox != null) {
                iouVal        = iou(bestBbox, gt);
                falsePositive = (scorePercent >= fpGate)        && (iouVal < 0.3);
                badIou        = (scorePercent >= passThreshold) && (iouVal >= 0.3) && (iouVal < iouThreshold);
                correctHit    = (scorePercent >= passThreshold) && (iouVal >= iouThreshold) && (iouVal <= iouUpperCap);
                // Over-expanded detection — bbox area exceeds 2× GT area
                if (!badIou && (scorePercent >= passThreshold) && (iouVal > iouUpperCap)) {
                    badIou = true;
                    correctHit = false;
                }
            }
            if (scorePercent < passThreshold) missed = true;
        } else {
            falsePositive = (scorePercent >= fpGate);
        }
        boolean lowScore = correctHit && (scorePercent < targetScore);

        // Extract scoring layers from AnalysisResult
        double layer1 = result != null && result.scoringLayers() != null 
                        ? result.scoringLayers().boundaryCount() : 0.0;
        double layer2 = result != null && result.scoringLayers() != null 
                        ? result.scoringLayers().structural() : 0.0;
        double layer3 = result != null && result.scoringLayers() != null 
                        ? result.scoringLayers().geometry() : 0.0;

        DiagRow row = new DiagRow(
            bgLabel, refId != null ? refId.name() : "?",
            scorePercent, iouVal,
            falsePositive, badIou, correctHit, lowScore, missed,
            gt != null ? gt.x      : -1, gt != null ? gt.y       : -1,
            gt != null ? gt.width  : -1, gt != null ? gt.height  : -1,
            bestBbox != null ? bestBbox.x      : -1, bestBbox != null ? bestBbox.y       : -1,
            bestBbox != null ? bestBbox.width  : -1, bestBbox != null ? bestBbox.height  : -1,
            "n/a", "n/a",   // refSig / detSig — not recomputed in fast path
            layer1, layer2, layer3,  // Layer 1: boundaryCount, Layer 2: structural, Layer 3: geometry
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


