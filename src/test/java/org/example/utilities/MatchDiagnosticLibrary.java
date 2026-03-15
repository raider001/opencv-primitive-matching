package org.example.utilities;

import org.example.analytics.AnalysisResult;
import org.example.analytics.AnalysisResult;
import org.example.colour.ColourCluster;
import org.example.colour.SceneColourClusters;
import org.example.factories.BackgroundFactory;
import org.example.factories.BackgroundId;
import org.example.factories.ReferenceId;
import org.example.factories.ReferenceImageFactory;
import org.example.matchers.SceneDescriptor;
import org.example.matchers.VectorMatcher;
import org.example.matchers.VectorSignature;
import org.example.matchers.VectorVariant;
import org.example.scene.SceneCategory;
import org.example.scene.SceneEntry;
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
 * {@link #evaluate(BackgroundId, String, ReferenceId, double, double, double, Path)}
 * then writes a combined JSON + stdout report via {@link #writeReport(Path)}.
 *
 * <p>Usage:
 * <pre>
 *   MatchDiagnosticLibrary diag = new MatchDiagnosticLibrary();
 *   diag.evaluate(BackgroundId.BG_SOLID_WHITE, "solid-white",
 *                 ReferenceId.CIRCLE_FILLED, 40.0, 75.0, 0.5, OUTPUT);
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
        /** All other contour hits above 1% similarity: each entry [x, y, w, h, scorePct]. */
        List<double[]> otherHits
    ) {}

    private final List<DiagRow> rows = new ArrayList<>();

    /** Clear all accumulated rows (call in @BeforeAll to reset between runs). */
    public void clear() { rows.clear(); }

    /** Read-only view of accumulated rows. */
    public List<DiagRow> rows() { return Collections.unmodifiableList(rows); }

    // ── Core evaluation ───────────────────────────────────────────────────────

    /**
     * Runs VectorMatcher on a composited scene and accumulates one {@link DiagRow}.
     *
     * @param bgId          background to composite the shape onto
     * @param bgLabel       human-readable background label
     * @param refId         reference shape to match
     * @param passThreshold min score (%) to count as a detection
     * @param targetScore   score (%) above which a correct hit is "good"
     * @param goodIou       IoU above which location is correct
     * @param outputDir     directory for matcher artefacts
     */
    public DiagRow evaluate(BackgroundId bgId, String bgLabel, ReferenceId refId,
                            double passThreshold, double targetScore, double goodIou,
                            Path outputDir) {
        Mat shapeMat = buildShapeMat(refId);
        Mat ref      = ReferenceImageFactory.build(refId);
        Rect gt      = groundTruthRect(shapeMat);
        Mat scene    = compositeOnBackground(shapeMat, bgId);

        SceneEntry sceneEntry = new SceneEntry(
                refId, SceneCategory.A_CLEAN, bgLabel, bgId,
                Collections.emptyList(), scene);

        List<AnalysisResult> results = VectorMatcher.match(
                refId, ref, sceneEntry, Collections.emptySet(), outputDir);

        AnalysisResult result = results.stream()
                .filter(r -> r.methodName().equals(VectorVariant.VECTOR_NORMAL.variantName()))
                .findFirst().orElse(results.isEmpty() ? null : results.get(0));

        double scorePercent = result != null ? result.matchScorePercent() : 0.0;
        Rect   bestBbox     = result != null ? result.boundingRect()      : null;

        double  iou           = Double.NaN;
        boolean falsePositive = false;
        boolean badIou        = false;
        boolean correctHit    = false;
        boolean missed        = false;

        // ── Pass / fail rules ─────────────────────────────────────────────
        // Positive scene (reference present): PASS when IoU ≥ 95 % of target.
        // Negative scene (reference absent):  PASS when score < 60 %;
        //                                     FAIL (false alarm) when score ≥ 60 %.
        final double iouThreshold = goodIou * 0.95;  // e.g. 0.475 when goodIou = 0.5
        final double fpGate       = 60.0;            // score above which a negative detection fails

        if (gt != null) {
            // Positive scene — reference is present
            if (bestBbox != null) {
                iou           = iou(bestBbox, gt);
                falsePositive = (scorePercent >= fpGate)        && (iou < 0.3);
                badIou        = (scorePercent >= passThreshold) && (iou >= 0.3) && (iou < iouThreshold);
                correctHit    = (scorePercent >= passThreshold) && (iou >= iouThreshold);
            }
            if (scorePercent < passThreshold) missed = true;
        } else {
            // Negative scene — reference is absent; any score ≥ fpGate is a false alarm
            falsePositive = (scorePercent >= fpGate);
        }
        boolean lowScore = correctHit && (scorePercent < targetScore);

        // Derive scene sig for reporting
        double eps       = VectorVariant.VECTOR_NORMAL.epsilonFactor();
        double sceneArea = (double) scene.rows() * scene.cols();
        VectorSignature refSig      = VectorMatcher.buildRefSignature(ref, eps);
        VectorSignature bestSceneSig = null;
        if (bestBbox != null) {
            outer:
            for (SceneDescriptor.ClusterContours cc : sceneEntry.descriptor().clusters()) {
                for (MatOfPoint c : cc.contours) {
                    if (Imgproc.boundingRect(c).equals(bestBbox)) {
                        bestSceneSig = VectorSignature.buildFromContour(c, eps, sceneArea);
                        break outer;
                    }
                }
            }
        }

        double circScore  = 0, solidScore = 0;
        if (refSig != null && bestSceneSig != null) {
            circScore  = 1.0 - Math.abs(refSig.circularity  - bestSceneSig.circularity);
            solidScore = 1.0 - Math.abs(refSig.solidity     - bestSceneSig.solidity);
        }

        // ── Collect all other hits above 1% ───────────────────────────────
        List<double[]> otherHits = allScoredBboxes(scene, refSig).stream()
                .filter(e -> e[1] > 0.01)
                .filter(e -> bestBbox == null || !(e[0] == bestBbox.x && e[2] == bestBbox.y
                             && e[3] == bestBbox.width && e[4] == bestBbox.height))
                .map(e -> new double[]{e[0], e[2], e[3], e[4], e[1] * 100.0}) // x,y,w,h,scorePct
                .sorted(Comparator.comparingDouble(e -> -e[4]))
                .collect(Collectors.toList());

        DiagRow row = new DiagRow(
            bgLabel, refId.name(),
            scorePercent, iou,
            falsePositive, badIou, correctHit, lowScore, missed,
            gt      != null ? gt.x          : -1, gt      != null ? gt.y           : -1,
            gt      != null ? gt.width      : -1, gt      != null ? gt.height      : -1,
            bestBbox != null ? bestBbox.x   : -1, bestBbox != null ? bestBbox.y    : -1,
            bestBbox != null ? bestBbox.width: -1, bestBbox != null ? bestBbox.height: -1,
            refSig       != null ? refSig.toString()       : "null",
            bestSceneSig != null ? bestSceneSig.toString() : "null",
            circScore, solidScore,
            refSig != null && bestSceneSig != null ? refSig.similarity(bestSceneSig) : 0,
            refSig       != null ? refSig.vertexCount       : -1,
            bestSceneSig != null ? bestSceneSig.vertexCount : -1,
            refSig       != null ? refSig.type.name()       : "null",
            bestSceneSig != null ? bestSceneSig.type.name() : "null",
            otherHits
        );

        rows.add(row);

        sceneEntry.release();
        shapeMat.release();
        ref.release();
        scene.release();
        return row;
    }

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

    public static Mat compositeOnBackground(Mat shapeMat, BackgroundId bgId) {
        Mat scene = BackgroundFactory.build(bgId, shapeMat.cols(), shapeMat.rows());
        Mat grey  = new Mat(); Imgproc.cvtColor(scene, grey, Imgproc.COLOR_BGR2GRAY);
        double bgLuma = Core.mean(grey).val[0]; grey.release();
        Mat foreground = shapeMat.clone();
        if (bgLuma > 100) Core.bitwise_not(foreground, foreground);
        Mat maskGrey = new Mat(), mask = new Mat();
        Imgproc.cvtColor(shapeMat, maskGrey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(maskGrey, mask, 5, 255, Imgproc.THRESH_BINARY);
        maskGrey.release();
        foreground.copyTo(scene, mask);
        foreground.release(); mask.release();
        return scene;
    }

    public static Mat buildShapeMat(ReferenceId id) {
        return switch (id) {
            case CIRCLE_FILLED        -> circle(320, 240, 60);
            case RECT_FILLED          -> rect(230, 160, 410, 320);
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
            default                   -> Mat.zeros(480, 640, CvType.CV_8UC3);
        };
    }

    private static Mat circle(int cx, int cy, int r) {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.circle(m, new Point(cx, cy), r, new Scalar(255,255,255), -1);
        return m;
    }
    private static Mat rect(int x1, int y1, int x2, int y2) {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.rectangle(m, new Point(x1,y1), new Point(x2,y2), new Scalar(255,255,255), -1);
        return m;
    }
    private static Mat triangle() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.fillPoly(m, List.of(new MatOfPoint(
            new Point(320,130), new Point(180,350), new Point(460,350))),
            new Scalar(255,255,255));
        return m;
    }
    private static Mat hexagon() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Point[] pts = new Point[6];
        for (int i = 0; i < 6; i++) {
            double a = Math.toRadians(60*i-30);
            pts[i] = new Point(320+80*Math.cos(a), 240+80*Math.sin(a));
        }
        Imgproc.fillPoly(m, List.of(new MatOfPoint(pts)), new Scalar(255,255,255));
        return m;
    }
    private static Mat pentagon() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Point[] pts = new Point[5];
        for (int i = 0; i < 5; i++) {
            double a = Math.toRadians(72*i-90);
            pts[i] = new Point(320+90*Math.cos(a), 240+90*Math.sin(a));
        }
        Imgproc.fillPoly(m, List.of(new MatOfPoint(pts)), new Scalar(255,255,255));
        return m;
    }
    private static Mat star() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Point[] pts = new Point[10];
        for (int i = 0; i < 10; i++) {
            double a = Math.toRadians(36*i-90);
            int r = (i%2==0) ? 100 : 40;
            pts[i] = new Point(320+r*Math.cos(a), 240+r*Math.sin(a));
        }
        Imgproc.fillPoly(m, List.of(new MatOfPoint(pts)), new Scalar(255,255,255));
        return m;
    }
    private static Mat diamond() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.fillPoly(m, List.of(new MatOfPoint(
            new Point(320,110), new Point(470,240),
            new Point(320,370), new Point(170,240))),
            new Scalar(255,255,255));
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
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.ellipse(m, new Point(320,240), new Size(140,70), 0, 0, 360, new Scalar(255,255,255), -1);
        return m;
    }
    private static Mat octagon() {
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Point[] pts = new Point[8];
        for (int i = 0; i < 8; i++) {
            double a = Math.toRadians(45*i-22.5);
            pts[i] = new Point(320+85*Math.cos(a), 240+85*Math.sin(a));
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
        Mat m = Mat.zeros(480, 640, CvType.CV_8UC3);
        Imgproc.fillPoly(m, List.of(new MatOfPoint(
            new Point(320,110), new Point(460,370),
            new Point(320,290), new Point(180,370))),
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

    /**
     * Scans every colour-cluster contour and returns all hits with their
     * penalised similarity score against the best-matching ref signature.
     * Each entry: {@code [x, scoreFraction, y, w, h]} (indexed for sort by [1]).
     * Uses all ref signatures so multi-colour refs are handled correctly.
     */
    public static List<double[]> allScoredBboxes(Mat scene, VectorSignature refSig) {
        // Build the full list of ref sigs via colour clusters (same as VectorMatcher)
        List<VectorSignature> refSigs = new ArrayList<>();
        List<ColourCluster> refClusters = SceneColourClusters.extract(scene);
        // We don't have the original ref Mat here, so fall back to the single sig passed in
        refSigs.add(refSig);
        for (ColourCluster c : refClusters) c.release();

        return allScoredBboxes(scene, refSigs);
    }

    /** Full version — scores against a list of ref signatures (one per colour cluster). */
    /** Full version — scores against a list of ref signatures (one per colour cluster). */
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


