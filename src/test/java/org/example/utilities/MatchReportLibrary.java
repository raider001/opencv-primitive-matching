package org.example.utilities;

import org.example.analytics.AnalysisResult;
import org.example.colour.SceneColourClusters;
import org.example.factories.BackgroundId;
import org.example.factories.ReferenceId;
import org.example.factories.ReferenceImageFactory;
import org.example.matchers.SceneDescriptor;
import org.example.matchers.VectorMatcher;
import org.example.matchers.VectorSignature;
import org.example.matchers.VectorVariant;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Shared HTML visual-report library for VectorMatcher tests.
 *
 * <p>Accumulates {@link ReportRow} entries via {@link #record} then writes a
 * self-contained HTML report via {@link #writeReport(Path, String)}.
 *
 * <p>Usage:
 * <pre>
 *   MatchReportLibrary report = new MatchReportLibrary();
 *   // inside a test:
 *   double score = report.record("Stage 1", "S1a", "CIRCLE_FILLED", "circle", scene,
 *                                runMatcher(...));
 *   // in @AfterAll:
 *   report.writeReport(OUTPUT, "My Report Title");
 * </pre>
 */
public class MatchReportLibrary {

    // ── Row ───────────────────────────────────────────────────────────────────
    public record ReportRow(
            String stage, String label, String shapeName, String sceneDesc,
            double score, boolean passed,
            String refOrig, String refPoints,
            String sceneOrig, String sceneBin,
            String allPoints, String sceneAnnot,
            double iou, boolean falsePositive,
            long elapsedMs, long descriptorMs) {}

    /** Simple carrier for a list of results + descriptor build time. */
    public record MatchRun(List<AnalysisResult> results, long descriptorMs) {
        public static MatchRun of(List<AnalysisResult> results) {
            return new MatchRun(results, 0L);
        }
    }

    private final List<ReportRow> rows = new CopyOnWriteArrayList<>();

    /** Clear accumulated rows (call in {@code @BeforeAll}). */
    public void clear() { rows.clear(); }

    public List<ReportRow> rows() { return Collections.unmodifiableList(rows); }

    // ── record overloads ──────────────────────────────────────────────────────

    /** Auto-derives ground-truth rect from white pixels in {@code sceneMat}. */
    public double record(String stage, String label, String shapeName,
                         String sceneDesc, Mat sceneMat, MatchRun run) {
        Rect gt = MatchDiagnosticLibrary.groundTruthRect(sceneMat);
        return record(stage, label, shapeName, sceneDesc, sceneMat, gt, run);
    }

    public double record(String stage, String label, String shapeName,
                         String sceneDesc, Mat sceneMat, Rect groundTruth, MatchRun run) {
        return record(stage, label, shapeName, sceneDesc, sceneMat,
                      groundTruth, run.results(), run.descriptorMs());
    }

    public double record(String stage, String label, String shapeName,
                         String sceneDesc, Mat sceneMat,
                         Rect groundTruth, List<AnalysisResult> results, long descriptorMs) {
        double score = normalScore(results);
        boolean passed = score >= 50.0;

        ReferenceId rid = results.isEmpty() ? null : results.get(0).referenceId();

        // Recolour the synthetic white-on-black scene to the reference colour
        Mat sceneWithRef = rid != null ? recolourToRef(sceneMat, rid) : sceneMat.clone();

        // ── Reference images ──────────────────────────────────────────────
        String refOrigPng = "", refPointsPng = "";
        if (rid != null) {
            Mat refOrig = ReferenceImageFactory.build(rid);
            refOrigPng = matToBase64Png(refOrig);
            Mat refBin  = VectorMatcher.extractBinaryRaw(refOrig);
            List<MatOfPoint> refContours = VectorMatcher.extractContoursFromBinary(refOrig);
            Mat graph = VectorMatcher.drawContourGraph(refOrig.size(), refBin, refContours, 0);
            refPointsPng = matToBase64Png(graph);
            graph.release(); refBin.release(); refOrig.release();
        }

        String sceneOrigPng = matToBase64Png(sceneWithRef);

        // ── Edges (binary + Canny) ────────────────────────────────────────
        String sceneBinPng;
        {
            Mat bin = VectorMatcher.extractBinaryRaw(sceneWithRef);
            Mat bgr = new Mat();
            Imgproc.cvtColor(bin, bgr, Imgproc.COLOR_GRAY2BGR);
            sceneBinPng = matToBase64Png(bgr);
            bgr.release(); bin.release();
        }

        // ── Colour clusters (what the matcher actually sees) ──────────────
        String allPointsPng;
        {
            List<SceneColourClusters.Cluster> clusters = SceneColourClusters.extract(sceneWithRef);
            List<MatOfPoint> allContours = new ArrayList<>();
            for (SceneColourClusters.Cluster c : clusters) {
                allContours.addAll(SceneDescriptor.contoursFromMask(c.mask));
                c.release();
            }
            Mat graph = VectorMatcher.drawContourGraph(sceneWithRef.size(), null, allContours, 0);
            allPointsPng = matToBase64Png(graph);
            graph.release();
        }

        // ── Annotated result ──────────────────────────────────────────────
        AnalysisResult normalResult = results.stream()
                .filter(r -> r.methodName().equals(VectorVariant.VECTOR_NORMAL.variantName()))
                .findFirst().orElse(results.isEmpty() ? null : results.get(0));

        Rect bestBbox = normalResult != null ? normalResult.boundingRect() : null;
        if (bestBbox == null && rid != null) {
            // fall back to scanning clusters
            VectorSignature refSig = VectorMatcher.buildRefSignature(
                    ReferenceImageFactory.build(rid), VectorVariant.VECTOR_NORMAL.epsilonFactor());
            bestBbox = findBestBbox(sceneWithRef, refSig);
        }
        long elapsedMs = normalResult != null ? normalResult.elapsedMs() : 0L;

        // ── Collect ALL scored hits for secondary outlines ────────────────
        VectorSignature refSigForHits = rid != null
                ? VectorMatcher.buildRefSignature(ReferenceImageFactory.build(rid),
                                                   VectorVariant.VECTOR_NORMAL.epsilonFactor())
                : null;
        List<double[]> allHits = refSigForHits != null
                ? allScoredBboxes(sceneWithRef, refSigForHits)
                : List.of();

        String sceneAnnotPng = buildAnnotated(sceneWithRef, bestBbox, groundTruth, score, allHits);

        // ── IoU / false-positive flag ─────────────────────────────────────
        double iou = Double.NaN;
        boolean fp = false;
        if (bestBbox != null && groundTruth != null) {
            iou = MatchDiagnosticLibrary.iou(bestBbox, groundTruth);
            fp  = (score >= 40.0) && (iou < 0.3);
        }

        rows.add(new ReportRow(stage, label, shapeName, sceneDesc,
                score, passed,
                refOrigPng, refPointsPng,
                sceneOrigPng, sceneBinPng, allPointsPng, sceneAnnotPng,
                iou, fp, elapsedMs, descriptorMs));

        sceneWithRef.release();
        return score;
    }

    // ── Report writer ─────────────────────────────────────────────────────────

    /**
     * Writes an HTML report covering only the rows accumulated since {@link #clear()}.
     * Deletes any existing file at the output path first.
     */
    public void writeReport(Path outputDir, String title) throws IOException {
        Files.createDirectories(outputDir);
        Path out = outputDir.resolve("report.html");
        Files.deleteIfExists(out);
        Files.writeString(out, buildHtml(rows, title), StandardCharsets.UTF_8);
        System.out.println("[MatchReportLibrary] Report: " + out.toAbsolutePath());

        // stdout summary
        long total  = rows.size();
        long passed = rows.stream().filter(ReportRow::passed).count();
        long fp     = rows.stream().filter(ReportRow::falsePositive).count();
        System.out.printf("[MatchReportLibrary] %d rows | %d passed | %d failed | %d false positives%n",
                total, passed, total - passed, fp);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    public static double normalScore(List<AnalysisResult> results) {
        return results.stream()
                .filter(r -> r.methodName().equals(VectorVariant.VECTOR_NORMAL.variantName()))
                .findFirst()
                .map(AnalysisResult::matchScorePercent)
                .orElse(0.0);
    }

    public static double normalScore(MatchRun run) {
        return normalScore(run.results());
    }

    private static Rect findBestBbox(Mat scene, VectorSignature refSig) {
        if (refSig == null) return null;
        return MatchDiagnosticLibrary.allScoredBboxes(scene, refSig).stream()
                .max(Comparator.comparingDouble(e -> e[1]))
                .map(e -> new Rect((int)e[0], (int)e[2], (int)e[3], (int)e[4]))
                .orElse(null);
    }

    /**
     * Returns all contour bboxes with their penalised similarity scores.
     * Each entry: [x, scoreFraction, y, w, h] — delegates to MatchDiagnosticLibrary.
     */
    private static List<double[]> allScoredBboxes(Mat scene, VectorSignature refSig) {
        return MatchDiagnosticLibrary.allScoredBboxes(scene, refSig);
    }

    private static String buildAnnotated(Mat scene, Rect winnerBbox, Rect gt, double winnerScore) {
        return buildAnnotated(scene, winnerBbox, gt, winnerScore, List.of());
    }

    private static String buildAnnotated(Mat scene, Rect winnerBbox, Rect gt,
                                          double winnerScore, List<double[]> allHits) {
        Mat annotated = scene.clone();

        // ── Secondary hits (thin outline + small label) ───────────────────
        // Draw all non-winner hits that scored > 15%, from lowest to highest
        // so the winner is drawn last (on top).
        allHits.stream()
                .filter(e -> e[1] > 0.15)
                .filter(e -> {
                    // Skip if this IS the winner bbox
                    if (winnerBbox == null) return true;
                    return !(e[0] == winnerBbox.x && e[2] == winnerBbox.y
                          && e[3] == winnerBbox.width && e[4] == winnerBbox.height);
                })
                .sorted(Comparator.comparingDouble(e -> e[1]))
                .forEach(e -> {
                    Rect bb = new Rect((int)e[0], (int)(e[2]), (int)e[3], (int)e[4]);
                    double pct = e[1] * 100.0;
                    // colour by score: dim versions of the same scheme
                    Scalar col = pct >= 70 ? new Scalar(0, 120, 0)
                               : pct >= 40 ? new Scalar(0, 120, 120)
                               :             new Scalar(80, 80, 80);
                    Imgproc.rectangle(annotated,
                            new Point(bb.x, bb.y),
                            new Point(bb.x + bb.width, bb.y + bb.height), col, 1);
                    // small label above top-left corner
                    int lx = Math.max(1, bb.x);
                    int ly = Math.max(9, bb.y - 2);
                    Imgproc.putText(annotated, String.format("%.0f%%", pct),
                            new Point(lx, ly),
                            Imgproc.FONT_HERSHEY_SIMPLEX, 0.30, col, 1);
                });

        // ── Ground-truth box (cyan) ───────────────────────────────────────
        if (gt != null)
            Imgproc.rectangle(annotated,
                    new Point(gt.x, gt.y), new Point(gt.x+gt.width, gt.y+gt.height),
                    new Scalar(220,220,0), 2);

        // ── Winner box (thick, bright) ────────────────────────────────────
        if (winnerBbox != null && winnerBbox.width > 1 && winnerBbox.height > 1) {
            Scalar col = winnerScore >= 70 ? new Scalar(0,200,0)
                       : winnerScore >= 40 ? new Scalar(0,200,200)
                       :                    new Scalar(0,0,200);
            Imgproc.rectangle(annotated,
                    new Point(winnerBbox.x, winnerBbox.y),
                    new Point(winnerBbox.x+winnerBbox.width, winnerBbox.y+winnerBbox.height), col, 3);
        }
        // Winner score label (top-left corner)
        Scalar lc = winnerScore >= 50 ? new Scalar(0,220,0) : new Scalar(0,0,220);
        Imgproc.putText(annotated, String.format("%.1f%%", winnerScore),
                new Point(6, 28), Imgproc.FONT_HERSHEY_SIMPLEX, 0.55, lc, 2);

        String png = matToBase64Png(annotated);
        annotated.release();
        return png;
    }

    private static Mat recolourToRef(Mat whiteOnBlack, ReferenceId rid) {
        Mat ref = ReferenceImageFactory.build(rid);
        Mat refGrey = new Mat(), refMask = new Mat();
        Imgproc.cvtColor(ref, refGrey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(refGrey, refMask, 20, 255, Imgproc.THRESH_BINARY);
        Scalar meanColour = Core.mean(ref, refMask);
        ref.release(); refGrey.release(); refMask.release();

        Mat sceneGrey = new Mat(), fgMask = new Mat();
        Imgproc.cvtColor(whiteOnBlack, sceneGrey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(sceneGrey, fgMask, 240, 255, Imgproc.THRESH_BINARY);
        sceneGrey.release();

        Mat result = whiteOnBlack.clone();
        Mat fill   = new Mat(result.size(), result.type(), meanColour);
        fill.copyTo(result, fgMask);
        fill.release(); fgMask.release();
        return result;
    }

    public static String matToBase64Png(Mat m) {
        try {
            MatOfByte buf = new MatOfByte();
            Imgcodecs.imencode(".png", m, buf);
            return Base64.getEncoder().encodeToString(buf.toArray());
        } catch (Exception e) { return ""; }
    }

    // ── HTML builder ──────────────────────────────────────────────────────────

    private static String buildHtml(List<ReportRow> rows, String title) {
        Map<String, List<ReportRow>> byStage = new LinkedHashMap<>();
        for (ReportRow r : rows)
            byStage.computeIfAbsent(r.stage(), k -> new ArrayList<>()).add(r);

        long total    = rows.size();
        long passed   = rows.stream().filter(ReportRow::passed).count();
        long fp       = rows.stream().filter(ReportRow::falsePositive).count();
        String ts     = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));

        StringBuilder sb = new StringBuilder();
        sb.append("<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>")
          .append("<title>").append(esc(title)).append("</title>")
          .append("<style>").append(CSS).append("</style></head><body>")
          .append("<div class='header'>")
          .append("<h1>").append(esc(title)).append("</h1>")
          .append("<div class='ts-line'>Generated: <span class='ts-val'>").append(ts).append("</span></div>")
          .append("<p class='subtitle'>")
          .append(total).append(" calls &nbsp;·&nbsp; ").append(passed).append(" passed")
          .append(" &nbsp;·&nbsp; <span style='color:#f85149'>").append(total-passed).append(" failed</span>")
          .append(" &nbsp;·&nbsp; <span style='color:#bf55ec;font-weight:700'>").append(fp).append(" false positives</span>")
          .append("</p>")
          .append("<div class='legend-block'>")
          .append("<div class='legend-title'>Pipeline (per row)</div>")
          .append("<div class='legend-row'>")
          .append("<span class='pl-step'>Ref</span><span class='pl-arrow'>→</span>")
          .append("<span class='pl-step'>Ref Points</span><span class='pl-arrow'>→</span>")
          .append("<span class='pl-step'>Scene</span><span class='pl-arrow'>→</span>")
          .append("<span class='pl-step'>Edges (ref)</span><span class='pl-arrow'>→</span>")
          .append("<span class='pl-step'>Colour Clusters</span><span class='pl-arrow'>→</span>")
          .append("<span class='pl-step'>Match</span>")
          .append("</div></div>")
          .append("<div class='legend-block'><div class='legend-title'>Row status</div>")
          .append("<div class='legend-row'>")
          .append("<span class='legend-pill pass-pill'>green = passed</span>")
          .append("<span class='legend-pill fail-pill'>red = failed</span>")
          .append("<span class='legend-pill fp-pill'>purple = false positive</span>")
          .append("</div></div>")
          .append("</div>");

        for (Map.Entry<String, List<ReportRow>> e : byStage.entrySet()) {
            sb.append("<section><h2>").append(esc(e.getKey())).append("</h2>");
            for (ReportRow r : e.getValue()) {
                String cls = r.falsePositive() ? "row fp"
                           : r.passed()        ? "row pass" : "row fail";
                sb.append("<div class='").append(cls).append("'>")
                  .append("<div class='row-meta'>")
                  .append("<span class='row-id'>").append(esc(r.label())).append("</span>")
                  .append("<span class='row-shape'>").append(esc(r.shapeName())).append("</span>")
                  .append("<span class='row-desc'>").append(esc(r.sceneDesc())).append("</span>");
                if (r.falsePositive())
                    sb.append("<span class='badge fp-badge'>⚠ FALSE POSITIVE</span>");
                if (!Double.isNaN(r.iou())) {
                    String ic = r.iou()>=0.5?"iou-good":r.iou()>=0.3?"iou-warn":"iou-bad";
                    sb.append("<span class='iou-val ").append(ic).append("'>IoU ")
                      .append(String.format("%.2f",r.iou())).append("</span>");
                }
                sb.append("<span class='timing-badge'>")
                  .append("desc:").append(r.descriptorMs()).append("ms")
                  .append(" match:").append(r.elapsedMs()).append("ms</span>")
                  .append("</div>")
                  .append("<div class='pipeline-row'><div class='pipeline'>");
                step(sb, r.refOrig(),    "Ref");
                step(sb, r.refPoints(),  "Ref Points");
                step(sb, r.sceneOrig(),  "Scene");
                step(sb, r.sceneBin(),   "Edges");
                step(sb, r.allPoints(),  "Colour Clusters");
                step(sb, r.sceneAnnot(), "Match");
                sb.append("</div>")
                  .append("<div class='pipeline-score'>")
                  .append("<span class='score-val ").append(r.score()>=70?"s-good":r.score()>=40?"s-warn":"s-bad").append("'>")
                  .append(String.format("%.1f%%",r.score())).append("</span>")
                  .append(bar(r.score()))
                  .append("</div></div></div>");
            }
            sb.append("</section>");
        }

        sb.append("<div id='lb' class='lb-overlay' onclick='closeLb()'>")
          .append("<div class='lb-box' onclick='event.stopPropagation()'>")
          .append("<button class='lb-close' onclick='closeLb()'>✕</button>")
          .append("<img id='lb-img' src='' alt='' class='lb-img'/>")
          .append("<div id='lb-caption' class='lb-caption'></div></div></div>")
          .append("<script>")
          .append("function openLb(s,c){document.getElementById('lb-img').src=s;")
          .append("document.getElementById('lb-caption').textContent=c;")
          .append("document.getElementById('lb').classList.add('lb-visible');}")
          .append("function closeLb(){document.getElementById('lb').classList.remove('lb-visible');}")
          .append("document.addEventListener('keydown',e=>{if(e.key==='Escape')closeLb();});")
          .append("</script></body></html>");
        return sb.toString();
    }

    private static void step(StringBuilder sb, String b64, String label) {
        sb.append("<div class='step'>");
        if (b64 != null && !b64.isEmpty()) {
            String src = "data:image/png;base64," + b64;
            sb.append("<img src='").append(src).append("' class='step-img' alt='").append(esc(label))
              .append("' title='Click to enlarge' onclick=\"openLb('").append(src)
              .append("','").append(esc(label)).append("')\" style='cursor:zoom-in'/>");
        } else {
            sb.append("<div class='step-img step-empty'></div>");
        }
        sb.append("<div class='step-label'>").append(esc(label)).append("</div></div>");
    }

    private static String bar(double s) {
        String col = s>=70?"#56d364":s>=40?"#d29922":"#f85149";
        int w = (int)Math.max(1,Math.min(100,s));
        return "<div class='bar-bg'><div class='bar-fill' style='width:"+w+"%;background:"+col+"'></div></div>";
    }

    private static String esc(String s) {
        return s==null?"":s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;");
    }

    private static final String CSS = """
        *{box-sizing:border-box;margin:0;padding:0}
        body{font-family:system-ui,sans-serif;background:#0d1117;color:#c9d1d9;padding:0 0 48px}
        .header{background:#161b22;padding:20px 32px 16px;border-bottom:1px solid #30363d;margin-bottom:20px}
        .header h1{color:#58a6ff;font-size:1.4rem;margin-bottom:4px}
        .ts-line{font-size:.75rem;color:#8b949e;margin-bottom:6px}
        .ts-val{color:#79c0ff;font-weight:600}
        .subtitle{color:#8b949e;font-size:.88rem;margin-bottom:8px}
        .pl-step{background:#21262d;border:1px solid #30363d;border-radius:4px;padding:2px 7px;font-size:.76rem}
        .pl-arrow{color:#484f58;font-size:.76rem}
        .legend-block{background:#21262d;border:1px solid #30363d;border-radius:6px;padding:8px 12px;margin-top:8px}
        .legend-title{font-size:.72rem;font-weight:700;color:#79c0ff;margin-bottom:5px;text-transform:uppercase}
        .legend-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
        .legend-pill{font-size:.72rem;font-weight:600;border-radius:4px;padding:2px 8px;border-left:3px solid transparent}
        .pass-pill{border-left-color:#238636;background:#0d2619;color:#56d364}
        .fail-pill{border-left-color:#da3633;background:#2b0c0c;color:#f85149}
        .fp-pill{border-left-color:#bf55ec;background:#1a0e2e;color:#d2a8ff}
        section{padding:0 24px 16px}
        h2{color:#79c0ff;font-size:1rem;margin:14px 0 10px;padding-bottom:4px;border-bottom:1px solid #21262d}
        .row{background:#161b22;border:1px solid #30363d;border-radius:8px;margin-bottom:8px;padding:10px 12px;display:flex;flex-direction:column;gap:8px}
        .row.pass{border-left:3px solid #238636}
        .row.fail{border-left:3px solid #da3633}
        .row.fp{border-left:3px solid #bf55ec;background:#1a0e2e}
        .fp-badge{background:#bf55ec;color:#fff;font-size:.68rem;font-weight:700;border-radius:3px;padding:1px 6px}
        .iou-val{font-size:.72rem;font-weight:600;white-space:nowrap}
        .iou-good{color:#56d364}.iou-warn{color:#d29922}.iou-bad{color:#f85149}
        .timing-badge{font-size:.68rem;color:#8b949e;background:#21262d;border:1px solid #30363d;border-radius:4px;padding:1px 8px;white-space:nowrap;margin-left:auto}
        .row-meta{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
        .row-id{font-size:.72rem;font-weight:700;background:#21262d;border-radius:3px;padding:1px 6px;color:#79c0ff}
        .row-shape{font-size:.78rem;font-weight:600;color:#c9d1d9}
        .row-desc{font-size:.72rem;color:#8b949e;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
        .pipeline-row{display:flex;align-items:center;gap:8px}
        .pipeline{display:flex;align-items:flex-start;gap:6px;overflow-x:auto;flex:1}
        .pipeline-score{display:flex;flex-direction:column;align-items:flex-end;gap:4px;flex-shrink:0;min-width:70px}
        .step{display:flex;flex-direction:column;align-items:center;gap:3px;flex-shrink:0}
        .step-img{width:128px;height:96px;object-fit:contain;border-radius:4px;border:1px solid #30363d;background:#0d1117;cursor:zoom-in;transition:border-color .15s}
        .step-img:hover{border-color:#58a6ff}
        .step-empty{width:128px;height:96px;border:1px dashed #30363d;border-radius:4px}
        .step-label{font-size:.65rem;color:#8b949e;text-align:center;max-width:128px}
        .score-val{font-size:.82rem;font-weight:700}
        .s-good{color:#56d364}.s-warn{color:#d29922}.s-bad{color:#f85149}
        .bar-bg{width:120px;height:7px;background:#21262d;border-radius:4px;overflow:hidden}
        .bar-fill{height:100%;border-radius:4px}
        .lb-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.88);z-index:9999;overflow:auto;padding:48px 24px 24px}
        .lb-overlay.lb-visible{display:flex;align-items:center;justify-content:center}
        .lb-box{position:relative;padding:8px;background:#161b22;border:1px solid #30363d;border-radius:8px}
        .lb-img{display:block;max-width:90vw;max-height:80vh;image-rendering:pixelated;border-radius:4px}
        .lb-caption{color:#c9d1d9;font-size:.85rem;text-align:center;margin-top:6px}
        .lb-close{position:absolute;top:-14px;right:-14px;width:28px;height:28px;border-radius:50%;background:#21262d;border:1px solid #484f58;color:#c9d1d9;font-size:1rem;cursor:pointer;display:flex;align-items:center;justify-content:center;z-index:10000}
        .lb-close:hover{background:#30363d;color:#fff}
        """;
}

