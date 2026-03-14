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
import java.lang.reflect.Method;
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
            String sceneOrig,
            String allPoints, String clusterJson, String sceneAnnot,
            double iou,
            long elapsedMs, long descriptorMs) {}

    /** Simple carrier for a list of results + descriptor build time. */
    public record MatchRun(List<AnalysisResult> results, long descriptorMs) {
        public static MatchRun of(List<AnalysisResult> results) {
            return new MatchRun(results, 0L);
        }
    }

    private final List<ReportRow> rows = new CopyOnWriteArrayList<>();

    // ── Test documentation ────────────────────────────────────────────────────

    /**
     * Holds the documentation extracted from a single {@code @Test} method's
     * {@link ExpectedOutcome} and {@link org.junit.jupiter.api.DisplayName} annotations.
     */
    public record TestDoc(
            String methodName,
            String displayName,
            String reason) {}

    private final List<TestDoc> testDocs = new ArrayList<>();
    /** Keyed by the derived label — see {@link #deriveLabel(String)}. */
    private final Map<String, TestDoc> docByLabel = new HashMap<>();

    /**
     * Reads every {@code @Test} method on {@code testClass} that carries an
     * {@link ExpectedOutcome} annotation and stores the metadata so that each
     * result row can render its expected outcome inline in the HTML report.
     *
     * <p>Call this once from {@code @BeforeAll} after {@link #clear()}.
     */
    public void scanTestAnnotations(Class<?> testClass) {
        testDocs.clear();
        docByLabel.clear();
        for (Method m : testClass.getDeclaredMethods()) {
            if (m.getAnnotation(org.junit.jupiter.api.Test.class) == null) continue;
            ExpectedOutcome eo = m.getAnnotation(ExpectedOutcome.class);
            if (eo == null) continue;
            org.junit.jupiter.api.DisplayName dn =
                    m.getAnnotation(org.junit.jupiter.api.DisplayName.class);
            String displayName = (dn != null && !dn.value().isBlank()) ? dn.value() : m.getName();
            TestDoc doc = new TestDoc(m.getName(), displayName, eo.reason());
            testDocs.add(doc);
            docByLabel.put(deriveLabel(displayName), doc);
        }
    }

    /**
     * Derives the {@code record()} label key from a {@code @DisplayName} string so that
     * each row can be matched to its {@link TestDoc} without changing test call-sites.
     *
     * <p>Supported patterns (all produced by VectorMatchingTest):
     * <ul>
     *   <li>{@code "REFNAME — ..."}                           → {@code "REFNAME"}
     *   <li>{@code "REFNAME — on random-lines background"}    → {@code "REFNAME@BG_RANDOM_LINES"}
     *   <li>{@code "REFNAME — on random-circles background"}  → {@code "REFNAME@BG_RANDOM_CIRCLES"}
     *   <li>{@code "QUERYREF in SCENEREF scene — ..."}        → {@code "QUERYREF→SCENEREF"}
     *   <li>{@code "QUERYREF in SCENEREF — lines bg ..."}     → {@code "QUERYREF→SCENEREF@BG_RANDOM_LINES"}
     *   <li>{@code "QUERYREF in SCENEREF — circles bg ..."}   → {@code "QUERYREF→SCENEREF@BG_RANDOM_CIRCLES"}
     * </ul>
     */
    static String deriveLabel(String displayName) {
        if (displayName.contains(" in ")) {
            int inIdx = displayName.indexOf(" in ");
            String queryRef = displayName.substring(0, inIdx).trim();
            String rest = displayName.substring(inIdx + 4);
            if (rest.contains(" scene — ")) {
                String sceneRef = rest.substring(0, rest.indexOf(" scene — ")).trim();
                return queryRef + "→" + sceneRef;
            } else if (rest.contains(" — lines bg")) {
                String sceneRef = rest.substring(0, rest.indexOf(" — lines bg")).trim();
                return queryRef + "→" + sceneRef + "@BG_RANDOM_LINES";
            } else if (rest.contains(" — circles bg")) {
                String sceneRef = rest.substring(0, rest.indexOf(" — circles bg")).trim();
                return queryRef + "→" + sceneRef + "@BG_RANDOM_CIRCLES";
            }
        }
        if (displayName.contains(" — ")) {
            String refName = displayName.substring(0, displayName.indexOf(" — ")).trim();
            String desc    = displayName.substring(displayName.indexOf(" — ") + 3).trim();
            if (desc.startsWith("on random-lines background"))   return refName + "@BG_RANDOM_LINES";
            if (desc.startsWith("on random-circles background")) return refName + "@BG_RANDOM_CIRCLES";
            return refName;
        }
        return displayName;
    }

    /** Clear accumulated rows and test docs (call in {@code @BeforeAll}). */
    public void clear() { rows.clear(); testDocs.clear(); docByLabel.clear(); }

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

        ReferenceId rid = results.isEmpty() ? null : results.get(0).referenceId();

        // Recolour the synthetic white-on-black scene to the reference colour
        Mat sceneWithRef = rid != null ? recolourToRef(sceneMat, rid) : sceneMat.clone();

        // ── Reference images ──────────────────────────────────────────────
        String refOrigPng = "", refPointsPng = "";
        if (rid != null) {
            Mat refOrig = ReferenceImageFactory.build(rid);
            refOrigPng = matToBase64Png(refOrig);

            // Build ref contours from ALL clusters (chromatic + achromatic) so that
            // every colour region and edge boundary is visible — exactly mirrors how
            // SceneColourClusters works on the scene side.
            List<MatOfPoint> refContours = new ArrayList<>();
            List<SceneColourClusters.Cluster> refClusters = SceneColourClusters.extract(refOrig);
            for (SceneColourClusters.Cluster c : refClusters) {
                refContours.addAll(SceneDescriptor.contoursFromMask(c.mask));
                c.release();
            }
            if (refContours.isEmpty()) {
                refContours = VectorMatcher.extractContoursFromBinary(refOrig);
            }
            Mat refBin = VectorMatcher.extractBinaryRaw(refOrig);
            // Dim the ref original and add as underlay so contour lines are
            // clearly visible against the actual shape colours.
            Mat refDimmed = new Mat();
            refOrig.convertTo(refDimmed, -1, 0.35, 0);
            Mat graph = VectorMatcher.drawContourGraph(refOrig.size(), null, refContours, 0);
            Core.add(graph, refDimmed, graph);
            refPointsPng = matToBase64Png(graph);
            graph.release(); refDimmed.release(); refBin.release(); refOrig.release();
        }

        String sceneOrigPng = matToBase64Png(sceneWithRef);


        // ── Colour clusters (what the matcher actually sees) ──────────────
        String allPointsPng;
        String clusterJson;
        {
            List<SceneColourClusters.Cluster> clusters = SceneColourClusters.extract(sceneWithRef);
            List<MatOfPoint> allContours = new ArrayList<>();

            // Build per-cluster JSON while contour masks are still live
            StringBuilder cj = new StringBuilder();
            cj.append("{\"imageW\":").append(sceneWithRef.cols())
              .append(",\"imageH\":").append(sceneWithRef.rows())
              .append(",\"clusters\":[");

            for (int ci = 0; ci < clusters.size(); ci++) {
                SceneColourClusters.Cluster c = clusters.get(ci);
                List<MatOfPoint> contours = SceneDescriptor.contoursFromMask(c.mask);
                allContours.addAll(contours);

                if (ci > 0) cj.append(",");
                String clabel  = clusterLabel(c);
                String ccolour = clusterColour(c);
                cj.append("{\"label\":\"").append(clabel.replace("\\", "\\\\").replace("\"", "\\\"")).append("\"")
                  .append(",\"colour\":\"").append(ccolour).append("\"")
                  .append(",\"contours\":[");
                for (int j = 0; j < contours.size(); j++) {
                    if (j > 0) cj.append(",");
                    cj.append("[");
                    Point[] pts = contours.get(j).toArray();
                    for (int k = 0; k < pts.length; k++) {
                        if (k > 0) cj.append(",");
                        cj.append("[").append((int) pts[k].x)
                          .append(",").append((int) pts[k].y).append("]");
                    }
                    cj.append("]");
                }
                cj.append("]}");
                c.release();
            }
            cj.append("]}");
            clusterJson = cj.toString();

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
        List<VectorSignature> refSigsForHits = rid != null
                ? VectorMatcher.buildRefSignatures(ReferenceImageFactory.build(rid),
                                                    VectorVariant.VECTOR_NORMAL.epsilonFactor())
                : List.of();
        List<double[]> allHits = refSigsForHits.isEmpty()
                ? List.of()
                : MatchDiagnosticLibrary.allScoredBboxes(sceneWithRef, refSigsForHits);

        String sceneAnnotPng = buildAnnotated(sceneWithRef, bestBbox, groundTruth, score, allHits);

        // ── IoU / pass / false-positive flag ─────────────────────────────
        // Raw IoU: 1.0 = exact match, >1.0 = det larger than GT, <1.0 = partial coverage.
        double iou = Double.NaN;
        if (bestBbox != null && groundTruth != null) {
            iou = MatchDiagnosticLibrary.iou(bestBbox, groundTruth);
        }

        boolean passed = determinePassed(label, score, iou);

        rows.add(new ReportRow(stage, label, shapeName, sceneDesc,
                score, passed,
                refOrigPng, refPointsPng,
                sceneOrigPng, allPointsPng, clusterJson, sceneAnnotPng,
                iou, elapsedMs, descriptorMs));

        sceneWithRef.release();
        return score;
    }

    /**
     * Pass/fail is determined purely by match score and IoU — no annotation dependency.
     *
     * <h3>Detection tests</h3> (label does NOT contain {@code "→"})
     * <ul>
     *   <li>PASS — score ≥ 60 % AND (IoU unavailable OR IoU ≥ 0.95)</li>
     *   <li>FAIL — everything else</li>
     * </ul>
     *
     * <h3>Rejection tests</h3> (label contains {@code "→"}, reference absent)
     * <ul>
     *   <li>PASS — score &lt; 60 % (correctly did not fire)</li>
     *   <li>FAIL — score ≥ 60 % (incorrectly fired on absent reference)</li>
     * </ul>
     */
    private boolean determinePassed(String label, double score, double iou) {
        if (label.contains("→")) return score < 60.0;                    // rejection: no fire = pass
        return score >= 60.0 && (Double.isNaN(iou) || iou >= 0.95);     // detection: score + location
    }

    /**
     * Writes an HTML report covering only the rows accumulated since {@link #clear()}.
     * Deletes any existing file at the output path first.
     */
    public void writeReport(Path outputDir, String title) throws IOException {
        Files.createDirectories(outputDir);
        Path out = outputDir.resolve("report.html");
        Files.deleteIfExists(out);
        Files.writeString(out, buildHtml(rows, docByLabel, title), StandardCharsets.UTF_8);
        System.out.println("[MatchReportLibrary] Report: " + out.toAbsolutePath());

        long total  = rows.size();
        long passed = rows.stream().filter(ReportRow::passed).count();
        System.out.printf("[MatchReportLibrary] %d rows  %d passed  %d failed%n",
                total, passed, total - passed);
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
        return MatchDiagnosticLibrary.allScoredBboxes(scene, List.of(refSig)).stream()
                .max(Comparator.comparingDouble(e -> e[1]))
                .map(e -> new Rect((int)e[0], (int)e[2], (int)e[3], (int)e[4]))
                .orElse(null);
    }

    /**
     * Delegates to MatchDiagnosticLibrary — single sig convenience wrapper.
     */
    private static List<double[]> allScoredBboxes(Mat scene, VectorSignature refSig) {
        return MatchDiagnosticLibrary.allScoredBboxes(scene, List.of(refSig));
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
                            Imgproc.FONT_HERSHEY_SIMPLEX, 0.20, col, 1);
                });

        // ── Ground-truth box (yellow) ─────────────────────────────────────
        if (gt != null)
            Imgproc.rectangle(annotated,
                    new Point(gt.x, gt.y), new Point(gt.x+gt.width, gt.y+gt.height),
                    new Scalar(220,220,0), 1);

        // ── Winner box (bright) ───────────────────────────────────────────
        if (winnerBbox != null && winnerBbox.width > 1 && winnerBbox.height > 1) {
            Scalar col = winnerScore >= 70 ? new Scalar(0,200,0)
                       : winnerScore >= 40 ? new Scalar(0,200,200)
                       :                    new Scalar(0,0,200);
            Imgproc.rectangle(annotated,
                    new Point(winnerBbox.x, winnerBbox.y),
                    new Point(winnerBbox.x+winnerBbox.width, winnerBbox.y+winnerBbox.height), col, 1);
        }
        // Winner score label (top-left corner)
        Scalar lc = winnerScore >= 50 ? new Scalar(0,220,0) : new Scalar(0,0,220);
        Imgproc.putText(annotated, String.format("%.1f%%", winnerScore),
                new Point(4, 14), Imgproc.FONT_HERSHEY_SIMPLEX, 0.28, lc, 1);

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
            // Upscale 4× with nearest-neighbour so contours and annotations
            // are clearly visible in the HTML report lightbox.
            Mat big = new Mat();
            Imgproc.resize(m, big, new Size(m.cols() * 4, m.rows() * 4),
                    0, 0, Imgproc.INTER_NEAREST);
            MatOfByte buf = new MatOfByte();
            Imgcodecs.imencode(".png", big, buf);
            big.release();
            return Base64.getEncoder().encodeToString(buf.toArray());
        } catch (Exception e) { return ""; }
    }

    // ── HTML builder ──────────────────────────────────────────────────────────

    private static String buildHtml(List<ReportRow> rows, Map<String, TestDoc> docByLabel, String title) {
        Map<String, List<ReportRow>> byStage = new LinkedHashMap<>();
        for (ReportRow r : rows)
            byStage.computeIfAbsent(r.stage(), k -> new ArrayList<>()).add(r);

        long total    = rows.size();
        long passed   = rows.stream().filter(ReportRow::passed).count();
        String ts     = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));

        StringBuilder sb = new StringBuilder();
        sb.append("<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>")
          .append("<title>").append(esc(title)).append("</title>")
          .append("<style>").append(CSS).append("</style></head><body>")
          .append("<div class='header'>")
          .append("<h1>").append(esc(title)).append("</h1>")
          .append("<div class='ts-line'>Generated: <span class='ts-val'>").append(ts).append("</span></div>")
          .append("<p class='subtitle'>")
          .append(total).append(" calls &nbsp;·&nbsp; ")
          .append("<span style='color:#56d364'>").append(passed).append(" passed</span>")
          .append(" &nbsp;·&nbsp; <span style='color:#f85149'>").append(total - passed).append(" failed</span>")
          .append("</p>")
          .append("<div class='legend-block'>")
          .append("<div class='legend-title'>Pipeline (per row)</div>")
          .append("<div class='legend-row'>")
          .append("<span class='pl-step'>Ref</span><span class='pl-arrow'>→</span>")
          .append("<span class='pl-step'>Ref Points</span><span class='pl-arrow'>→</span>")
          .append("<span class='pl-step'>Scene</span><span class='pl-arrow'>→</span>")
          .append("<span class='pl-step'>Colour Clusters</span><span class='pl-arrow'>→</span>")
          .append("<span class='pl-step'>Match</span>")
          .append("</div></div>")
          .append("<div class='legend-block'><div class='legend-title'>Row status</div>")
          .append("<div class='legend-row'>")
          .append("<span class='legend-pill pass-pill'>green = passed (score ≥ 60 % &amp; IoU ≥ 0.95)</span>")
          .append("<span class='legend-pill fail-pill'>red = failed</span>")
          .append("</div></div>")
          .append("</div>");

        for (Map.Entry<String, List<ReportRow>> e : byStage.entrySet()) {
            sb.append("<section><h2>").append(esc(e.getKey())).append("</h2>");
            for (ReportRow r : e.getValue()) {
                int ri = rows.indexOf(r); // unique row index for element IDs
                // Row is green (pass) or red (fail) — purely score + IoU
                String cls = r.passed() ? "row pass" : "row fail";
                sb.append("<div class='").append(cls).append("'>")
                  .append("<div class='row-meta'>")
                  .append("<span class='row-id'>").append(esc(r.label())).append("</span>")
                  .append("<span class='row-shape'>").append(esc(r.shapeName())).append("</span>")
                  .append("<span class='row-desc'>").append(esc(r.sceneDesc())).append("</span>");
                if (!Double.isNaN(r.iou())) {
                    // 1.0 = exact fit | >1.0 = det larger than GT | <1.0 = partial coverage
                    String ic = r.iou() >= 1.0 ? "iou-excellent"
                              : r.iou() >= 0.8 ? "iou-good"
                              : r.iou() >= 0.5 ? "iou-warn"
                              : "iou-bad";
                    sb.append("<span class='iou-val ").append(ic).append("'>IoU ")
                      .append(String.format("%.2f", r.iou())).append("</span>");
                }
                sb.append("<span class='timing-badge'>")
                  .append("desc:").append(r.descriptorMs()).append("ms")
                  .append(" match:").append(r.elapsedMs()).append("ms</span>")
                  .append("</div>");

                // ── Inline expected outcome (matched by label) ────────────
                TestDoc doc = docByLabel.get(r.label());
                if (doc != null) {
                    sb.append("<div class='outcome-line'>")
                      .append("<details class='outcome-details'><summary>Scenario: ")
                      .append(esc(doc.displayName())).append("</summary>")
                      .append("<span class='outcome-reason'>").append(esc(doc.reason())).append("</span>")
                      .append("</details>")
                      .append("</div>");
                }

                sb.append("<div class='pipeline-row'><div class='pipeline'>");
                step(sb, r.refOrig(),    "Ref");
                step(sb, r.refPoints(),  "Ref Points");
                stepWithId(sb, r.sceneOrig(), "Scene", "si-" + ri);
                stepWithClusters(sb, r.allPoints(), "Colour Clusters",
                        r.clusterJson(), "si-" + ri, r.label());
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

          // Cluster explorer overlay — separate from the standard image lightbox
          .append("<div id='cl-lb' class='cl-lb-overlay' onclick='closeCluster()'>")
          .append("<div class='cl-lb-box' onclick='event.stopPropagation()'>")
          .append("<button class='lb-close' onclick='closeCluster()'>✕</button>")
          .append("<div class='cl-lb-header'>")
          .append("<span>&#x1F52C; Colour Cluster Explorer</span>")
          .append("<span id='cl-counter' class='cl-counter'></span>")
          .append("</div>")
          .append("<div id='cl-status' class='cl-status'></div>")
          .append("<div class='cl-img-wrap'>")
          .append("<img id='cl-base-img' src='' alt=''>")
          .append("<svg id='cl-svg' class='cl-svg' xmlns='http://www.w3.org/2000/svg'></svg>")
          .append("</div>")
          .append("<div class='cl-controls'>")
          .append("<button class='cl-nav-btn' onclick='clusterStep(-1)'>&#9664; Prev</button>")
          .append("<div class='cl-info'>")
          .append("<span id='cl-label'></span>")
          .append("<span id='cl-cnt'></span>")
          .append("</div>")
          .append("<button class='cl-nav-btn' onclick='clusterStep(+1)'>Next &#9654;</button>")
          .append("</div>")
          .append("<div id='cl-cap' class='lb-caption'></div>")
          .append("</div></div>")

          .append("<script>")
          .append("function openLb(s,c){document.getElementById('lb-img').src=s;")
          .append("document.getElementById('lb-caption').textContent=c;")
          .append("document.getElementById('lb').classList.add('lb-visible');}")
          .append("function closeLb(){document.getElementById('lb').classList.remove('lb-visible');}")

          .append("var _clData=null,_clIdx=0;")
          .append("function openCluster(btn){")
          .append("  var json=btn.getAttribute('data-clusters');")
          .append("  var sceneId=btn.getAttribute('data-scene-id');")
          .append("  var cap=btn.getAttribute('data-cap');")
          .append("  var imgEl=document.getElementById(sceneId);")
          .append("  document.getElementById('cl-base-img').src=imgEl?imgEl.src:'';")
          .append("  document.getElementById('cl-cap').textContent=cap||'';")
          .append("  _clData=JSON.parse(json); _clIdx=0;")
          .append("  document.getElementById('cl-lb').classList.add('cl-visible');")
          .append("  clRender(0);")
          .append("}")
          .append("function closeCluster(){")
          .append("  document.getElementById('cl-lb').classList.remove('cl-visible');")
          .append("  document.getElementById('cl-svg').innerHTML='';")
          .append("  _clData=null;")
          .append("}")
          .append("function clusterStep(d){")
          .append("  if(!_clData||!_clData.clusters.length)return;")
          .append("  var n=_clData.clusters.length;")
          .append("  clRender((_clIdx+d+n)%n);")
          .append("}")
          .append("function clRender(idx){")
          .append("  if(!_clData)return;")
          .append("  var cls=_clData.clusters;")
          .append("  if(!cls||!cls.length){document.getElementById('cl-status').textContent='No clusters.';return;}")
          .append("  idx=((idx%cls.length)+cls.length)%cls.length; _clIdx=idx;")
          .append("  var c=cls[idx];")
          .append("  document.getElementById('cl-status').textContent='';")
          .append("  document.getElementById('cl-counter').textContent=(idx+1)+' / '+cls.length;")
          .append("  document.getElementById('cl-label').textContent=c.label;")
          .append("  document.getElementById('cl-cnt').textContent=c.contours.length+' contour(s)';")
          .append("  var svg=document.getElementById('cl-svg');")
          .append("  svg.setAttribute('viewBox','0 0 '+_clData.imageW+' '+_clData.imageH);")
          .append("  svg.setAttribute('preserveAspectRatio','none');")
          .append("  svg.innerHTML='';")
          .append("  c.contours.forEach(function(pts){")
          .append("    if(!pts.length)return;")
          .append("    var el=document.createElementNS('http://www.w3.org/2000/svg','polygon');")
          .append("    el.setAttribute('points',pts.map(function(p){return p[0]+','+p[1];}).join(' '));")
          .append("    el.setAttribute('fill',c.colour+'33');")
          .append("    el.setAttribute('stroke',c.colour);")
          .append("    el.setAttribute('stroke-width','1.5');")
          .append("    svg.appendChild(el);")
          .append("  });")
          .append("}")
          .append("document.addEventListener('keydown',function(e){")
          .append("  if(document.getElementById('cl-lb').classList.contains('cl-visible')){")
          .append("    if(e.key==='ArrowLeft')clusterStep(-1);")
          .append("    if(e.key==='ArrowRight')clusterStep(+1);")
          .append("    if(e.key==='Escape')closeCluster();")
          .append("  }else if(e.key==='Escape')closeLb();")
          .append("});")
          .append("</script></body></html>");
        return sb.toString();
    }

    /** Like {@link #step} but stamps the img element with a unique {@code id}. */
    private static void stepWithId(StringBuilder sb, String b64, String label, String imgId) {
        sb.append("<div class='step'>");
        if (b64 != null && !b64.isEmpty()) {
            String src = "data:image/png;base64," + b64;
            sb.append("<img id='").append(imgId).append("' src='").append(src)
              .append("' class='step-img' alt='").append(esc(label))
              .append("' title='Click to enlarge' onclick=\"openLb('").append(src)
              .append("','").append(esc(label)).append("')\" style='cursor:zoom-in'/>");
        } else {
            sb.append("<img id='").append(imgId).append("' class='step-img step-empty' src='' alt=''>");
        }
        sb.append("<div class='step-label'>").append(esc(label)).append("</div></div>");
    }

    /**
     * Renders the "Colour Clusters" pipeline step: the combined image plus a
     * 🔬 button that opens the per-cluster SVG explorer in the cluster lightbox.
     * Falls back to a plain {@link #step} call when no cluster JSON is available.
     */
    private static void stepWithClusters(StringBuilder sb, String b64, String label,
                                          String clusterJson, String sceneImgId,
                                          String rowLabel) {
        sb.append("<div class='step'>");
        if (b64 != null && !b64.isEmpty()) {
            String src = "data:image/png;base64," + b64;
            sb.append("<img src='").append(src).append("' class='step-img' alt='").append(esc(label))
              .append("' title='Click to enlarge' onclick=\"openLb('").append(src)
              .append("','").append(esc(label)).append("')\" style='cursor:zoom-in'/>");
        } else {
            sb.append("<div class='step-img step-empty'></div>");
        }
        sb.append("<div class='step-label'>").append(esc(label)).append("</div>");
        // Cluster explorer button — only when cluster data is present
        if (clusterJson != null && !clusterJson.isEmpty()) {
            // Embed JSON as a single-quoted attribute; escape ' and & within the value.
            String jsonAttr = clusterJson.replace("&", "&amp;").replace("'", "&#39;");
            String cap = esc(rowLabel) + " — Colour Clusters";
            sb.append("<button class='cl-btn'")
              .append(" data-clusters='").append(jsonAttr).append("'")
              .append(" data-scene-id='").append(sceneImgId).append("'")
              .append(" data-cap='").append(cap).append("'")
              .append(" onclick='openCluster(this)'")
              .append(" title='Explore individual colour clusters'>")
              .append("&#x1F52C; Clusters</button>");
        }
        sb.append("</div>");
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

    // ── Cluster helpers ───────────────────────────────────────────────────────

    private static String clusterLabel(SceneColourClusters.Cluster c) {
        if (c.achromatic) return c.brightAchromatic
                ? "Bright Achromatic (White / Light-Grey)"
                : "Dark Achromatic (Black / Dark-Grey)";
        return String.format("Chromatic H=%.0f\u00b0 (%s)", c.hue * 2.0, hueNameApprox(c.hue));
    }

    private static String clusterColour(SceneColourClusters.Cluster c) {
        if (c.achromatic) return c.brightAchromatic ? "#e8e8e8" : "#888888";
        int cssDeg = ((int) Math.round(c.hue * 2.0)) % 360;
        return String.format("hsl(%d,100%%,60%%)", cssDeg);
    }

    private static String hueNameApprox(double ocvHue) {
        int deg = ((int) (ocvHue * 2.0)) % 360;
        if (deg < 15 || deg >= 345) return "Red";
        if (deg <  45)              return "Orange";
        if (deg <  75)              return "Yellow";
        if (deg < 150)              return "Green";
        if (deg < 195)              return "Cyan";
        if (deg < 255)              return "Blue";
        if (deg < 285)              return "Violet";
        return "Magenta";
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
        section{padding:0 24px 16px}
        h2{color:#79c0ff;font-size:1rem;margin:14px 0 10px;padding-bottom:4px;border-bottom:1px solid #21262d}
        .row{background:#161b22;border:1px solid #30363d;border-radius:8px;margin-bottom:8px;padding:10px 12px;display:flex;flex-direction:column;gap:6px}
        .row.pass{border-left:3px solid #238636}
        .row.fail{border-left:3px solid #da3633}
        .iou-val{font-size:.72rem;font-weight:600;white-space:nowrap}
        .iou-excellent{color:#79c0ff}.iou-good{color:#56d364}.iou-warn{color:#d29922}.iou-bad{color:#f85149}
        .timing-badge{font-size:.68rem;color:#8b949e;background:#21262d;border:1px solid #30363d;border-radius:4px;padding:1px 8px;white-space:nowrap;margin-left:auto}
        .row-meta{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
        .row-id{font-size:.72rem;font-weight:700;background:#21262d;border-radius:3px;padding:1px 6px;color:#79c0ff}
        .row-shape{font-size:.78rem;font-weight:600;color:#c9d1d9}
        .row-desc{font-size:.72rem;color:#8b949e;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
        .outcome-line{padding:3px 0 1px}
        .outcome-details{font-size:.75rem;color:#8b949e}
        .outcome-details summary{cursor:pointer;color:#8b949e;list-style:none;display:inline}
        .outcome-details summary::-webkit-details-marker{display:none}
        .outcome-details summary::before{content:'▸ ';font-size:.65rem;color:#484f58}
        .outcome-details[open] summary::before{content:'▾ ';color:#58a6ff}
        .outcome-details[open] summary{color:#c9d1d9}
        .outcome-reason{display:block;margin-top:4px;padding:6px 10px;background:#0d1117;border-left:2px solid #30363d;border-radius:0 4px 4px 0;font-size:.73rem;color:#8b949e;line-height:1.55;white-space:pre-wrap}
        .pipeline-row{display:flex;align-items:flex-start;gap:12px}
        .pipeline{display:flex;align-items:flex-start;gap:8px;flex-wrap:nowrap}
        .step{display:flex;flex-direction:column;align-items:center;gap:4px}
        .step-img{width:160px;height:auto;display:block;border:1px solid #30363d;border-radius:4px}
        .step-empty{width:160px;height:120px;background:#21262d;border:1px solid #30363d;border-radius:4px;display:block}
        .step-label{font-size:.68rem;color:#8b949e;text-align:center;max-width:160px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
        .pipeline-score{display:flex;flex-direction:column;align-items:center;gap:6px;padding:8px 0 0;min-width:80px}
        .score-val{font-size:1.3rem;font-weight:700}
        .s-good{color:#56d364}.s-warn{color:#d29922}.s-bad{color:#f85149}
        .bar-bg{width:80px;height:8px;background:#21262d;border-radius:4px;overflow:hidden}
        .bar-fill{height:100%;border-radius:4px}
        .lb-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:1000;align-items:center;justify-content:center;cursor:zoom-out}
        .lb-visible{display:flex}
        .lb-box{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;max-width:90vw;max-height:90vh;overflow:auto;position:relative;cursor:default}
        .lb-close{position:absolute;top:8px;right:10px;background:none;border:none;color:#8b949e;font-size:1.2rem;cursor:pointer;line-height:1}
        .lb-close:hover{color:#c9d1d9}
        .lb-img{display:block;max-width:100%;max-height:80vh;border-radius:4px}
        .lb-caption{font-size:.78rem;color:#8b949e;margin-top:8px;text-align:center}
        .cl-lb-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.88);z-index:1001;align-items:center;justify-content:center;cursor:zoom-out}
        .cl-visible{display:flex!important}
        .cl-lb-box{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;max-width:95vw;max-height:95vh;overflow:auto;position:relative;cursor:default;display:flex;flex-direction:column;gap:10px;align-items:center}
        .cl-lb-header{display:flex;align-items:center;gap:14px;color:#58a6ff;font-weight:bold;font-size:.9rem}
        .cl-counter{color:#484f58;font-size:.82rem;background:#21262d;padding:1px 10px;border-radius:10px;min-width:60px;text-align:center}
        .cl-status{font-size:.78rem;color:#8b949e;min-height:1em;text-align:center}
        .cl-img-wrap{position:relative;display:inline-block}
        .cl-img-wrap img{display:block;max-width:88vw;max-height:62vh;background:#0d1117}
        .cl-svg{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
        .cl-controls{display:flex;align-items:center;gap:16px}
        .cl-nav-btn{background:#21262d;border:1px solid #30363d;color:#79c0ff;padding:5px 14px;border-radius:4px;cursor:pointer;font-size:.85rem}
        .cl-nav-btn:hover{background:#2d333b}
        .cl-info{text-align:center;min-width:220px}
        #cl-label{display:block;color:#c9d1d9;font-weight:bold;font-size:.88rem}
        #cl-cnt{display:block;color:#8b949e;font-size:.74rem}
        .cl-btn{display:block;width:100%;background:#0d1219;border:0;border-top:1px solid #21262d;color:#484f58;font-size:.68rem;padding:3px 0;cursor:pointer;text-align:center;border-radius:0 0 4px 4px}
        .cl-btn:hover{background:#21262d;color:#79c0ff}
        """;
}

