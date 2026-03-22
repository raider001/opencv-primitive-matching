package org.example.utilities;

import org.example.analytics.AnalysisResult;
import org.example.colour.ColourCluster;
import org.example.colour.ExperimentalSceneColourClusters;
import org.example.factories.ReferenceId;
import org.example.factories.ReferenceImageFactory;
import org.example.matchers.SceneDescriptor;
import org.example.matchers.vectormatcher.VectorMatcher;
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
import java.util.Locale;
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
            String allPoints, String sceneAnnot,
            double iou,
            long elapsedMs, long descriptorMs,
            /** Layer 1 — Boundary Count raw score (0–100 %). */
            double layerBoundaryCount,
            /** Layer 2 — Structural Coherence raw score (0–100 %). */
            double layerStructural,
            /** Layer 3 — Geometry raw score (0–100 %). */
            double layerGeometry) {}

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
        docByLabel.clear();
        for (Method m : testClass.getDeclaredMethods()) {
            if (m.getAnnotation(org.junit.jupiter.api.Test.class) == null) continue;
            ExpectedOutcome eo = m.getAnnotation(ExpectedOutcome.class);
            if (eo == null) continue;
            org.junit.jupiter.api.DisplayName dn =
                    m.getAnnotation(org.junit.jupiter.api.DisplayName.class);
            String displayName = (dn != null && !dn.value().isBlank()) ? dn.value() : m.getName();
            TestDoc doc = new TestDoc(m.getName(), displayName, eo.reason());
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
    public void clear() { rows.clear(); docByLabel.clear(); }

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

        // Recolour only white-on-black synthetic scenes (self-match tests) so the
        // report shows the reference colour.  Scenes that already contain coloured
        // shapes on a non-black background (e.g. solid-white, gradient, random-circles)
        // must NOT be recoloured — doing so fills the bright background with the ref
        // colour and turns the entire image into a solid block, hiding all contours.
        Mat sceneWithRef;
        if (rid != null && isWhiteOnBlackScene(sceneMat)) {
            sceneWithRef = recolourToRef(sceneMat, rid);
        } else {
            sceneWithRef = sceneMat.clone();
        }

        // ── Reference images ──────────────────────────────────────────────
        String refOrigPng = "", refPointsPng = "";
        if (rid != null) {
            Mat refOrig = ReferenceImageFactory.build(rid);
            refOrigPng = matToBase64Png(refOrig);

            // Build ref contours from ALL clusters (chromatic + achromatic) so that
            // every colour region and edge boundary is visible — uses the same
            // ExperimentalSceneColourClusters extractor as the scene side (critical invariant).
            List<MatOfPoint> refContours = new ArrayList<>();
            List<ColourCluster> refClusters =
                    ExperimentalSceneColourClusters.INSTANCE.extractFromBorderPixels(refOrig);
            for (ColourCluster c : refClusters) {
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
        // Each cluster is drawn in its actual hue colour (chromatic) or grey
        // (achromatic).  Labels show peak hue and valley-based exclusive bounds.
        // Uses ExperimentalSceneColourClusters.extractFromBorderPixels — the
        // same path as SceneDescriptor.build() — so the report is faithful
        // to what the matcher really works with.
        String allPointsPng;
        {
            List<ColourCluster> clusters =
                    ExperimentalSceneColourClusters.INSTANCE.extractFromBorderPixels(sceneWithRef);
            allPointsPng = buildClusterOverlay(sceneWithRef, clusters);
            clusters.forEach(ColourCluster::release);
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

        AnalysisResult.ScoringLayers layers = (normalResult != null)
                ? normalResult.scoringLayers()
                : AnalysisResult.ScoringLayers.ZERO;
        if (layers == null) layers = AnalysisResult.ScoringLayers.ZERO;

        rows.add(new ReportRow(stage, label, shapeName, sceneDesc,
                score, passed,
                refOrigPng, refPointsPng,
                sceneOrigPng, allPointsPng, sceneAnnotPng,
                iou, elapsedMs, descriptorMs,
                layers.boundaryCount(),
                layers.structural(),
                layers.geometry()));

        sceneWithRef.release();
        return score;
    }

    /**
     * Pass/fail is determined purely by match score and IoU — no annotation dependency.
     *
     * <h3>Detection tests</h3> (label does NOT contain {@code "→"})
     * <ul>
     *   <li>PASS — IoU &gt; 0.9 AND score &gt; 70 %</li>
     *   <li>FAIL — everything else</li>
     * </ul>
     *
     * <p>No upper IoU cap is applied.  The coverage-scaled formula
     * {@code IoU = recall × scale} (where {@code scale = detArea / gtArea ≥ 1}) can
     * legitimately exceed 1.1 for degenerate-aspect shapes such as horizontal or vertical
     * lines whose ground-truth bounding box is only a few pixels tall/wide.  A detected
     * region that fully covers the GT but has a larger bounding box is still a correct
     * detection — the {@code > 0.9} floor already guarantees adequate GT coverage via
     * the recall component; {@code score > 70} guards against weak geometric matches.
     *
     * <h3>Rejection tests</h3> (label contains {@code "→"}, reference absent)
     * <ul>
     *   <li>PASS — score &lt; 60 % (correctly did not fire)</li>
     *   <li>FAIL — score ≥ 60 % (incorrectly fired on absent reference)</li>
     * </ul>
     */
    private boolean determinePassed(String label, double score, double iou) {
        if (label.contains("→")) return isRejectionPass(score);          // rejection: no fire = pass
        return isDetectionPass(score, iou);                               // detection: score + location
    }

    /**
     * Shared detection gate used by both HTML report rows and JUnit assertions.
     *
     * <p>Passes when {@code iou > 0.90} (GT is adequately covered),
     * {@code iou <= 1.30} (detection bbox is at most 1.3× the GT area), and
     * {@code score > 70}.
     *
     * <p>The upper IoU cap prevents accepting detections whose bounding box
     * dwarfs the actual shape — e.g. a thin horizontal line (GT: 342×9) whose
     * detection bbox includes nearby background lines (det: 399×92) producing
     * a coverage-scaled IoU of ~12.  Such detections are geometrically correct
     * (the line IS in the bbox) but positionally wrong (the bbox covers far
     * more than the target shape).
     */
    public static boolean isDetectionPass(double score, double iou) {
        return !Double.isNaN(iou)
                && iou > 0.90
                && iou <= 1.30
                && score > 70.0;
    }

    /** Shared rejection gate used by both HTML report rows and JUnit assertions. */
    public static boolean isRejectionPass(double score) {
        return score < 60.0;
    }

    /**
     * Writes the report as a lightweight {@code report.html} shell that lazily loads
     * per-stage section files from a {@code sections/} sub-directory via iframes.
     *
     * <p>Each stage produces one self-contained {@code sections/&lt;slug&gt;.html} file
     * containing only that stage's rows (with all embedded base64 images).  The main
     * shell holds only the header, collapsible accordion entries, and the
     * {@code <iframe>} placeholders that load on demand when a stage is expanded —
     * keeping the main file tiny regardless of how many images the full suite generates.
     *
     * <p>Iframes auto-size to their content via a {@code postMessage} handshake so no
     * scroll-bars appear inside the frames.  Each section file also has its own
     * independent lightbox, so image zoom works without cross-frame JS.
     *
     * <p>Deletes any existing output at each path before writing.
     */
    public void writeReport(Path outputDir, String title) throws IOException {
        Files.createDirectories(outputDir);
        Path sectionsDir = outputDir.resolve("sections");
        Files.createDirectories(sectionsDir);

        Map<String, List<ReportRow>> byStage = new LinkedHashMap<>();
        for (ReportRow r : rows)
            byStage.computeIfAbsent(r.stage(), k -> new ArrayList<>()).add(r);

        // Write one self-contained HTML file per stage
        Map<String, String> slugMap = new LinkedHashMap<>();
        for (Map.Entry<String, List<ReportRow>> e : byStage.entrySet()) {
            String slug = toSlug(e.getKey());
            slugMap.put(e.getKey(), slug);
            Path sectionFile = sectionsDir.resolve(slug + ".html");
            Files.deleteIfExists(sectionFile);
            Files.writeString(sectionFile,
                    buildSectionHtml(e.getValue(), docByLabel, e.getKey()),
                    StandardCharsets.UTF_8);
            System.out.println("[MatchReportLibrary] Section: " + sectionFile.getFileName());
        }

        // Write the main shell
        Path out = outputDir.resolve("report.html");
        Files.deleteIfExists(out);
        Files.writeString(out, buildMainHtml(rows, byStage, slugMap, title), StandardCharsets.UTF_8);
        System.out.println("[MatchReportLibrary] Report: " + out.toAbsolutePath());

        long total  = rows.size();
        long passed = rows.stream().filter(ReportRow::passed).count();
        System.out.printf("[MatchReportLibrary] %d rows  %d passed  %d failed%n",
                total, passed, total - passed);
    }

    /**
     * Converts a stage name to a filesystem-safe slug used as the section file name.
     * E.g. {@code "BG Random Lines"} → {@code "bg_random_lines"}.
     */
    static String toSlug(String stage) {
        return stage.toLowerCase(Locale.ROOT)
                    .replaceAll("[^a-z0-9]+", "_")
                    .replaceAll("^_+|_+$", "");
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    public static double normalScore(List<AnalysisResult> results) {
        return results.stream()
                .filter(r -> r.methodName().equals(VectorVariant.VECTOR_NORMAL.variantName()))
                .findFirst()
                .map(AnalysisResult::matchScorePercent)
                .orElse(0.0);
    }

    private static Rect findBestBbox(Mat scene, VectorSignature refSig) {
        if (refSig == null) return null;
        return MatchDiagnosticLibrary.allScoredBboxes(scene, List.of(refSig)).stream()
                .max(Comparator.comparingDouble(e -> e[1]))
                .map(e -> new Rect((int)e[0], (int)e[2], (int)e[3], (int)e[4]))
                .orElse(null);
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

    /**
     * Returns {@code true} if the scene looks like a white-on-black synthetic image
     * (self-match tests) where recolouring is appropriate.
     *
     * <p>Heuristic: if fewer than 30 % of pixels are "bright" (grayscale &gt; 240), the
     * scene is predominantly dark and the bright pixels are the foreground shape → safe
     * to recolour.  Scenes on white / light / gradient / noisy backgrounds have &gt; 30 %
     * bright pixels → must NOT be recoloured (the bright pixels are the background, and
     * filling them would erase all contrast).
     */
    private static boolean isWhiteOnBlackScene(Mat sceneMat) {
        Mat grey = new Mat();
        Imgproc.cvtColor(sceneMat, grey, Imgproc.COLOR_BGR2GRAY);
        Mat bright = new Mat();
        Imgproc.threshold(grey, bright, 240, 255, Imgproc.THRESH_BINARY);
        double brightFraction = Core.countNonZero(bright) / (double) (grey.rows() * grey.cols());
        grey.release();
        bright.release();
        return brightFraction < 0.30;
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

    // ── HTML builders ─────────────────────────────────────────────────────────

    /** Builds the lightweight main shell: header + one lazy-loading iframe per stage. */
    private static String buildMainHtml(List<ReportRow> allRows,
                                         Map<String, List<ReportRow>> byStage,
                                         Map<String, String> slugMap,
                                         String title) {
        long total  = allRows.size();
        long passed = allRows.stream().filter(ReportRow::passed).count();
        String ts   = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));

        StringBuilder sb = new StringBuilder();
        sb.append("<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>")
          .append("<title>").append(esc(title)).append("</title>")
          .append("<style>").append(MAIN_CSS).append("</style>")
          .append("</head><body>")
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
          .append("<span class='legend-pill pass-pill'>green = passed (IoU &gt; 0.9 and score &gt; 70%)</span>")
          .append("<span class='legend-pill fail-pill'>red = failed</span>")
          .append("</div></div>")
          .append("<div class='legend-block'><div class='legend-title'>IoU badge colours</div>")
          .append("<div class='legend-row'>")
          .append("<span class='legend-pill iou-pill-exc'>blue IoU ≥ 1.0 exact/near-exact</span>")
          .append("<span class='legend-pill iou-pill-inf'>purple ⚑ IoU &gt; 1.1 — inflated (thin GT bbox, expected for line shapes)</span>")
          .append("<span class='legend-pill iou-pill-gd'>green IoU ≥ 0.8</span>")
          .append("<span class='legend-pill iou-pill-warn'>yellow IoU ≥ 0.5</span>")
          .append("<span class='legend-pill iou-pill-bad'>red IoU &lt; 0.5</span>")
          .append("</div></div>")
          .append("</div>"); // end .header

        for (Map.Entry<String, List<ReportRow>> e : byStage.entrySet()) {
            List<ReportRow> stageRows = e.getValue();
            long stageTotal  = stageRows.size();
            long stagePassed = stageRows.stream().filter(ReportRow::passed).count();
            long stageFailed = stageTotal - stagePassed;
            int barW = (int) Math.max(1, Math.min(100, stageTotal == 0 ? 0 : stagePassed * 100 / stageTotal));
            String slug = slugMap.get(e.getKey());

            sb.append("<section>")
              .append("<details class='stage-details' ontoggle='loadSection(this)'>")
              .append("<summary class='stage-summary'>")
              .append("<span class='stage-chevron'></span>")
              .append("<span class='stage-name'>").append(esc(e.getKey())).append("</span>")
              .append("<span class='stage-stats'>")
              .append("<span class='stat-pass'>✔ ").append(stagePassed).append(" passed</span>")
              .append("<span class='stat-sep'>·</span>")
              .append("<span class='stat-fail'>✘ ").append(stageFailed).append(" failed</span>")
              .append("<span class='stat-sep'>·</span>")
              .append("<span class='stat-total'>").append(stageTotal).append(" total</span>")
              .append("</span>")
              .append("<div class='stage-bar-bg'><div class='stage-bar-fill' style='width:").append(barW).append("%'></div></div>")
              .append("</summary>")
              .append("<div class='frame-wrap'>")
              .append("<div class='frame-loading'>⏳ Loading section…</div>")
              .append("<iframe class='section-frame' data-src='sections/").append(slug).append(".html'></iframe>")
              .append("</div>")
              .append("</details></section>");
        }

        // Lazy iframe loading + auto-resize + lightbox relay via postMessage
        sb.append("<script>")
          .append("window.addEventListener('message',function(e){")
          .append("  if(!e.data)return;")
          .append("  if(e.data.type==='section-resize'){")
          .append("    document.querySelectorAll('.section-frame').forEach(function(f){")
          .append("      try{if(f.contentWindow===e.source){")
          .append("        f.style.height=(e.data.height+4)+'px';")
          .append("        var w=f.closest('.frame-wrap');")
          .append("        if(w){var l=w.querySelector('.frame-loading');if(l)l.style.display='none';}")
          .append("      }}catch(err){}")
          .append("    });")
          .append("  } else if(e.data.type==='open-lightbox'){")
          .append("    document.getElementById('lb-img').src=e.data.src;")
          .append("    document.getElementById('lb-caption').textContent=e.data.caption||'';")
          .append("    document.getElementById('lb').classList.add('lb-visible');")
          .append("  }")
          .append("});")
          .append("function loadSection(det){")
          .append("  if(!det.open)return;")
          .append("  var f=det.querySelector('.section-frame');")
          .append("  if(!f.getAttribute('src')){f.src=f.dataset.src;}")
          .append("}")
          .append("function closeLb(){document.getElementById('lb').classList.remove('lb-visible');}")
          .append("document.addEventListener('keydown',function(e){if(e.key==='Escape')closeLb();});")
          .append("</script>")
          // Global lightbox — position:fixed is relative to the real browser viewport here
          .append("<div id='lb' class='lb-overlay' onclick='closeLb()'>")
          .append("<div class='lb-box' onclick='event.stopPropagation()'>")
          .append("<button class='lb-close' onclick='closeLb()'>✕</button>")
          .append("<img id='lb-img' src='' alt='' class='lb-img'/>")
          .append("<div id='lb-caption' class='lb-caption'></div></div></div>")
          .append("</body></html>");
        return sb.toString();
    }

    /**
     * Builds a self-contained section HTML file containing all rows for one stage.
     * Has its own CSS, lightbox, and a {@code postMessage} call to report its height
     * to the parent shell so the enclosing iframe auto-sizes.
     */
    private static String buildSectionHtml(List<ReportRow> stageRows,
                                            Map<String, TestDoc> docByLabel,
                                            String stageName) {
        StringBuilder sb = new StringBuilder();
        sb.append("<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>")
          .append("<title>").append(esc(stageName)).append("</title>")
          .append("<style>").append(SECTION_CSS).append("</style>")
          .append("</head><body>");

        for (ReportRow r : stageRows) {
            String cls = r.passed() ? "row pass" : "row fail";
            sb.append("<div class='").append(cls).append("'>")
              .append("<div class='row-meta'>")
              .append("<span class='row-id'>").append(esc(r.label())).append("</span>")
              .append("<span class='row-shape'>").append(esc(r.shapeName())).append("</span>")
              .append("<span class='row-desc'>").append(esc(r.sceneDesc())).append("</span>");
            if (!Double.isNaN(r.iou())) {
                String ic = r.iou() > 1.1    ? "iou-inflated"
                          : r.iou() >= 1.0   ? "iou-excellent"
                          : r.iou() >= 0.8   ? "iou-good"
                          : r.iou() >= 0.5   ? "iou-warn"
                          :                    "iou-bad";
                String iouText = String.format("%.2f", r.iou());
                if (r.iou() > 1.1) iouText += " ⚑";   // flag: inflated due to thin GT bbox
                sb.append("<span class='iou-val ").append(ic).append("'>IoU ")
                  .append(iouText).append("</span>");
            }
            sb.append("<span class='timing-badge'>")
              .append("desc:").append(r.descriptorMs()).append("ms")
              .append(" match:").append(r.elapsedMs()).append("ms</span>")
              .append("</div>");

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
            step(sb, r.sceneOrig(),  "Scene");
            step(sb, r.allPoints(),  "Clusters (hue-coloured)");
            step(sb, r.sceneAnnot(), "Match");
            sb.append("</div>")
              .append("<div class='pipeline-score'>")
              .append("<span class='score-val ").append(r.score()>=70?"s-good":r.score()>=40?"s-warn":"s-bad").append("'>")
              .append(String.format("%.1f%%",r.score())).append("</span>")
              .append(bar(r.score()))
              .append("<div class='layers-breakdown'>")
              .append(layerRow("Boundary Count", r.layerBoundaryCount(), "×0.15"))
              .append(layerRow("Structural",     r.layerStructural(),    "×0.25"))
              .append(layerRow("Geometry",       r.layerGeometry(),      "×0.60"))
              .append("</div>")
              .append("</div></div></div>");
        }

        // Lightbox JS — delegates to parent shell's global lightbox via postMessage
        // so position:fixed is relative to the real browser viewport, not the iframe.
        sb.append("<script>")
          .append("function openLb(s,c){window.parent.postMessage({type:'open-lightbox',src:s,caption:c},'*');}")
          .append("function _reportH(){")
          .append("  window.parent.postMessage({type:'section-resize',height:document.body.scrollHeight},'*');")
          .append("}")
          .append("window.addEventListener('load',_reportH);")
          .append("new ResizeObserver(_reportH).observe(document.body);")
          .append("</script></body></html>");
        return sb.toString();
    }

    /**
     * Builds a cluster-coloured contour overlay image.
     *
     * <p>Each colour cluster's contours are drawn in a colour derived from the
     * cluster's actual peak hue — so a red cluster's contours appear red, a blue
     * cluster's appear blue, etc.  Achromatic clusters use light-grey (bright) or
     * mid-grey (dark).  A small label above each cluster group shows the peak hue
     * and the valley-based exclusive bounds: {@code H=30 [16..44]}.
     *
     * <p>This makes the valley-based exclusive assignment visually verifiable:
     * adjacent clusters should have touching (not overlapping) hue windows, and
     * each boundary pixel should appear in only one cluster's colour.
     */
    private static String buildClusterOverlay(Mat scene,
                                               List<ColourCluster> clusters) {
        Mat overlay = Mat.zeros(scene.rows(), scene.cols(), CvType.CV_8UC3);
        // Dim the original scene as a subtle underlay so shapes are identifiable
        Mat dimmed = new Mat();
        scene.convertTo(dimmed, -1, 0.20, 0);
        Core.add(overlay, dimmed, overlay);
        dimmed.release();

        for (ColourCluster c : clusters) {
            List<MatOfPoint> contours = SceneDescriptor.contoursFromMask(c.mask);
            if (contours.isEmpty()) continue;

            // ── Determine display colour ──────────────────────────────────
            Scalar colour;
            String label;
            if (c.achromatic) {
                colour = c.brightAchromatic
                        ? new Scalar(220, 220, 220)  // bright → white-ish
                        : new Scalar(100, 100, 100); // dark  → mid-grey
                label  = c.brightAchromatic ? "bright-ach" : "dark-ach";
            } else {
                // Convert peak hue to a saturated BGR display colour
                Mat hsvPx = new Mat(1, 1, CvType.CV_8UC3, new Scalar(c.hue, 210, 220));
                Mat bgrPx = new Mat();
                Imgproc.cvtColor(hsvPx, bgrPx, Imgproc.COLOR_HSV2BGR);
                double[] bgr = bgrPx.get(0, 0);
                colour = new Scalar(bgr[0], bgr[1], bgr[2]);
                hsvPx.release(); bgrPx.release();
                label = String.format("H=%.0f [%d..%d]", c.hue, c.loBound, c.hiBound);
            }

            // ── Draw contours in cluster colour ───────────────────────────
            for (MatOfPoint cnt : contours) {
                Imgproc.drawContours(overlay, List.of(cnt), 0, colour, 1);
            }

            // ── Label at the bounding-box top-left of the first contour ───
            Rect bb = Imgproc.boundingRect(contours.getFirst());
            int lx = Math.max(2, bb.x);
            int ly = Math.max(10, bb.y - 3);
            Imgproc.putText(overlay, label, new Point(lx, ly),
                    Imgproc.FONT_HERSHEY_PLAIN, 0.75, colour, 1);

            contours.forEach(MatOfPoint::release);
        }

        String png = matToBase64Png(overlay);
        overlay.release();
        return png;
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

    /** Renders one row of the per-layer scoring breakdown (name | mini-bar | pct weight). */
    private static String layerRow(String name, double scorePercent, String weight) {
        String col = scorePercent>=70?"#56d364":scorePercent>=40?"#d29922":"#f85149";
        int w = (int)Math.max(1,Math.min(100,scorePercent));
        return "<div class='layer-row'>"
             + "<span class='layer-name'>" + esc(name) + "</span>"
             + "<div class='layer-bar-bg'><div class='layer-bar-fill' style='width:"
             + w + "%;background:" + col + "'></div></div>"
             + "<span class='layer-pct'>" + String.format("%.0f%%", scorePercent)
             + " <span class='layer-wt'>" + weight + "</span></span>"
             + "</div>";
    }

    private static String esc(String s) {
        return s==null?"":s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;");
    }

    // ── Stylesheets ───────────────────────────────────────────────────────────

    /** CSS for the main shell (report.html): header, legend, accordion, iframe wrappers. */
    private static final String MAIN_CSS = """
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
        .iou-pill-exc{border-left-color:#388bfd;background:#0a1628;color:#79c0ff}
        .iou-pill-inf{border-left-color:#9a6ed8;background:#1a0d2e;color:#bc8cff}
        .iou-pill-gd{border-left-color:#238636;background:#0d2619;color:#56d364}
        .iou-pill-warn{border-left-color:#9e6a03;background:#271c00;color:#d29922}
        .iou-pill-bad{border-left-color:#da3633;background:#2b0c0c;color:#f85149}
        section{padding:0 24px 16px}
        /* ── Collapsible stage sections ──────────────────────────────── */
        .stage-details{background:#161b22;border:1px solid #30363d;border-radius:8px;margin-bottom:12px}
        .stage-summary{display:flex;align-items:center;gap:10px;padding:10px 14px;cursor:pointer;list-style:none;user-select:none;border-radius:8px}
        .stage-summary::-webkit-details-marker{display:none}
        .stage-summary:hover{background:#21262d}
        .stage-details[open]>.stage-summary{border-bottom:1px solid #30363d;border-radius:8px 8px 0 0}
        .stage-chevron{width:14px;height:14px;flex-shrink:0;border:2px solid #8b949e;border-radius:3px;position:relative;transition:transform .15s}
        .stage-chevron::after{content:'';position:absolute;top:50%;left:50%;transform:translate(-50%,-60%) rotate(45deg);width:5px;height:5px;border-right:2px solid #8b949e;border-bottom:2px solid #8b949e}
        .stage-details[open]>.stage-summary .stage-chevron::after{transform:translate(-50%,-40%) rotate(-135deg)}
        .stage-name{color:#79c0ff;font-size:.95rem;font-weight:700;white-space:nowrap}
        .stage-stats{display:flex;align-items:center;gap:7px;font-size:.75rem;margin-left:4px}
        .stat-pass{color:#56d364;font-weight:600}
        .stat-fail{color:#f85149;font-weight:600}
        .stat-total{color:#8b949e}
        .stat-sep{color:#484f58}
        .stage-bar-bg{flex:1;height:6px;background:#21262d;border-radius:3px;overflow:hidden;min-width:60px;margin-left:8px}
        .stage-bar-fill{height:100%;background:#238636;border-radius:3px;transition:width .3s}
        /* ── iframe section wrapper ──────────────────────────────────── */
        .frame-wrap{position:relative;background:#0d1117}
        .frame-loading{padding:20px;text-align:center;color:#8b949e;font-size:.82rem}
        .section-frame{display:block;width:100%;border:none;overflow:hidden;background:#0d1117}
        /* ── Global lightbox (lives in main shell — position:fixed is real viewport) ── */
        .lb-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:1000;align-items:center;justify-content:center;cursor:zoom-out}
        .lb-visible{display:flex}
        .lb-box{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;max-width:90vw;max-height:90vh;overflow:auto;position:relative;cursor:default}
        .lb-close{position:absolute;top:8px;right:10px;background:none;border:none;color:#8b949e;font-size:1.2rem;cursor:pointer;line-height:1}
        .lb-close:hover{color:#c9d1d9}
        .lb-img{display:block;max-width:100%;max-height:80vh;border-radius:4px}
        .lb-caption{font-size:.78rem;color:#8b949e;margin-top:8px;text-align:center}
        """;

    /** CSS for section files (sections/*.html): rows, pipeline steps, scores, lightbox. */
    private static final String SECTION_CSS = """
        *{box-sizing:border-box;margin:0;padding:0}
        body{font-family:system-ui,sans-serif;background:#0d1117;color:#c9d1d9;padding:4px 12px 12px}
        .row{background:#161b22;border:1px solid #30363d;border-radius:8px;margin-bottom:8px;padding:10px 12px;display:flex;flex-direction:column;gap:6px}
        .row.pass{border-left:3px solid #238636}
        .row.fail{border-left:3px solid #da3633}
        .iou-val{font-size:.72rem;font-weight:600;white-space:nowrap}
        .iou-excellent{color:#79c0ff}.iou-good{color:#56d364}.iou-warn{color:#d29922}.iou-bad{color:#f85149}
        .iou-inflated{color:#bc8cff}  /* > 1.1 — inflated due to thin GT bbox (expected for line shapes) */
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
        .pipeline-score{display:flex;flex-direction:column;align-items:center;gap:6px;padding:8px 0 0;min-width:100px}
        .score-val{font-size:1.3rem;font-weight:700}
        .s-good{color:#56d364}.s-warn{color:#d29922}.s-bad{color:#f85149}
        .bar-bg{width:80px;height:8px;background:#21262d;border-radius:4px;overflow:hidden}
        .bar-fill{height:100%;border-radius:4px}
        .layers-breakdown{width:100%;display:flex;flex-direction:column;gap:3px;margin-top:2px}
        .layer-row{display:flex;align-items:center;gap:4px}
        .layer-name{font-size:.60rem;color:#8b949e;width:82px;flex-shrink:0;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
        .layer-bar-bg{flex:1;height:4px;background:#21262d;border-radius:2px;overflow:hidden;min-width:30px}
        .layer-bar-fill{height:100%;border-radius:2px}
        .layer-pct{font-size:.60rem;color:#c9d1d9;min-width:52px;text-align:right;white-space:nowrap}
        .layer-wt{color:#484f58}
        """;
}

