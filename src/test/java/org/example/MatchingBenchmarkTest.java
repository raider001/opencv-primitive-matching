package org.example;

import org.junit.jupiter.api.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.regex.*;
import java.util.stream.*;

/**
 * Milestone 20 — Unified Benchmark Report (collator + analyser).
 *
 * <p>Reads the {@code <script type="application/json" id="report-meta">} block
 * embedded in each per-technique {@code report.html} by {@link HtmlReportWriter}
 * and assembles a cross-technique analysis covering:
 * <ul>
 *   <li>Overall ranking by composite score (all base + CF + CF1 variants)</li>
 *   <li>Scenario-variant breakdown — accuracy per scale/rotation/degradation variant</li>
 *   <li>CF / CF1 uplift comparison — every tier side-by-side per technique</li>
 *   <li>Automated recommendation: best overall, fastest, most accurate, best CF uplift</li>
 * </ul>
 *
 * <h2>Composite score</h2>
 * {@code composite = 0.45 × accuracy + 0.30 × F1 + 0.25 × speedScore}
 * where {@code speedScore = 100 × min(1, 10ms / avgMs)}.
 *
 * <p>Output: {@code test_output/benchmark/report.html}
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
@DisplayName("Milestone 20 — Unified Benchmark Report (collator)")
public class MatchingBenchmarkTest {

    private static final Path BENCHMARK_DIR = Paths.get("test_output", "benchmark");

    // Ordered display groups for scenario-variant tables
    private static final List<String> SCALE_VARIANTS = List.of(
            "scale_0.50", "scale_0.75", "scale_1.25", "scale_1.50", "scale_2.00",
            "scale1.5_rot45", "scale0.75_rot30");
    private static final List<String> ROT_VARIANTS = List.of(
            "rot_15", "rot_30", "rot_45", "rot_90", "rot_180");
    private static final List<String> DEGRAD_VARIANTS = List.of(
            "noise_s10", "noise_s25", "contrast_40pct", "blur_5x5",
            "occ_25pct", "occ_50pct", "hue_shift_40");
    private static final List<String> OFFSET_VARIANTS = List.of(
            "offset_topleft", "offset_botright", "offset_random42");
    private static final List<String> CLEAN_VARIANTS = List.of(
            "clean_bg_solid_black", "clean_bg_noise_light",
            "clean_bg_gradient_h_colour", "clean_bg_random_mixed");

    /** All known per-technique report paths, in presentation order. */
    private static final List<TechniqueReport> KNOWN_REPORTS = List.of(
        new TechniqueReport("Template Matching",            Paths.get("test_output","template_matching",      "report.html"), "TM",      true,  true),
        new TechniqueReport("Feature Matching",             Paths.get("test_output","feature_matching",       "report.html"), "FM",      true,  false),
        new TechniqueReport("Contour Shape Matching",       Paths.get("test_output","contour_shape_matching", "report.html"), "CSM",     true,  false),
        new TechniqueReport("Hough Transforms",             Paths.get("test_output","hough_transforms",       "report.html"), "HOUGH",   true,  false),
        new TechniqueReport("Generalized Hough",            Paths.get("test_output","generalized_hough",      "report.html"), "GHT",     true,  false),
        new TechniqueReport("Histogram Comparison",         Paths.get("test_output","histogram_matching",     "report.html"), "HIST",    true,  false),
        new TechniqueReport("Phase Correlation",            Paths.get("test_output","phase_correlation",      "report.html"), "PC",      true,  false),
        new TechniqueReport("Morphology Analysis",          Paths.get("test_output","morphology_analysis",    "report.html"), "MORPH",   true,  false),
        new TechniqueReport("Pixel Diff",                   Paths.get("test_output","pixel_diff",             "report.html"), "PDIFF",   true,  false),
        new TechniqueReport("SSIM",                         Paths.get("test_output","ssim_matching",          "report.html"), "SSIM",    true,  false),
        new TechniqueReport("Chamfer Distance",             Paths.get("test_output","chamfer_matching",       "report.html"), "CHAMFER", true,  false),
        new TechniqueReport("Fourier Shape Descriptors",    Paths.get("test_output","fourier_shape_matching", "report.html"), "FSM",     true,  false),
        new TechniqueReport("Colour-First Region Proposal", Paths.get("test_output","colour_first",           "report.html"), "CF1",     false, true),
        new TechniqueReport("Multi-Colour-First Proposal",  Paths.get("test_output","multi_colour_first",     "report.html"), "MCF1",    false, false)
    );

    @BeforeAll
    void setup() throws IOException {
        Files.createDirectories(BENCHMARK_DIR.toAbsolutePath().normalize());
    }

    @Test @Order(1)
    @DisplayName("Collate and analyse per-technique reports into unified benchmark HTML")
    void collateBenchmarkReport() throws IOException {
        Path absOut = BENCHMARK_DIR.resolve("report.html").toAbsolutePath().normalize();

        List<TechniqueReport> available = KNOWN_REPORTS.stream()
                .filter(r -> Files.exists(r.path())).toList();
        List<TechniqueReport> missing = KNOWN_REPORTS.stream()
                .filter(r -> !Files.exists(r.path())).toList();

        System.out.printf("%n[BENCHMARK] Found %d / %d technique reports.%n",
                available.size(), KNOWN_REPORTS.size());
        if (!missing.isEmpty()) {
            System.out.printf("[BENCHMARK] Missing:%n");
            missing.forEach(r -> System.out.printf("  - %s%n", r.name()));
        }

        List<ReportMeta> metas = new ArrayList<>();
        for (TechniqueReport tr : available) {
            ReportMeta meta = parseReportMeta(tr);
            if (meta != null) {
                metas.add(meta);
                long varCount = meta.variants().stream()
                        .filter(v -> v.cfTier().equals("BASE")).count();
                long cfCount  = meta.variants().size() - varCount;
                System.out.printf("[BENCHMARK]  %-32s  %d base + %d CF variants, "
                        + "best acc=%.1f%% best F1=%.1f%%%n",
                        tr.name(), varCount, cfCount,
                        meta.variants().stream().mapToDouble(VariantMeta::accuracyPercent).max().orElse(0),
                        meta.variants().stream().mapToDouble(VariantMeta::f1Percent).max().orElse(0));
            } else {
                System.out.printf("[BENCHMARK]  %-32s  (no metadata — re-run test to embed it)%n",
                        tr.name());
            }
        }

        String html = buildBenchmarkHtml(available, missing, metas, absOut.getParent());
        Files.write(absOut, html.getBytes(StandardCharsets.UTF_8));
        System.out.printf("[BENCHMARK] Report written: %s%n", absOut);
    }

    // =========================================================================
    // JSON metadata parser
    // =========================================================================

    private static ReportMeta parseReportMeta(TechniqueReport tr) {
        try {
            String html = Files.readString(tr.path(), StandardCharsets.UTF_8);
            int start = html.indexOf("\"report-meta\">");
            if (start < 0) return null;
            start += "\"report-meta\">".length();
            int end = html.indexOf("</script>", start);
            if (end < 0) return null;
            return parseJson(tr, html.substring(start, end).trim());
        } catch (IOException e) { return null; }
    }

    private static ReportMeta parseJson(TechniqueReport tr, String json) {
        String technique = extractStr(json, "technique");
        int totalResults = (int) extractDbl(json, "totalResults");

        // Collect all scene variant keys from "sceneVariants" array
        List<String> sceneVariantKeys = new ArrayList<>();
        int svStart = json.indexOf("\"sceneVariants\":[");
        if (svStart >= 0) {
            int svEnd = json.indexOf("]", svStart);
            String svArr = json.substring(svStart + "\"sceneVariants\":[".length(), svEnd);
            Matcher m = Pattern.compile("\"([^\"]+)\"").matcher(svArr);
            while (m.find()) sceneVariantKeys.add(m.group(1));
        }

        List<VariantMeta> variants = new ArrayList<>();
        int varStart = json.indexOf("\"variants\":[");
        if (varStart < 0) return new ReportMeta(tr, technique, totalResults, variants, sceneVariantKeys);
        String varArray = json.substring(varStart + "\"variants\":[".length());

        int depth = 0, objStart = -1;
        for (int i = 0; i < varArray.length(); i++) {
            char c = varArray.charAt(i);
            if (c == '{') { if (depth++ == 0) objStart = i; }
            else if (c == '}') {
                if (--depth == 0 && objStart >= 0) {
                    String obj = varArray.substring(objStart, i + 1);
                    // byCategory block
                    Map<String, SceneBucket> byCategory = parseNamedBuckets(obj, "byCategory");
                    // byVariant block
                    Map<String, SceneBucket> byVariant  = parseNamedBuckets(obj, "byVariant");

                    variants.add(new VariantMeta(
                            extractStr(obj, "name"),
                            extractStr(obj, "cfTier"),
                            extractDbl(obj, "avgScorePercent"),
                            extractDbl(obj, "accuracyPercent"),
                            extractDbl(obj, "precisionPercent"),
                            extractDbl(obj, "recallPercent"),
                            extractDbl(obj, "f1Percent"),
                            extractDbl(obj, "avgMs"),
                            (long) extractDbl(obj, "correct"),
                            (long) extractDbl(obj, "wrongLoc"),
                            (long) extractDbl(obj, "missed"),
                            (long) extractDbl(obj, "falseAlarm"),
                            (long) extractDbl(obj, "rejected"),
                            byCategory,
                            byVariant
                    ));
                    objStart = -1;
                }
            }
        }
        return new ReportMeta(tr, technique, totalResults, variants, sceneVariantKeys);
    }

    /** Parses a {@code "blockName":{"key1":{...},"key2":{...}}} into a map of SceneBucket. */
    private static Map<String, SceneBucket> parseNamedBuckets(String obj, String blockName) {
        Map<String, SceneBucket> map = new LinkedHashMap<>();
        int bStart = obj.indexOf("\"" + blockName + "\":{");
        if (bStart < 0) return map;
        bStart += ("\"" + blockName + "\":{").length();
        // Walk to find each "key":{...} entry at depth 1
        int depth = 1, keyStart = bStart;
        for (int i = bStart; i < obj.length() && depth > 0; i++) {
            char c = obj.charAt(i);
            if (c == '{') depth++;
            else if (c == '}') { if (--depth == 0) break; }
        }
        // Now re-parse with regex for each "label":{...acc...f1...}
        String block = obj.substring(bStart);
        // Find each key → value object
        Pattern keyPat = Pattern.compile("\"([^\"]+)\":\\{([^}]+)\\}");
        Matcher m = keyPat.matcher(block);
        while (m.find()) {
            String key = m.group(1);
            String val = "{" + m.group(2) + "}";
            map.put(key, new SceneBucket(
                    extractDbl(val, "acc"), extractDbl(val, "f1"),
                    extractDbl(val, "prec"), extractDbl(val, "rec"),
                    (long) extractDbl(val, "c"), (long) extractDbl(val, "m"),
                    (long) extractDbl(val, "fa")));
        }
        return map;
    }

    private static String extractStr(String json, String key) {
        Pattern p = Pattern.compile("\"" + key + "\"\\s*:\\s*\"([^\"\\\\]*(\\\\.[^\"\\\\]*)*)\"");
        Matcher m = p.matcher(json);
        return m.find() ? m.group(1).replace("\\\"", "\"").replace("\\\\", "\\") : "";
    }
    private static double extractDbl(String json, String key) {
        Pattern p = Pattern.compile("\"" + key + "\"\\s*:\\s*(-?[0-9]+\\.?[0-9]*)");
        Matcher m = p.matcher(json);
        return m.find() ? Double.parseDouble(m.group(1)) : 0.0;
    }

    // =========================================================================
    // HTML top-level builder
    // =========================================================================

    private String buildBenchmarkHtml(List<TechniqueReport> available,
                                       List<TechniqueReport> missing,
                                       List<ReportMeta> metas,
                                       Path reportDir) {
        String generated = LocalDateTime.now()
                .format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        StringBuilder sb = new StringBuilder();
        sb.append("<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>");
        sb.append("<meta name='viewport' content='width=device-width,initial-scale=1'>");
        sb.append("<title>Pattern Matching — Unified Benchmark</title>");
        sb.append("<style>").append(css()).append("</style></head><body>");

        sb.append("<div class='header'>");
        sb.append("<h1>Pattern Matching — Unified Benchmark</h1>");
        sb.append("<p class='subtitle'>Generated ").append(generated)
          .append(" &nbsp;·&nbsp; ").append(available.size()).append(" / ")
          .append(KNOWN_REPORTS.size()).append(" techniques</p></div>");

        sb.append("<div class='tab-bar'>");
        sb.append("<button class='tab-btn active' onclick=\"switchTab('analysis')\">📊 Analysis</button>");
        sb.append("<button class='tab-btn' onclick=\"switchTab('scenarios')\">🔄 Scenarios</button>");
        sb.append("<button class='tab-btn' onclick=\"switchTab('cf')\">🎨 CF / CF1</button>");
        sb.append("<button class='tab-btn' onclick=\"switchTab('overview')\">📋 Index</button>");
        for (TechniqueReport r : available) {
            sb.append("<button class='tab-btn' onclick=\"switchTab('").append(r.tag())
              .append("')\">").append(r.name()).append("</button>");
        }
        sb.append("</div>");

        sb.append("<div id='tab-analysis'  class='tab-content active'>").append(buildAnalysisTab(metas, missing)).append("</div>");
        sb.append("<div id='tab-scenarios' class='tab-content'>").append(buildScenarioTab(metas)).append("</div>");
        sb.append("<div id='tab-cf'        class='tab-content'>").append(buildCfTab(metas)).append("</div>");
        sb.append("<div id='tab-overview'  class='tab-content'>").append(buildOverviewTab(available, missing, reportDir)).append("</div>");

        for (TechniqueReport r : available) {
            String rel = reportDir.relativize(r.path().toAbsolutePath().normalize())
                    .toString().replace('\\', '/');
            sb.append("<div id='tab-").append(r.tag()).append("' class='tab-content'>");
            sb.append("<div class='iframe-header'><strong>").append(r.name()).append("</strong>");
            sb.append(" &nbsp;<a href='").append(rel).append("' target='_blank'>Open standalone ↗</a></div>");
            sb.append("<iframe src='").append(rel).append("' class='technique-frame' title='")
              .append(r.name()).append("'></iframe></div>");
        }

        sb.append("<script>function switchTab(id){");
        sb.append("document.querySelectorAll('.tab-content').forEach(e=>e.classList.remove('active'));");
        sb.append("document.querySelectorAll('.tab-btn').forEach(e=>e.classList.remove('active'));");
        sb.append("var el=document.getElementById('tab-'+id);if(el)el.classList.add('active');");
        sb.append("document.querySelectorAll('.tab-btn').forEach(b=>{");
        sb.append("if(b.getAttribute('onclick').includes(\"'\"+id+\"'\"))b.classList.add('active');});");
        sb.append("}</script></body></html>");
        return sb.toString();
    }

    // =========================================================================
    // Tab 1 — Overall Analysis (base variants only in main ranking)
    // =========================================================================

    private String buildAnalysisTab(List<ReportMeta> metas, List<TechniqueReport> missing) {
        if (metas.isEmpty()) return noDataMsg();
        StringBuilder sb = new StringBuilder();
        sb.append("<h2>Cross-Technique Analysis</h2>");
        sb.append("<p>Ranked by <b>composite = 45% Accuracy + 30% F1 + 25% Speed</b>. ");
        sb.append("Speed score = 100 × min(1, 10ms/avgMs). ");
        sb.append("Best <em>base</em> variant per technique shown — see the ");
        sb.append("<button class='link-btn' onclick=\"switchTab('cf')\">🎨 CF / CF1 tab</button> ");
        sb.append("for colour-filter comparisons and the ");
        sb.append("<button class='link-btn' onclick=\"switchTab('scenarios')\">🔄 Scenarios tab</button> ");
        sb.append("for per-variant (scale/rotation/degradation) breakdown.</p>");

        // Collect best-base row per technique
        List<RankRow> rows = buildRankRows(metas, "BASE");
        rows.sort(Comparator.comparingDouble(RankRow::composite).reversed());
        sb.append("<h3>🏆 Overall Ranking — Base Variants</h3>");
        sb.append(rankTable(rows));

        // Verdict breakdown
        sb.append("<h3>🔍 Verdict Breakdown</h3>");
        sb.append(verdictBreakdownTable(rows));

        // Recommendation box
        sb.append("<h3>💡 Recommendation</h3>");
        sb.append(recommendationBox(rows, metas));

        if (!missing.isEmpty()) {
            sb.append("<p class='missing-note'>⏳ <strong>").append(missing.size())
              .append(" technique(s) not yet run:</strong> ")
              .append(missing.stream().map(TechniqueReport::name).collect(Collectors.joining(", ")))
              .append("</p>");
        }
        return sb.toString();
    }

    // =========================================================================
    // Tab 2 — Scenario Variants breakdown
    // =========================================================================

    private String buildScenarioTab(List<ReportMeta> metas) {
        if (metas.isEmpty()) return noDataMsg();
        StringBuilder sb = new StringBuilder();
        sb.append("<h2>Scenario Variant Breakdown</h2>");
        sb.append("<p>Accuracy% per scene variant for every method variant (base + CF + CF1). ");
        sb.append("Each cell shows accuracy = (Correct + Correctly&thinsp;Rejected) / Total. ");
        sb.append("🟢 ≥80% &nbsp; 🟡 ≥60% &nbsp; 🔴 &lt;60% &nbsp; — = no data. ");
        sb.append("Best accuracy across <em>all</em> variants in each row is <b>bolded</b>.</p>");

        // Collect every distinct method variant across all techniques, grouped by tier
        // Format: "Technique · VariantName"
        List<MethodCol> cols = buildAllMethodCols(metas);

        sb.append(scenarioGroupTable("🧼 Category A — Clean Scenes",   CLEAN_VARIANTS,  cols, metas));
        sb.append(scenarioGroupTable("📐 Category B — Scale Variants", SCALE_VARIANTS,  cols, metas));
        sb.append(scenarioGroupTable("🔄 Category B — Rotation Variants", ROT_VARIANTS, cols, metas));
        sb.append(scenarioGroupTable("📍 Category B — Offset Variants", OFFSET_VARIANTS, cols, metas));
        sb.append(scenarioGroupTable("🌩 Category C — Degraded Scenes", DEGRAD_VARIANTS, cols, metas));

        // Category-level summary (A/B/C/D)
        sb.append("<h3>Category-Level Summary (all method variants)</h3>");
        sb.append("<p>Average accuracy per category for every method variant.</p>");
        sb.append(categoryTable(cols, metas));

        return sb.toString();
    }

    /** Builds a scenario-group table for a list of variant labels × all method columns. */
    private String scenarioGroupTable(String title, List<String> variantLabels,
                                       List<MethodCol> cols, List<ReportMeta> metas) {
        // Filter to only variant labels that appear in any meta
        Set<String> available = metas.stream()
                .flatMap(m -> m.variants().stream())
                .flatMap(v -> v.byVariant().keySet().stream())
                .collect(Collectors.toSet());
        List<String> present = variantLabels.stream().filter(available::contains).toList();
        if (present.isEmpty()) return "";

        StringBuilder sb = new StringBuilder();
        sb.append("<h3>").append(title).append("</h3>");
        sb.append("<div class='scroll-wrap'><table class='scenario-table'><thead>");
        sb.append("<tr><th>Variant</th>");
        for (MethodCol c : cols) {
            sb.append("<th class='col-hdr' title='").append(c.fullName()).append("'>")
              .append(c.shortName()).append("</th>");
        }
        sb.append("</tr></thead><tbody>");

        for (String vl : present) {
            // Find best accuracy across all cols for bolding
            double bestAcc = cols.stream()
                    .mapToDouble(c -> getAcc(c, vl, metas))
                    .filter(v -> v >= 0).max().orElse(-1);

            sb.append("<tr><td class='variant-label'>").append(vl).append("</td>");
            for (MethodCol c : cols) {
                double acc = getAcc(c, vl, metas);
                sb.append(accTd(acc, acc >= 0 && Math.abs(acc - bestAcc) < 0.05));
            }
            sb.append("</tr>");
        }
        sb.append("</tbody></table></div>");
        return sb.toString();
    }

    private String categoryTable(List<MethodCol> cols, List<ReportMeta> metas) {
        String[] cats = {"A_CLEAN", "B_TRANSFORMED", "C_DEGRADED", "D_NEGATIVE"};
        StringBuilder sb = new StringBuilder();
        sb.append("<div class='scroll-wrap'><table class='scenario-table'><thead>");
        sb.append("<tr><th>Category</th>");
        for (MethodCol c : cols) {
            sb.append("<th class='col-hdr' title='").append(c.fullName()).append("'>")
              .append(c.shortName()).append("</th>");
        }
        sb.append("</tr></thead><tbody>");
        for (String cat : cats) {
            double bestAcc = cols.stream()
                    .mapToDouble(c -> getCatAcc(c, cat, metas))
                    .filter(v -> v >= 0).max().orElse(-1);
            sb.append("<tr><td class='variant-label'>").append(cat).append("</td>");
            for (MethodCol c : cols) {
                double acc = getCatAcc(c, cat, metas);
                sb.append(accTd(acc, acc >= 0 && Math.abs(acc - bestAcc) < 0.05));
            }
            sb.append("</tr>");
        }
        sb.append("</tbody></table></div>");
        return sb.toString();
    }

    // =========================================================================
    // Tab 3 — CF / CF1 full comparison
    // =========================================================================

    private String buildCfTab(List<ReportMeta> metas) {
        if (metas.isEmpty()) return noDataMsg();
        StringBuilder sb = new StringBuilder();
        sb.append("<h2>🎨 CF &amp; CF1 Variant Comparison</h2>");
        sb.append("<p>Full accuracy comparison across <em>all</em> colour-filter tiers for every technique. ");
        sb.append("BASE = no colour filter; CF_LOOSE / CF_TIGHT = colour pre-filter applied before matching; ");
        sb.append("CF1_LOOSE / CF1_TIGHT = colour-first region proposal (Milestone 18/19). ");
        sb.append("Δ columns show the accuracy change vs. the same technique's best BASE variant. ");
        sb.append("🟢 Δ ≥ +2% &nbsp; 🔴 Δ ≤ −2% &nbsp; ⬜ within ±2%.</p>");

        // ── Per-technique CF uplift table ─────────────────────────────────────
        sb.append("<h3>Per-Technique Accuracy by CF Tier</h3>");
        sb.append("<table class='rank-table'><thead><tr>");
        sb.append("<th>Technique</th><th>Best BASE</th><th>BASE Acc%</th>");
        sb.append("<th>CF_LOOSE Acc%</th><th>LOOSE Δ</th>");
        sb.append("<th>CF_TIGHT Acc%</th><th>TIGHT Δ</th>");
        sb.append("<th>CF1_LOOSE Acc%</th><th>CF1_LOOSE Δ</th>");
        sb.append("<th>CF1_TIGHT Acc%</th><th>CF1_TIGHT Δ</th>");
        sb.append("<th>🏆 Best Tier</th>");
        sb.append("</tr></thead><tbody>");

        for (ReportMeta meta : metas) {
            double baseAcc = meta.variants().stream()
                    .filter(v -> v.cfTier().equals("BASE"))
                    .mapToDouble(VariantMeta::accuracyPercent).max().orElse(-1);
            if (baseAcc < 0) continue;
            String bestBase = meta.variants().stream()
                    .filter(v -> v.cfTier().equals("BASE"))
                    .max(Comparator.comparingDouble(VariantMeta::accuracyPercent))
                    .map(VariantMeta::name).orElse("—");

            Map<String, Double> tierBest = new LinkedHashMap<>();
            for (String tier : List.of("CF_LOOSE","CF_TIGHT","CF1_LOOSE","CF1_TIGHT")) {
                tierBest.put(tier, meta.variants().stream()
                        .filter(v -> v.cfTier().equals(tier))
                        .mapToDouble(VariantMeta::accuracyPercent).max().orElse(-1));
            }
            // Find overall best tier
            String bestTier = "BASE";
            double bestTierAcc = baseAcc;
            for (var e : tierBest.entrySet()) {
                if (e.getValue() > bestTierAcc) { bestTierAcc = e.getValue(); bestTier = e.getKey(); }
            }

            sb.append("<tr>");
            sb.append("<td>").append(meta.tr().name()).append("</td>");
            sb.append("<td><code>").append(shorten(bestBase)).append("</code></td>");
            sb.append(scoreTd(baseAcc, 80, 60));
            for (String tier : List.of("CF_LOOSE","CF_TIGHT","CF1_LOOSE","CF1_TIGHT")) {
                double acc = tierBest.get(tier);
                if (acc < 0) { sb.append("<td class='na'>—</td><td class='na'>—</td>"); }
                else { sb.append(scoreTd(acc, 80, 60)).append(deltaTd(acc - baseAcc)); }
            }
            boolean isWinner = !bestTier.equals("BASE");
            sb.append("<td>").append(isWinner ? "🟢 " : "").append(bestTier).append("</td>");
            sb.append("</tr>");
        }
        sb.append("</tbody></table>");

        // ── Full all-variants ranking (base + all CF tiers) ───────────────────
        sb.append("<h3>🏆 Full Ranking — All Variants (Base + CF + CF1)</h3>");
        sb.append("<p>Every individual method variant from every technique, ranked by composite score. ");
        sb.append("This shows the absolute best configuration per technique.</p>");
        List<RankRow> allRows = buildRankRows(metas, null); // null = include all tiers
        allRows.sort(Comparator.comparingDouble(RankRow::composite).reversed());
        // Limit to top 40 to keep the table manageable
        List<RankRow> top = allRows.stream().limit(40).toList();
        sb.append(rankTable(top));
        if (allRows.size() > 40) {
            sb.append("<p class='note'>Showing top 40 of ").append(allRows.size())
              .append(" total variants.</p>");
        }

        // ── Scenario-variant CF analysis: for each group, which CF tier wins? ─
        sb.append("<h3>CF Tier Performance by Scenario Type</h3>");
        sb.append("<p>For each scenario group, the average accuracy of the best variant per tier, ");
        sb.append("across all techniques that have that tier.</p>");
        sb.append(cfByScenarioTable(metas));

        return sb.toString();
    }

    /** Shows average accuracy of each CF tier per scenario group. */
    private String cfByScenarioTable(List<ReportMeta> metas) {
        record Group(String name, List<String> labels) {}
        List<Group> groups = List.of(
                new Group("Clean (A)",      CLEAN_VARIANTS),
                new Group("Scale (B)",      SCALE_VARIANTS),
                new Group("Rotation (B)",   ROT_VARIANTS),
                new Group("Offset (B)",     OFFSET_VARIANTS),
                new Group("Degraded (C)",   DEGRAD_VARIANTS));
        List<String> tiers = List.of("BASE","CF_LOOSE","CF_TIGHT","CF1_LOOSE","CF1_TIGHT");

        StringBuilder sb = new StringBuilder();
        sb.append("<table class='rank-table'><thead><tr><th>Scenario Group</th>");
        for (String t : tiers) sb.append("<th>").append(t).append("</th>");
        sb.append("</tr></thead><tbody>");

        for (Group group : groups) {
            sb.append("<tr><td class='variant-label'>").append(group.name()).append("</td>");
            double bestAvg = -1;
            double[] tierAvgs = new double[tiers.size()];
            for (int ti = 0; ti < tiers.size(); ti++) {
                String tier = tiers.get(ti);
                DoubleStream ds = metas.stream()
                        .flatMap(m -> m.variants().stream())
                        .filter(v -> v.cfTier().equals(tier))
                        .flatMap(v -> group.labels().stream()
                                .map(lbl -> v.byVariant().get(lbl))
                                .filter(Objects::nonNull)
                                .mapToDouble(SceneBucket::acc)
                                .filter(a -> a >= 0)
                                .boxed())
                        .mapToDouble(Double::doubleValue);
                OptionalDouble avg = ds.average();
                tierAvgs[ti] = avg.isPresent() ? avg.getAsDouble() : -1;
                if (tierAvgs[ti] > bestAvg) bestAvg = tierAvgs[ti];
            }
            for (int ti = 0; ti < tiers.size(); ti++) {
                double avg = tierAvgs[ti];
                sb.append(accTd(avg, avg >= 0 && Math.abs(avg - bestAvg) < 0.5));
            }
            sb.append("</tr>");
        }
        sb.append("</tbody></table>");
        return sb.toString();
    }

    // =========================================================================
    // Tab 4 — Index
    // =========================================================================

    private String buildOverviewTab(List<TechniqueReport> available,
                                     List<TechniqueReport> missing, Path reportDir) {
        StringBuilder sb = new StringBuilder();
        sb.append("<h2>Technique Index</h2>");
        sb.append("<table class='index-table'><thead><tr>");
        sb.append("<th>#</th><th>Technique</th><th>Tag</th><th>CF</th><th>CF1</th><th>Status</th><th>Report</th>");
        sb.append("</tr></thead><tbody>");
        int i = 1;
        for (TechniqueReport r : KNOWN_REPORTS) {
            boolean avail = Files.exists(r.path());
            String rel = avail ? reportDir.relativize(r.path().toAbsolutePath().normalize())
                    .toString().replace('\\', '/') : "";
            sb.append("<tr").append(avail ? "" : " class='missing'").append(">");
            sb.append("<td>").append(i++).append("</td><td>").append(r.name()).append("</td>");
            sb.append("<td><code>").append(r.tag()).append("</code></td>");
            sb.append("<td>").append(r.hasCf() ? "✅" : "—").append("</td>");
            sb.append("<td>").append(r.hasCf1() ? "✅" : "—").append("</td>");
            sb.append("<td>").append(avail ? "✅ Available" : "⏳ Pending").append("</td><td>");
            if (avail) {
                sb.append("<a href='").append(rel).append("' target='_blank'>Open ↗</a>");
                sb.append(" <button class='small-btn' onclick=\"switchTab('")
                  .append(r.tag()).append("')\">View →</button>");
            } else { sb.append("<em>pending</em>"); }
            sb.append("</td></tr>");
        }
        sb.append("</tbody></table>");
        if (!missing.isEmpty()) {
            sb.append("<p class='missing-note'>⏳ Pending: ")
              .append(missing.stream().map(TechniqueReport::name).collect(Collectors.joining(", ")))
              .append("</p>");
        }
        return sb.toString();
    }

    // =========================================================================
    // Ranking helpers
    // =========================================================================

    /**
     * Builds RankRow entries for every variant matching the given tier filter.
     * Pass {@code null} to include all tiers.
     */
    private List<RankRow> buildRankRows(List<ReportMeta> metas, String tierFilter) {
        List<RankRow> rows = new ArrayList<>();
        for (ReportMeta meta : metas) {
            // If filtering to BASE only, pick the single best-composite base variant
            if ("BASE".equals(tierFilter)) {
                meta.variants().stream()
                    .filter(v -> v.cfTier().equals("BASE"))
                    .max(Comparator.comparingDouble(v -> composite(v.accuracyPercent(), v.f1Percent(), v.avgMs())))
                    .ifPresent(best -> rows.add(toRankRow(meta.tr(), best)));
            } else if (tierFilter == null) {
                // All variants from all tiers, one row each
                for (VariantMeta v : meta.variants()) {
                    rows.add(toRankRow(meta.tr(), v));
                }
            } else {
                meta.variants().stream()
                    .filter(v -> v.cfTier().equals(tierFilter))
                    .max(Comparator.comparingDouble(v -> composite(v.accuracyPercent(), v.f1Percent(), v.avgMs())))
                    .ifPresent(best -> rows.add(toRankRow(meta.tr(), best)));
            }
        }
        return rows;
    }

    private RankRow toRankRow(TechniqueReport tr, VariantMeta v) {
        long total = v.correct() + v.wrongLoc() + v.missed() + v.falseAlarm() + v.rejected();
        double spd = speedScore(v.avgMs());
        double comp = composite(v.accuracyPercent(), v.f1Percent(), v.avgMs());
        return new RankRow(tr.name(), tr.tag(), v.name(), v.cfTier(),
                v.accuracyPercent(), v.f1Percent(), v.precisionPercent(), v.recallPercent(),
                v.avgMs(), spd, comp, v.correct(), v.missed(), v.falseAlarm(), total);
    }

    private String rankTable(List<RankRow> rows) {
        StringBuilder sb = new StringBuilder();
        sb.append("<table class='rank-table'><thead><tr>");
        sb.append("<th>#</th><th>Technique</th><th>Variant</th><th>Tier</th>");
        sb.append("<th title='(Correct+Rejected)/Total'>Acc%</th>");
        sb.append("<th title='2×P×R/(P+R)'>F1%</th>");
        sb.append("<th>Prec%</th><th>Rec%</th>");
        sb.append("<th>Avg ms</th><th>Speed</th>");
        sb.append("<th title='0.45×Acc+0.30×F1+0.25×Speed'>Composite</th>");
        sb.append("</tr></thead><tbody>");
        int rank = 1;
        for (RankRow r : rows) {
            String medal = rank == 1 ? "🥇" : rank == 2 ? "🥈" : rank == 3 ? "🥉" : "";
            sb.append("<tr>");
            sb.append("<td class='rank-num'>").append(medal).append(rank++).append("</td>");
            sb.append("<td><button class='link-btn' onclick=\"switchTab('").append(r.tag())
              .append("')\">").append(r.name()).append("</button></td>");
            sb.append("<td><code>").append(shorten(r.variant())).append("</code></td>");
            sb.append("<td><span class='tier-badge tier-").append(r.cfTier().replace("_","-").toLowerCase())
              .append("'>").append(r.cfTier()).append("</span></td>");
            sb.append(scoreTd(r.acc(), 80, 60)).append(scoreTd(r.f1(), 70, 50));
            sb.append(scoreTd(r.prec(), 70, 50)).append(scoreTd(r.rec(), 70, 50));
            sb.append(msTd(r.avgMs())).append(scoreTd(r.speedScore(), 80, 40));
            sb.append(scoreTd(r.composite(), 70, 50));
            sb.append("</tr>");
        }
        sb.append("</tbody></table>");
        return sb.toString();
    }

    private String verdictBreakdownTable(List<RankRow> rows) {
        StringBuilder sb = new StringBuilder();
        sb.append("<table class='rank-table'><thead><tr>");
        sb.append("<th>Technique</th><th>Variant</th><th>✅ Correct</th>");
        sb.append("<th>❌ Missed</th><th>⚠️ False Alarm</th><th>Total</th><th>Correct%</th>");
        sb.append("</tr></thead><tbody>");
        for (RankRow r : rows) {
            double pct = r.total() > 0 ? 100.0 * r.correct() / r.total() : 0;
            sb.append("<tr><td>").append(r.name()).append("</td>");
            sb.append("<td><code>").append(shorten(r.variant())).append("</code></td>");
            sb.append("<td class='tp'>").append(r.correct()).append("</td>");
            sb.append("<td class='fn'>").append(r.missed()).append("</td>");
            sb.append("<td class='fp'>").append(r.falseAlarm()).append("</td>");
            sb.append("<td>").append(r.total()).append("</td>");
            sb.append(scoreTd(pct, 70, 50)).append("</tr>");
        }
        sb.append("</tbody></table>");
        return sb.toString();
    }

    private String recommendationBox(List<RankRow> baseRows, List<ReportMeta> metas) {
        if (baseRows.isEmpty()) return "<p>Insufficient data.</p>";
        RankRow top  = baseRows.get(0); // already sorted desc by composite
        RankRow fast = baseRows.stream().filter(r -> r.avgMs() > 0)
                .min(Comparator.comparingDouble(RankRow::avgMs)).orElse(top);
        RankRow acc  = baseRows.stream()
                .max(Comparator.comparingDouble(RankRow::acc)).orElse(top);

        // Best CF uplift across all techniques
        String cfWinner = null; double bestDelta = 0;
        for (ReportMeta meta : metas) {
            double baseAcc = meta.variants().stream()
                    .filter(v -> v.cfTier().equals("BASE"))
                    .mapToDouble(VariantMeta::accuracyPercent).max().orElse(-1);
            double bestCfAcc = meta.variants().stream()
                    .filter(v -> !v.cfTier().equals("BASE"))
                    .mapToDouble(VariantMeta::accuracyPercent).max().orElse(-1);
            if (baseAcc >= 0 && bestCfAcc >= 0 && bestCfAcc - baseAcc > bestDelta) {
                bestDelta = bestCfAcc - baseAcc;
                cfWinner  = meta.tr().name();
            }
        }

        // Best scale, rotation, degradation variant from scenario data
        String bestForScale = bestForGroup(metas, SCALE_VARIANTS);
        String bestForRot   = bestForGroup(metas, ROT_VARIANTS);
        String bestForDegrad= bestForGroup(metas, DEGRAD_VARIANTS);

        StringBuilder sb = new StringBuilder();
        sb.append("<div class='rec-box'>");
        sb.append("<h4>🏆 Best Overall: ").append(top.name()).append("</h4>");
        sb.append("<p>Composite score Acc=").append(f1(top.acc())).append("% F1=")
          .append(f1(top.f1())).append("% at ").append(f1(top.avgMs())).append("ms. ");
        sb.append("Recommended default. Variant: <code>").append(top.variant()).append("</code>.</p>");

        if (!fast.tag().equals(top.tag())) {
            sb.append("<h4>⚡ Fastest: ").append(fast.name()).append("</h4>");
            sb.append("<p>").append(f1(fast.avgMs())).append("ms avg — use for real-time pipelines. ");
            sb.append("Accuracy=").append(f1(fast.acc())).append("%.</p>");
        }
        if (!acc.tag().equals(top.tag())) {
            sb.append("<h4>🎯 Most Accurate: ").append(acc.name()).append("</h4>");
            sb.append("<p>").append(f1(acc.acc())).append("% accuracy — use when false positives are costly.</p>");
        }
        if (cfWinner != null && bestDelta >= 2) {
            sb.append("<h4>🎨 Best CF Uplift: ").append(cfWinner).append("</h4>");
            sb.append("<p>CF variants improve accuracy by +").append(f1(bestDelta))
              .append("% — see the <button class='link-btn' onclick=\"switchTab('cf')\">CF/CF1 tab</button> for details.</p>");
        }
        sb.append("<h4>📌 Guidance by Use Case</h4><ul>");
        sb.append("<li><b>Real-time (≤10ms):</b> ").append(fast.name())
          .append(" — <code>").append(fast.variant()).append("</code></li>");
        sb.append("<li><b>Batch, high accuracy:</b> ").append(acc.name())
          .append(" — <code>").append(acc.variant()).append("</code></li>");
        sb.append("<li><b>Balanced default:</b> ").append(top.name())
          .append(" — <code>").append(top.variant()).append("</code></li>");
        if (bestForScale != null)
            sb.append("<li><b>Scale-invariant matching:</b> ").append(bestForScale).append("</li>");
        if (bestForRot != null)
            sb.append("<li><b>Rotation-invariant matching:</b> ").append(bestForRot).append("</li>");
        if (bestForDegrad != null)
            sb.append("<li><b>Noisy/degraded scenes:</b> ").append(bestForDegrad).append("</li>");
        if (cfWinner != null && bestDelta >= 2)
            sb.append("<li><b>Coloured primitives (known hue):</b> add CF_LOOSE pre-filter to ")
              .append(cfWinner).append(" (+").append(f1(bestDelta)).append("% accuracy)</li>");
        sb.append("</ul></div>");
        return sb.toString();
    }

    /** Returns the technique+variant name that achieves the best average accuracy over a group of scene variants. */
    private String bestForGroup(List<ReportMeta> metas, List<String> group) {
        double bestAvg = -1; String bestLabel = null;
        for (ReportMeta meta : metas) {
            for (VariantMeta v : meta.variants()) {
                OptionalDouble avg = group.stream()
                        .map(lbl -> v.byVariant().get(lbl))
                        .filter(Objects::nonNull)
                        .mapToDouble(SceneBucket::acc)
                        .filter(a -> a >= 0)
                        .average();
                if (avg.isPresent() && avg.getAsDouble() > bestAvg) {
                    bestAvg = avg.getAsDouble();
                    bestLabel = meta.tr().name() + " <code>" + shorten(v.name()) + "</code> ("
                            + f1(bestAvg) + "% avg)";
                }
            }
        }
        return bestLabel;
    }

    // =========================================================================
    // Scenario lookup helpers
    // =========================================================================

    /** Builds all method columns — one per (technique, variantName) pair, sorted by technique order then tier. */
    private List<MethodCol> buildAllMethodCols(List<ReportMeta> metas) {
        List<MethodCol> cols = new ArrayList<>();
        for (ReportMeta meta : metas) {
            // Sort variants: BASE first, then CF_LOOSE, CF_TIGHT, CF1_LOOSE, CF1_TIGHT
            List<VariantMeta> sorted = meta.variants().stream()
                    .sorted(Comparator.comparingInt(v -> tierOrder(v.cfTier())))
                    .toList();
            for (VariantMeta v : sorted) {
                cols.add(new MethodCol(
                        meta.tr().name() + " · " + v.name(),          // fullName
                        meta.tr().tag() + "/" + v.cfTier(),            // shortName
                        meta.tr().tag(),
                        v.name(),
                        v.cfTier()
                ));
            }
        }
        return cols;
    }

    private static int tierOrder(String tier) {
        return switch (tier) {
            case "BASE"      -> 0;
            case "CF_LOOSE"  -> 1;
            case "CF_TIGHT"  -> 2;
            case "CF1_LOOSE" -> 3;
            case "CF1_TIGHT" -> 4;
            default          -> 5;
        };
    }

    private double getAcc(MethodCol col, String variantLabel, List<ReportMeta> metas) {
        return metas.stream()
                .filter(m -> m.tr().tag().equals(col.techTag()))
                .flatMap(m -> m.variants().stream())
                .filter(v -> v.name().equals(col.variantName()))
                .findFirst()
                .map(v -> { SceneBucket b = v.byVariant().get(variantLabel);
                            return b != null ? b.acc() : -1.0; })
                .orElse(-1.0);
    }

    private double getCatAcc(MethodCol col, String cat, List<ReportMeta> metas) {
        return metas.stream()
                .filter(m -> m.tr().tag().equals(col.techTag()))
                .flatMap(m -> m.variants().stream())
                .filter(v -> v.name().equals(col.variantName()))
                .findFirst()
                .map(v -> { SceneBucket b = v.byCategory().get(cat);
                            return b != null ? b.acc() : -1.0; })
                .orElse(-1.0);
    }

    // =========================================================================
    // Composite / speed
    // =========================================================================

    private static double composite(double acc, double f1, double ms) {
        return 0.45 * acc + 0.30 * f1 + 0.25 * speedScore(ms);
    }
    private static double speedScore(double ms) {
        if (ms <= 0) return 0;
        return 100.0 * Math.min(1.0, 10.0 / ms);
    }

    // =========================================================================
    // HTML cell helpers
    // =========================================================================

    private static String scoreTd(double v, double good, double ok) {
        String cls = v >= good ? "g" : v >= ok ? "y" : "r";
        return "<td class='" + cls + "'>" + f1(v) + "%</td>";
    }
    private static String accTd(double acc, boolean bold) {
        if (acc < 0) return "<td class='na'>—</td>";
        String cls = acc >= 80 ? "g" : acc >= 60 ? "y" : "r";
        String val = f1(acc) + "%";
        return "<td class='" + cls + (bold ? " best-cell" : "") + "'>"
                + (bold ? "<b>" + val + "</b>" : val) + "</td>";
    }
    private static String msTd(double ms) {
        if (ms <= 0) return "<td class='na'>—</td>";
        String cls = ms <= 10 ? "g" : ms <= 100 ? "y" : "r";
        return "<td class='" + cls + "'>" + f1(ms) + "</td>";
    }
    private static String deltaTd(double d) {
        String cls = d >= 2 ? "g" : d <= -2 ? "r" : "na";
        String sign = d >= 0 ? "+" : "";
        return "<td class='" + cls + "'>" + sign + f1(d) + "%</td>";
    }
    private static String f1(double v) { return String.format("%.1f", v); }
    private static String shorten(String s) {
        // Remove the technique prefix leaving just the algorithm+tier
        return s.replaceAll("^[A-Z_]+_(?=CF|LOOSE|TIGHT)", "")
                .replaceAll("^(.*?)_(CF.*)", "$1·$2");
    }
    private static String noDataMsg() {
        return "<div class='analysis-empty'><h2>No data</h2><p>Re-run technique tests to embed metadata.</p></div>";
    }

    // =========================================================================
    // CSS
    // =========================================================================

    private static String css() {
        return """
            *{box-sizing:border-box;margin:0;padding:0}
            body{font-family:system-ui,sans-serif;background:#0d1117;color:#c9d1d9}
            .header{background:#161b22;padding:24px 32px;border-bottom:1px solid #30363d}
            .header h1{font-size:1.6rem;color:#58a6ff}
            .subtitle{margin-top:6px;color:#8b949e;font-size:.9rem}
            .tab-bar{display:flex;flex-wrap:wrap;gap:4px;padding:12px 16px;
                     background:#161b22;border-bottom:1px solid #30363d}
            .tab-btn{background:#21262d;color:#c9d1d9;border:1px solid #30363d;
                     padding:6px 14px;border-radius:6px;cursor:pointer;font-size:.82rem}
            .tab-btn:hover{background:#30363d}
            .tab-btn.active{background:#1f6feb;color:#fff;border-color:#1f6feb}
            .tab-content{display:none;padding:24px 32px}
            .tab-content.active{display:block}
            h2{margin:20px 0 10px;color:#58a6ff;font-size:1.2rem}
            h3{margin:18px 0 8px;color:#79c0ff;font-size:1rem}
            h4{margin:10px 0 4px;color:#a5d6ff;font-size:.92rem}
            p{margin-bottom:10px;line-height:1.6;font-size:.88rem}
            ul{margin:6px 0 10px 18px;font-size:.88rem;line-height:1.8}
            .scroll-wrap{overflow-x:auto;margin-bottom:20px}
            .rank-table,.index-table,.scenario-table{width:100%;border-collapse:collapse;font-size:.82rem;margin-bottom:20px}
            .rank-table th,.rank-table td,
            .index-table th,.index-table td,
            .scenario-table th,.scenario-table td{padding:6px 10px;border:1px solid #30363d;text-align:left;white-space:nowrap}
            .rank-table th,.index-table th,.scenario-table th{background:#161b22;color:#8b949e;font-size:.75rem;text-transform:uppercase}
            .scenario-table .col-hdr{writing-mode:vertical-rl;transform:rotate(180deg);max-height:120px;font-size:.7rem;padding:8px 4px}
            .scenario-table .variant-label{font-size:.78rem;color:#8b9eb8;min-width:140px}
            .rank-table tr:hover td,.index-table tr:hover td,.scenario-table tr:hover td{background:#1c2128}
            .index-table tr.missing td{color:#8b949e;font-style:italic}
            td.g{color:#56d364}td.y{color:#d29922}td.r{color:#f85149}td.na{color:#484f58}
            td.best-cell{background:#0d2a14}
            td.tp{color:#56d364}td.fn{color:#f85149}td.fp{color:#d29922}
            .rank-num{font-size:1rem;font-weight:bold;text-align:center}
            code{background:#21262d;padding:1px 5px;border-radius:3px;font-size:.78rem}
            a{color:#58a6ff;text-decoration:none}a:hover{text-decoration:underline}
            .small-btn,.link-btn{background:transparent;color:#58a6ff;border:none;
                cursor:pointer;font-size:.83rem;padding:0;text-decoration:underline}
            .small-btn{background:#21262d;color:#c9d1d9;border:1px solid #30363d;
                padding:2px 8px;border-radius:4px;text-decoration:none}
            .small-btn:hover{background:#1f6feb;color:#fff}
            .iframe-header{padding:8px 0 12px;color:#8b949e;font-size:.9rem}
            .technique-frame{width:100%;height:calc(100vh - 160px);min-height:600px;
                border:1px solid #30363d;border-radius:6px;background:#fff}
            .note{background:#1c2128;border-left:3px solid #d29922;padding:8px 14px;
                border-radius:0 4px 4px 0;color:#c9d1d9;font-size:.85rem;margin:0 0 12px}
            .missing-note{background:#1c2128;border-left:3px solid #d29922;padding:8px 14px;
                border-radius:0 4px 4px 0;color:#c9d1d9;font-size:.85rem;margin-top:16px}
            .rec-box{background:#0d2340;border:1px solid #1f4070;border-radius:6px;padding:16px 20px;margin-top:8px}
            .rec-box h4{margin-top:14px}.rec-box h4:first-child{margin-top:0}
            .analysis-empty{padding:40px;text-align:center}
            .tier-badge{display:inline-block;border-radius:3px;padding:1px 5px;font-size:.72rem;font-weight:bold}
            .tier-badge.tier-base{background:#1a2740;color:#58a6ff}
            .tier-badge.tier-cf-loose{background:#1a3020;color:#56d364}
            .tier-badge.tier-cf-tight{background:#2a2010;color:#d29922}
            .tier-badge.tier-cf1-loose{background:#2a1a30;color:#c78cff}
            .tier-badge.tier-cf1-tight{background:#302010;color:#ff9955}
            """;
    }

    // =========================================================================
    // Data records
    // =========================================================================

    record TechniqueReport(String name, Path path, String tag, boolean hasCf, boolean hasCf1) {}

    record ReportMeta(TechniqueReport tr, String technique, int totalResults,
                      List<VariantMeta> variants, List<String> sceneVariantKeys) {}

    record VariantMeta(String name, String cfTier,
                       double avgScorePercent, double accuracyPercent,
                       double precisionPercent, double recallPercent, double f1Percent,
                       double avgMs,
                       long correct, long wrongLoc, long missed, long falseAlarm, long rejected,
                       Map<String, SceneBucket> byCategory,
                       Map<String, SceneBucket> byVariant) {}

    record SceneBucket(double acc, double f1, double prec, double rec,
                       long correct, long missed, long falseAlarm) {}

    record RankRow(String name, String tag, String variant, String cfTier,
                   double acc, double f1, double prec, double rec,
                   double avgMs, double speedScore, double composite,
                   long correct, long missed, long falseAlarm, long total) {}

    record MethodCol(String fullName, String shortName, String techTag,
                     String variantName, String cfTier) {}
}
