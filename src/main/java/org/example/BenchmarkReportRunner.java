package org.example;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.function.Consumer;
import java.util.regex.*;
import java.util.stream.*;

/**
 * Collates all per-technique {@code report.html} files into a unified
 * benchmark report at {@code test_output/benchmark/report.html}.
 *
 * <p>Fully self-contained in main scope — no test-scope dependencies.
 * Called directly by {@link org.example.ui.BenchmarkLauncher} and also
 * delegated to by the JUnit {@code MatchingBenchmarkTest}.
 */
public final class BenchmarkReportRunner {

    private BenchmarkReportRunner() {}

    private static final Path BENCHMARK_DIR = Paths.get("test_output", "benchmark");

    private static final List<String> SCALE_VARIANTS   = List.of(
            "scale_0.50","scale_0.75","scale_1.25","scale_1.50","scale_2.00",
            "scale1.5_rot45","scale0.75_rot30");
    private static final List<String> ROT_VARIANTS     = List.of(
            "rot_15","rot_30","rot_45","rot_90","rot_180");
    private static final List<String> DEGRAD_VARIANTS  = List.of(
            "noise_s10","noise_s25","contrast_40pct","blur_5x5",
            "occ_25pct","occ_50pct","hue_shift_40");
    private static final List<String> OFFSET_VARIANTS  = List.of(
            "offset_topleft","offset_botright","offset_random42");
    private static final List<String> CLEAN_VARIANTS   = List.of(
            "clean_bg_solid_black","clean_bg_noise_light",
            "clean_bg_gradient_h_colour","clean_bg_random_mixed");

    public static final List<TechniqueReport> KNOWN_REPORTS = List.of(
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
        new TechniqueReport("Vector Matching",              Paths.get("test_output","vector_matching",        "report.html"), "VM",      true,  false),
        new TechniqueReport("Colour-First Region Proposal", Paths.get("test_output","colour_first",           "report.html"), "CF1",     false, true),
        new TechniqueReport("Multi-Colour-First Proposal",  Paths.get("test_output","multi_colour_first",     "report.html"), "MCF1",    false, false)
    );

    // =========================================================================
    //  Public entry point
    // =========================================================================

    /**
     * Collates all available per-technique report.html files and writes the
     * unified benchmark report to {@code test_output/benchmark/report.html}.
     *
     * @param log progress callback (called on the caller's thread)
     * @throws IOException on write failure
     */
    public static void run(Consumer<String> log) throws IOException {
        Path absOut = BENCHMARK_DIR.toAbsolutePath().normalize().resolve("report.html");
        Files.createDirectories(absOut.getParent());

        List<TechniqueReport> available = KNOWN_REPORTS.stream()
                .filter(r -> Files.exists(r.path())).toList();
        List<TechniqueReport> missing   = KNOWN_REPORTS.stream()
                .filter(r -> !Files.exists(r.path())).toList();

        log.accept(String.format("Found %d / %d technique reports.", available.size(), KNOWN_REPORTS.size()));
        if (!missing.isEmpty())
            log.accept("Missing: " + missing.stream().map(TechniqueReport::name).collect(Collectors.joining(", ")));

        List<ReportMeta> metas = new ArrayList<>();
        for (TechniqueReport tr : available) {
            ReportMeta meta = parseReportMeta(tr);
            if (meta != null) {
                metas.add(meta);
                log.accept(String.format("  %-32s  %d variants", tr.name(), meta.variants().size()));
            } else {
                log.accept(String.format("  %-32s  (no metadata — re-run test to embed it)", tr.name()));
            }
        }

        log.accept("Building HTML…");
        String html = buildBenchmarkHtml(available, missing, metas, absOut.getParent());
        Files.write(absOut, html.getBytes(StandardCharsets.UTF_8));
        log.accept("Written: " + absOut);
    }

    // =========================================================================
    //  JSON metadata parser
    // =========================================================================

    private static ReportMeta parseReportMeta(TechniqueReport tr) {
        try {
            String html  = Files.readString(tr.path(), StandardCharsets.UTF_8);
            int start = html.indexOf("\"report-meta\">");
            if (start < 0) return null;
            start += "\"report-meta\">".length();
            int end = html.indexOf("</script>", start);
            if (end < 0) return null;
            return parseJson(tr, html.substring(start, end).trim());
        } catch (IOException e) { return null; }
    }

    private static ReportMeta parseJson(TechniqueReport tr, String json) {
        String technique   = extractStr(json, "technique");
        int    totalResults = (int) extractDbl(json, "totalResults");

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
                    variants.add(new VariantMeta(
                            extractStr(obj, "name"),       extractStr(obj, "cfTier"),
                            extractDbl(obj, "avgScorePercent"), extractDbl(obj, "accuracyPercent"),
                            extractDbl(obj, "precisionPercent"), extractDbl(obj, "recallPercent"),
                            extractDbl(obj, "f1Percent"),   extractDbl(obj, "avgMs"),
                            (long) extractDbl(obj, "correct"),  (long) extractDbl(obj, "wrongLoc"),
                            (long) extractDbl(obj, "missed"),   (long) extractDbl(obj, "falseAlarm"),
                            (long) extractDbl(obj, "rejected"),
                            parseNamedBuckets(obj, "byCategory"),
                            parseNamedBuckets(obj, "byVariant")));
                    objStart = -1;
                }
            }
        }
        return new ReportMeta(tr, technique, totalResults, variants, sceneVariantKeys);
    }

    private static Map<String, SceneBucket> parseNamedBuckets(String obj, String blockName) {
        Map<String, SceneBucket> map = new LinkedHashMap<>();
        int bStart = obj.indexOf("\"" + blockName + "\":{");
        if (bStart < 0) return map;
        bStart += ("\"" + blockName + "\":{").length();
        String block = obj.substring(bStart);
        Pattern keyPat = Pattern.compile("\"([^\"]+)\":\\{([^}]+)\\}");
        Matcher m = keyPat.matcher(block);
        while (m.find()) {
            String key = m.group(1);
            String val = "{" + m.group(2) + "}";
            map.put(key, new SceneBucket(
                    extractDbl(val,"acc"), extractDbl(val,"f1"),
                    extractDbl(val,"prec"), extractDbl(val,"rec"),
                    (long) extractDbl(val,"c"), (long) extractDbl(val,"m"),
                    (long) extractDbl(val,"fa")));
        }
        return map;
    }

    private static String extractStr(String json, String key) {
        Pattern p = Pattern.compile("\"" + key + "\"\\s*:\\s*\"([^\"\\\\]*(\\\\.[^\"\\\\]*)*)\"");
        Matcher m = p.matcher(json);
        return m.find() ? m.group(1).replace("\\\"","\"").replace("\\\\","\\") : "";
    }
    private static double extractDbl(String json, String key) {
        Pattern p = Pattern.compile("\"" + key + "\"\\s*:\\s*(-?[0-9]+\\.?[0-9]*)");
        Matcher m = p.matcher(json);
        return m.find() ? Double.parseDouble(m.group(1)) : 0.0;
    }

    // =========================================================================
    //  HTML builder
    // =========================================================================

    private static String buildBenchmarkHtml(List<TechniqueReport> available,
                                              List<TechniqueReport> missing,
                                              List<ReportMeta> metas,
                                              Path reportDir) {
        String generated = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
        StringBuilder sb = new StringBuilder();
        sb.append("<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>");
        sb.append("<meta name='viewport' content='width=device-width,initial-scale=1'>");
        sb.append("<title>Pattern Matching — Unified Benchmark</title>");
        sb.append("<style>").append(css()).append("</style></head><body>");
        sb.append("<div class='header'><h1>Pattern Matching — Unified Benchmark</h1>");
        sb.append("<p class='subtitle'>Generated ").append(generated)
          .append(" &nbsp;·&nbsp; ").append(available.size()).append(" / ")
          .append(KNOWN_REPORTS.size()).append(" techniques</p></div>");

        sb.append("<div class='tab-bar'>");
        sb.append("<button class='tab-btn active' onclick=\"switchTab('analysis')\">📊 Analysis</button>");
        sb.append("<button class='tab-btn' onclick=\"switchTab('scenarios')\">🔄 Scenarios</button>");
        sb.append("<button class='tab-btn' onclick=\"switchTab('cf')\">🎨 CF / CF1</button>");
        sb.append("<button class='tab-btn' onclick=\"switchTab('overview')\">📋 Index</button>");
        for (TechniqueReport r : available)
            sb.append("<button class='tab-btn' onclick=\"switchTab('").append(r.tag())
              .append("')\">").append(r.name()).append("</button>");
        sb.append("</div>");

        sb.append("<div id='tab-analysis'  class='tab-content active'>").append(buildAnalysisTab(metas, missing)).append("</div>");
        sb.append("<div id='tab-scenarios' class='tab-content'>").append(buildScenarioTab(metas)).append("</div>");
        sb.append("<div id='tab-cf'        class='tab-content'>").append(buildCfTab(metas)).append("</div>");
        sb.append("<div id='tab-overview'  class='tab-content'>").append(buildOverviewTab(reportDir)).append("</div>");

        for (TechniqueReport r : available) {
            String rel = reportDir.relativize(r.path().toAbsolutePath().normalize()).toString().replace('\\','/');
            sb.append("<div id='tab-").append(r.tag()).append("' class='tab-content'>");
            sb.append("<div class='iframe-header'><strong>").append(r.name()).append("</strong>")
              .append(" &nbsp;<a href='").append(rel).append("' target='_blank'>Open standalone ↗</a></div>");
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

    // ── Tab builders (verbatim from MatchingBenchmarkTest) ─────────────────

    private static String buildAnalysisTab(List<ReportMeta> metas, List<TechniqueReport> missing) {
        if (metas.isEmpty()) return noDataMsg();
        StringBuilder sb = new StringBuilder();
        sb.append("<h2>Cross-Technique Analysis</h2>");
        sb.append("<p>Ranked by <b>composite = 45% Accuracy + 30% F1 + 25% Speed</b>. ");
        sb.append("Speed score = 100 × min(1, 10ms/avgMs). Best <em>base</em> variant per technique shown.</p>");
        List<RankRow> rows = buildRankRows(metas, "BASE");
        rows.sort(Comparator.comparingDouble(RankRow::composite).reversed());
        sb.append("<h3>🏆 Overall Ranking — Base Variants</h3>").append(rankTable(rows));
        sb.append("<h3>🔍 Verdict Breakdown</h3>").append(verdictBreakdownTable(rows));
        sb.append("<h3>💡 Recommendation</h3>").append(recommendationBox(rows, metas));
        if (!missing.isEmpty())
            sb.append("<p class='missing-note'>⏳ <strong>").append(missing.size())
              .append(" technique(s) not yet run:</strong> ")
              .append(missing.stream().map(TechniqueReport::name).collect(Collectors.joining(", ")))
              .append("</p>");
        return sb.toString();
    }

    private static String buildScenarioTab(List<ReportMeta> metas) {
        if (metas.isEmpty()) return noDataMsg();
        StringBuilder sb = new StringBuilder();
        sb.append("<h2>Scenario Variant Breakdown</h2>");
        sb.append("<p>Accuracy% per scene variant. 🟢 ≥80% &nbsp; 🟡 ≥60% &nbsp; 🔴 &lt;60% &nbsp; — = no data.</p>");
        List<MethodCol> cols = buildAllMethodCols(metas);
        sb.append(scenarioGroupTable("🧼 Category A — Clean Scenes",      CLEAN_VARIANTS,  cols, metas));
        sb.append(scenarioGroupTable("📐 Category B — Scale Variants",    SCALE_VARIANTS,  cols, metas));
        sb.append(scenarioGroupTable("🔄 Category B — Rotation Variants", ROT_VARIANTS,    cols, metas));
        sb.append(scenarioGroupTable("📍 Category B — Offset Variants",   OFFSET_VARIANTS, cols, metas));
        sb.append(scenarioGroupTable("🌩 Category C — Degraded Scenes",   DEGRAD_VARIANTS, cols, metas));
        sb.append("<h3>Category-Level Summary</h3>").append(categoryTable(cols, metas));
        return sb.toString();
    }

    private static String scenarioGroupTable(String title, List<String> variantLabels,
                                              List<MethodCol> cols, List<ReportMeta> metas) {
        Set<String> available = metas.stream().flatMap(m -> m.variants().stream())
                .flatMap(v -> v.byVariant().keySet().stream()).collect(Collectors.toSet());
        List<String> present = variantLabels.stream().filter(available::contains).toList();
        if (present.isEmpty()) return "";
        StringBuilder sb = new StringBuilder();
        sb.append("<h3>").append(title).append("</h3>");
        sb.append("<div class='scroll-wrap'><table class='scenario-table'><thead><tr><th>Variant</th>");
        for (MethodCol c : cols)
            sb.append("<th class='col-hdr' title='").append(c.fullName()).append("'>").append(c.shortName()).append("</th>");
        sb.append("</tr></thead><tbody>");
        for (String vl : present) {
            double bestAcc = cols.stream().mapToDouble(c -> getAcc(c,vl,metas)).filter(v -> v>=0).max().orElse(-1);
            sb.append("<tr><td class='variant-label'>").append(vl).append("</td>");
            for (MethodCol c : cols) { double acc = getAcc(c,vl,metas); sb.append(accTd(acc, acc>=0 && Math.abs(acc-bestAcc)<0.05)); }
            sb.append("</tr>");
        }
        sb.append("</tbody></table></div>");
        return sb.toString();
    }

    private static String categoryTable(List<MethodCol> cols, List<ReportMeta> metas) {
        String[] cats = {"A_CLEAN","B_TRANSFORMED","C_DEGRADED","D_NEGATIVE"};
        StringBuilder sb = new StringBuilder();
        sb.append("<div class='scroll-wrap'><table class='scenario-table'><thead><tr><th>Category</th>");
        for (MethodCol c : cols)
            sb.append("<th class='col-hdr' title='").append(c.fullName()).append("'>").append(c.shortName()).append("</th>");
        sb.append("</tr></thead><tbody>");
        for (String cat : cats) {
            double bestAcc = cols.stream().mapToDouble(c -> getCatAcc(c,cat,metas)).filter(v -> v>=0).max().orElse(-1);
            sb.append("<tr><td class='variant-label'>").append(cat).append("</td>");
            for (MethodCol c : cols) { double acc = getCatAcc(c,cat,metas); sb.append(accTd(acc, acc>=0 && Math.abs(acc-bestAcc)<0.05)); }
            sb.append("</tr>");
        }
        sb.append("</tbody></table></div>");
        return sb.toString();
    }

    private static String buildCfTab(List<ReportMeta> metas) {
        if (metas.isEmpty()) return noDataMsg();
        StringBuilder sb = new StringBuilder();
        sb.append("<h2>🎨 CF &amp; CF1 Variant Comparison</h2>");
        sb.append("<h3>Per-Technique Accuracy by CF Tier</h3>");
        sb.append("<table class='rank-table'><thead><tr><th>Technique</th><th>Best BASE</th><th>BASE Acc%</th>");
        for (String t : List.of("CF_LOOSE","CF_TIGHT","CF1_LOOSE","CF1_TIGHT"))
            sb.append("<th>").append(t).append(" Acc%</th><th>Δ</th>");
        sb.append("<th>🏆 Best Tier</th></tr></thead><tbody>");
        for (ReportMeta meta : metas) {
            double baseAcc = meta.variants().stream().filter(v -> v.cfTier().equals("BASE"))
                    .mapToDouble(VariantMeta::accuracyPercent).max().orElse(-1);
            if (baseAcc < 0) continue;
            String bestBase = meta.variants().stream().filter(v -> v.cfTier().equals("BASE"))
                    .max(Comparator.comparingDouble(VariantMeta::accuracyPercent))
                    .map(VariantMeta::name).orElse("—");
            Map<String,Double> tierBest = new LinkedHashMap<>();
            for (String tier : List.of("CF_LOOSE","CF_TIGHT","CF1_LOOSE","CF1_TIGHT"))
                tierBest.put(tier, meta.variants().stream().filter(v -> v.cfTier().equals(tier))
                        .mapToDouble(VariantMeta::accuracyPercent).max().orElse(-1));
            String bestTier = "BASE"; double bestTierAcc = baseAcc;
            for (var e : tierBest.entrySet()) if (e.getValue() > bestTierAcc) { bestTierAcc = e.getValue(); bestTier = e.getKey(); }
            sb.append("<tr><td>").append(meta.tr().name()).append("</td><td><code>").append(shorten(bestBase)).append("</code></td>")
              .append(scoreTd(baseAcc,80,60));
            for (String tier : List.of("CF_LOOSE","CF_TIGHT","CF1_LOOSE","CF1_TIGHT")) {
                double acc = tierBest.get(tier);
                if (acc < 0) sb.append("<td class='na'>—</td><td class='na'>—</td>");
                else sb.append(scoreTd(acc,80,60)).append(deltaTd(acc-baseAcc));
            }
            sb.append("<td>").append(!bestTier.equals("BASE") ? "🟢 " : "").append(bestTier).append("</td></tr>");
        }
        sb.append("</tbody></table>");
        sb.append("<h3>🏆 Full Ranking — All Variants</h3>");
        List<RankRow> allRows = buildRankRows(metas, null);
        allRows.sort(Comparator.comparingDouble(RankRow::composite).reversed());
        sb.append(rankTable(allRows.stream().limit(40).toList()));
        return sb.toString();
    }

    private static String buildOverviewTab(Path reportDir) {
        StringBuilder sb = new StringBuilder();
        sb.append("<h2>Technique Index</h2>");
        sb.append("<table class='index-table'><thead><tr><th>#</th><th>Technique</th><th>Tag</th><th>CF</th><th>CF1</th><th>Status</th><th>Report</th></tr></thead><tbody>");
        int i = 1;
        for (TechniqueReport r : KNOWN_REPORTS) {
            boolean avail = Files.exists(r.path());
            String rel = avail ? reportDir.relativize(r.path().toAbsolutePath().normalize()).toString().replace('\\','/') : "";
            sb.append("<tr").append(avail ? "" : " class='missing'").append(">");
            sb.append("<td>").append(i++).append("</td><td>").append(r.name()).append("</td>");
            sb.append("<td><code>").append(r.tag()).append("</code></td>");
            sb.append("<td>").append(r.hasCf() ? "✅":"—").append("</td><td>").append(r.hasCf1() ? "✅":"—").append("</td>");
            sb.append("<td>").append(avail ? "✅ Available":"⏳ Pending").append("</td><td>");
            if (avail) sb.append("<a href='").append(rel).append("' target='_blank'>Open ↗</a>")
                         .append(" <button class='small-btn' onclick=\"switchTab('").append(r.tag()).append("')\">View →</button>");
            else sb.append("<em>pending</em>");
            sb.append("</td></tr>");
        }
        sb.append("</tbody></table>");
        return sb.toString();
    }

    // ── Ranking helpers ─────────────────────────────────────────────────────

    private static List<RankRow> buildRankRows(List<ReportMeta> metas, String tierFilter) {
        List<RankRow> rows = new ArrayList<>();
        for (ReportMeta meta : metas) {
            if ("BASE".equals(tierFilter)) {
                meta.variants().stream().filter(v -> v.cfTier().equals("BASE"))
                    .max(Comparator.comparingDouble(v -> composite(v.accuracyPercent(),v.f1Percent(),v.avgMs())))
                    .ifPresent(best -> rows.add(toRankRow(meta.tr(), best)));
            } else if (tierFilter == null) {
                meta.variants().forEach(v -> rows.add(toRankRow(meta.tr(), v)));
            } else {
                meta.variants().stream().filter(v -> v.cfTier().equals(tierFilter))
                    .max(Comparator.comparingDouble(v -> composite(v.accuracyPercent(),v.f1Percent(),v.avgMs())))
                    .ifPresent(best -> rows.add(toRankRow(meta.tr(), best)));
            }
        }
        return rows;
    }

    private static RankRow toRankRow(TechniqueReport tr, VariantMeta v) {
        double spd  = speedScore(v.avgMs());
        double comp = composite(v.accuracyPercent(), v.f1Percent(), v.avgMs());
        long total  = v.correct() + v.wrongLoc() + v.missed() + v.falseAlarm() + v.rejected();
        return new RankRow(tr.name(), tr.tag(), v.name(), v.cfTier(),
                v.accuracyPercent(), v.f1Percent(), v.precisionPercent(), v.recallPercent(),
                v.avgMs(), spd, comp, v.correct(), v.missed(), v.falseAlarm(), total);
    }

    private static String rankTable(List<RankRow> rows) {
        StringBuilder sb = new StringBuilder();
        sb.append("<table class='rank-table'><thead><tr><th>#</th><th>Technique</th><th>Variant</th><th>Tier</th>");
        sb.append("<th>Acc%</th><th>F1%</th><th>Prec%</th><th>Rec%</th><th>Avg ms</th><th>Speed</th><th>Composite</th></tr></thead><tbody>");
        int rank = 1;
        for (RankRow r : rows) {
            String medal = rank==1?"🥇":rank==2?"🥈":rank==3?"🥉":"";
            sb.append("<tr><td class='rank-num'>").append(medal).append(rank++).append("</td>");
            sb.append("<td><button class='link-btn' onclick=\"switchTab('").append(r.tag()).append("')\">").append(r.name()).append("</button></td>");
            sb.append("<td><code>").append(shorten(r.variant())).append("</code></td>");
            sb.append("<td><span class='tier-badge tier-").append(r.cfTier().replace("_","-").toLowerCase()).append("'>").append(r.cfTier()).append("</span></td>");
            sb.append(scoreTd(r.acc(),80,60)).append(scoreTd(r.f1(),70,50))
              .append(scoreTd(r.prec(),70,50)).append(scoreTd(r.rec(),70,50))
              .append(msTd(r.avgMs())).append(scoreTd(r.speedScore(),80,40))
              .append(scoreTd(r.composite(),70,50)).append("</tr>");
        }
        sb.append("</tbody></table>");
        return sb.toString();
    }

    private static String verdictBreakdownTable(List<RankRow> rows) {
        StringBuilder sb = new StringBuilder();
        sb.append("<table class='rank-table'><thead><tr><th>Technique</th><th>Variant</th><th>✅ Correct</th><th>❌ Missed</th><th>⚠️ False Alarm</th><th>Total</th><th>Correct%</th></tr></thead><tbody>");
        for (RankRow r : rows) {
            double pct = r.total()>0 ? 100.0*r.correct()/r.total() : 0;
            sb.append("<tr><td>").append(r.name()).append("</td><td><code>").append(shorten(r.variant())).append("</code></td>");
            sb.append("<td class='tp'>").append(r.correct()).append("</td><td class='fn'>").append(r.missed())
              .append("</td><td class='fp'>").append(r.falseAlarm()).append("</td><td>").append(r.total()).append("</td>");
            sb.append(scoreTd(pct,70,50)).append("</tr>");
        }
        sb.append("</tbody></table>");
        return sb.toString();
    }

    private static String recommendationBox(List<RankRow> baseRows, List<ReportMeta> metas) {
        if (baseRows.isEmpty()) return "<p>Insufficient data.</p>";
        RankRow top  = baseRows.get(0);
        RankRow fast = baseRows.stream().filter(r -> r.avgMs()>0).min(Comparator.comparingDouble(RankRow::avgMs)).orElse(top);
        RankRow acc  = baseRows.stream().max(Comparator.comparingDouble(RankRow::acc)).orElse(top);
        String cfWinner = null; double bestDelta = 0;
        for (ReportMeta meta : metas) {
            double baseAcc  = meta.variants().stream().filter(v -> v.cfTier().equals("BASE")).mapToDouble(VariantMeta::accuracyPercent).max().orElse(-1);
            double bestCfAcc= meta.variants().stream().filter(v -> !v.cfTier().equals("BASE")).mapToDouble(VariantMeta::accuracyPercent).max().orElse(-1);
            if (baseAcc>=0 && bestCfAcc>=0 && bestCfAcc-baseAcc>bestDelta) { bestDelta=bestCfAcc-baseAcc; cfWinner=meta.tr().name(); }
        }
        StringBuilder sb = new StringBuilder();
        sb.append("<div class='rec-box'>");
        sb.append("<h4>🏆 Best Overall: ").append(top.name()).append("</h4>");
        sb.append("<p>Composite score Acc=").append(f1(top.acc())).append("% F1=").append(f1(top.f1())).append("% at ").append(f1(top.avgMs())).append("ms.</p>");
        if (!fast.tag().equals(top.tag())) sb.append("<h4>⚡ Fastest: ").append(fast.name()).append("</h4><p>").append(f1(fast.avgMs())).append("ms avg.</p>");
        if (!acc.tag().equals(top.tag()))  sb.append("<h4>🎯 Most Accurate: ").append(acc.name()).append("</h4><p>").append(f1(acc.acc())).append("% accuracy.</p>");
        if (cfWinner!=null && bestDelta>=2) sb.append("<h4>🎨 Best CF Uplift: ").append(cfWinner).append("</h4><p>+").append(f1(bestDelta)).append("% accuracy.</p>");
        sb.append("</div>");
        return sb.toString();
    }

    // ── Scenario lookup ──────────────────────────────────────────────────────

    private static List<MethodCol> buildAllMethodCols(List<ReportMeta> metas) {
        List<MethodCol> cols = new ArrayList<>();
        for (ReportMeta meta : metas) {
            meta.variants().stream()
                .sorted(Comparator.comparingInt(v -> tierOrder(v.cfTier())))
                .forEach(v -> cols.add(new MethodCol(
                        meta.tr().name() + " · " + v.name(),
                        meta.tr().tag() + "/" + v.cfTier(),
                        meta.tr().tag(), v.name(), v.cfTier())));
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

    private static double getAcc(MethodCol col, String vl, List<ReportMeta> metas) {
        return metas.stream().filter(m -> m.tr().tag().equals(col.techTag()))
                .flatMap(m -> m.variants().stream()).filter(v -> v.name().equals(col.variantName()))
                .findFirst().map(v -> { SceneBucket b = v.byVariant().get(vl); return b!=null?b.acc():-1.0; }).orElse(-1.0);
    }

    private static double getCatAcc(MethodCol col, String cat, List<ReportMeta> metas) {
        return metas.stream().filter(m -> m.tr().tag().equals(col.techTag()))
                .flatMap(m -> m.variants().stream()).filter(v -> v.name().equals(col.variantName()))
                .findFirst().map(v -> { SceneBucket b = v.byCategory().get(cat); return b!=null?b.acc():-1.0; }).orElse(-1.0);
    }

    // ── Composite / speed ───────────────────────────────────────────────────

    private static double composite(double acc, double f1, double ms) { return 0.45*acc + 0.30*f1 + 0.25*speedScore(ms); }
    private static double speedScore(double ms) { return ms<=0 ? 0 : 100.0*Math.min(1.0, 10.0/ms); }

    // ── Cell helpers ────────────────────────────────────────────────────────

    private static String scoreTd(double v, double good, double ok) {
        return "<td class='" + (v>=good?"g":v>=ok?"y":"r") + "'>" + f1(v) + "%</td>";
    }
    private static String accTd(double acc, boolean bold) {
        if (acc<0) return "<td class='na'>—</td>";
        String cls = acc>=80?"g":acc>=60?"y":"r";
        String val = f1(acc)+"%";
        return "<td class='" + cls + (bold?" best-cell":"") + "'>" + (bold?"<b>"+val+"</b>":val) + "</td>";
    }
    private static String msTd(double ms) {
        if (ms<=0) return "<td class='na'>—</td>";
        return "<td class='" + (ms<=10?"g":ms<=100?"y":"r") + "'>" + f1(ms) + "</td>";
    }
    private static String deltaTd(double d) {
        return "<td class='" + (d>=2?"g":d<=-2?"r":"na") + "'>" + (d>=0?"+":"") + f1(d) + "%</td>";
    }
    private static String f1(double v) { return String.format("%.1f", v); }
    private static String shorten(String s) { return s.replaceAll("^[A-Z_]+_(?=CF|LOOSE|TIGHT)","").replaceAll("^(.*?)_(CF.*)","$1·$2"); }
    private static String noDataMsg() { return "<div class='analysis-empty'><h2>No data</h2><p>Re-run technique tests to embed metadata.</p></div>"; }

    // ── CSS ─────────────────────────────────────────────────────────────────

    private static String css() {
        return """
            *{box-sizing:border-box;margin:0;padding:0}
            body{font-family:system-ui,sans-serif;background:#0d1117;color:#c9d1d9}
            .header{background:#161b22;padding:24px 32px;border-bottom:1px solid #30363d}
            .header h1{font-size:1.6rem;color:#58a6ff}
            .subtitle{margin-top:6px;color:#8b949e;font-size:.9rem}
            .tab-bar{display:flex;flex-wrap:wrap;gap:4px;padding:12px 16px;background:#161b22;border-bottom:1px solid #30363d}
            .tab-btn{background:#21262d;color:#c9d1d9;border:1px solid #30363d;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:.82rem}
            .tab-btn:hover{background:#30363d}.tab-btn.active{background:#1f6feb;color:#fff;border-color:#1f6feb}
            .tab-content{display:none;padding:24px 32px}.tab-content.active{display:block}
            h2{margin:20px 0 10px;color:#58a6ff;font-size:1.2rem}h3{margin:18px 0 8px;color:#79c0ff;font-size:1rem}
            p{margin-bottom:10px;line-height:1.6;font-size:.88rem}
            .scroll-wrap{overflow-x:auto;margin-bottom:20px}
            .rank-table,.index-table,.scenario-table{width:100%;border-collapse:collapse;font-size:.82rem;margin-bottom:20px}
            .rank-table th,.rank-table td,.index-table th,.index-table td,.scenario-table th,.scenario-table td{padding:6px 10px;border:1px solid #30363d;text-align:left;white-space:nowrap}
            .rank-table th,.index-table th,.scenario-table th{background:#161b22;color:#8b949e;font-size:.75rem;text-transform:uppercase}
            .scenario-table .col-hdr{writing-mode:vertical-rl;transform:rotate(180deg);max-height:120px;font-size:.7rem;padding:8px 4px}
            .scenario-table .variant-label{font-size:.78rem;color:#8b9eb8;min-width:140px}
            .rank-table tr:hover td,.index-table tr:hover td,.scenario-table tr:hover td{background:#1c2128}
            .index-table tr.missing td{color:#8b949e;font-style:italic}
            td.g{color:#56d364}td.y{color:#d29922}td.r{color:#f85149}td.na{color:#484f58}td.best-cell{background:#0d2a14}
            td.tp{color:#56d364}td.fn{color:#f85149}td.fp{color:#d29922}
            .rank-num{font-size:1rem;font-weight:bold;text-align:center}
            code{background:#21262d;padding:1px 5px;border-radius:3px;font-size:.78rem}
            a{color:#58a6ff;text-decoration:none}a:hover{text-decoration:underline}
            .small-btn,.link-btn{background:transparent;color:#58a6ff;border:none;cursor:pointer;font-size:.83rem;padding:0;text-decoration:underline}
            .small-btn{background:#21262d;color:#c9d1d9;border:1px solid #30363d;padding:2px 8px;border-radius:4px;text-decoration:none}
            .small-btn:hover{background:#1f6feb;color:#fff}
            .iframe-header{padding:8px 0 12px;color:#8b949e;font-size:.9rem}
            .technique-frame{width:100%;height:calc(100vh - 160px);min-height:600px;border:1px solid #30363d;border-radius:6px;background:#fff}
            .missing-note,.note{background:#1c2128;border-left:3px solid #d29922;padding:8px 14px;border-radius:0 4px 4px 0;color:#c9d1d9;font-size:.85rem;margin-top:16px}
            .rec-box{background:#0d2340;border:1px solid #1f4070;border-radius:6px;padding:16px 20px;margin-top:8px}
            .analysis-empty{padding:40px;text-align:center}
            .tier-badge{display:inline-block;border-radius:3px;padding:1px 5px;font-size:.72rem;font-weight:bold}
            .tier-badge.tier-base{background:#1a2740;color:#58a6ff}.tier-badge.tier-cf-loose{background:#1a3020;color:#56d364}
            .tier-badge.tier-cf-tight{background:#2a2010;color:#d29922}.tier-badge.tier-cf1-loose{background:#2a1a30;color:#c78cff}
            .tier-badge.tier-cf1-tight{background:#302010;color:#ff9955}
            """;
    }

    // =========================================================================
    //  Data records (public so MatchingBenchmarkTest can reference them)
    // =========================================================================

    public record TechniqueReport(String name, Path path, String tag, boolean hasCf, boolean hasCf1) {}
    public record ReportMeta(TechniqueReport tr, String technique, int totalResults,
                             List<VariantMeta> variants, List<String> sceneVariantKeys) {}
    public record VariantMeta(String name, String cfTier,
                              double avgScorePercent, double accuracyPercent,
                              double precisionPercent, double recallPercent, double f1Percent,
                              double avgMs,
                              long correct, long wrongLoc, long missed, long falseAlarm, long rejected,
                              Map<String, SceneBucket> byCategory,
                              Map<String, SceneBucket> byVariant) {}
    public record SceneBucket(double acc, double f1, double prec, double rec,
                              long correct, long missed, long falseAlarm) {}
    public record RankRow(String name, String tag, String variant, String cfTier,
                          double acc, double f1, double prec, double rec,
                          double avgMs, double speedScore, double composite,
                          long correct, long missed, long falseAlarm, long total) {}
    public record MethodCol(String fullName, String shortName, String techTag,
                            String variantName, String cfTier) {}
}
