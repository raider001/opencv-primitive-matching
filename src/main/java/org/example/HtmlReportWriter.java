package org.example;

import org.example.analytics.AnalysisResult;
import org.example.analytics.DetectionVerdict;
import org.example.analytics.PerformanceProfile;
import org.example.analytics.PerformanceProfiler;
import org.example.factories.ReferenceId;
import org.example.scene.SceneCategory;
import org.example.scene.SceneEntry;
import org.example.matchers.SceneDescriptor;
import org.example.scene.SceneShapePlacement;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Produces a fully self-contained HTML report for one matching technique.
 *
 * <p>The report contains two tabs:
 * <ul>
 *   <li><b>Results</b> â€” summary table per reference (avg score by category),
 *       colour-coded ðŸŸ¢ðŸŸ¡ðŸ”´, with expandable per-reference accordion showing
 *       scene thumbnails embedded as base64 PNGs.</li>
 *   <li><b>Performance</b> â€” timing table, resolution projection table,
 *       memory estimation table, CSS bar chart, and interpretation notes.</li>
 * </ul>
 *
 * <p>No JavaScript libraries required â€” all interactivity is pure CSS/HTML.
 */
public final class HtmlReportWriter {

    private HtmlReportWriter() {}

    /** Carries the per-scene cluster JS sidecar URL and its window-key. */
    private record ClusterRef(String jsUrl, String winKey) {}

    /**
     * Writes the report to {@code reportPath}.
     *
     * @param results       all {@link AnalysisResult} objects for this technique
     * @param profiles      performance profiles — one per method variant
     * @param techniqueName display name shown in the report header
     * @param reportPath    destination file (created/overwritten)
     */
    public static void write(List<AnalysisResult> results,
                             List<PerformanceProfile> profiles,
                             String techniqueName,
                             Path reportPath) throws IOException {
        write(results, profiles, techniqueName, reportPath,
                Collections.emptyMap(), Collections.emptyMap());
    }

    /**
     * Full overload — includes ground-truth verdict data for TP/FP/FN/TN display.
     *
     * @param verdicts     map from result → {@link DetectionVerdict}
     * @param sceneMap     map from result → its source {@link SceneEntry} (for ground-truth bbox)
     */
    public static void write(List<AnalysisResult> results,
                             List<PerformanceProfile> profiles,
                             String techniqueName,
                             Path reportPath,
                             Map<AnalysisResult, DetectionVerdict> verdicts,
                             Map<AnalysisResult, SceneEntry> sceneMap) throws IOException {
        boolean hasCf = results.stream().anyMatch(r -> r.methodName().contains("_CF_"));
        boolean hasVerdicts = !verdicts.isEmpty();
        String html = buildHtml(results, profiles, techniqueName, hasCf, hasVerdicts,
                verdicts, sceneMap, reportPath);
        Files.write(reportPath, html.getBytes(StandardCharsets.UTF_8));
    }

    // =========================================================================
    // Top-level HTML builder
    // =========================================================================

    private static String buildHtml(List<AnalysisResult> results,
                                    List<PerformanceProfile> profiles,
                                    String techniqueName,
                                    boolean withCfTab,
                                    boolean withVerdictsTab,
                                    Map<AnalysisResult, DetectionVerdict> verdicts,
                                    Map<AnalysisResult, SceneEntry> sceneMap,
                                    Path reportPath) {
        Map<AnalysisResult, ClusterRef> clusterRefs =
                writeClusterSidecars(results, sceneMap, reportPath);
        String resultsTab     = buildResultsTab(results, verdicts, clusterRefs, reportPath);
        String performanceTab = buildPerformanceTab(profiles);
        String cfTab          = withCfTab       ? buildCfComparisonTab(results) : "";
        String verdictsTab    = withVerdictsTab ? buildVerdictsTab(results, verdicts, sceneMap) : "";

        String tabActivationCss =
                "#t-results:checked ~ #results-content { display: block; }\n" +
                "#t-perf:checked    ~ #perf-content    { display: block; }\n" +
                (withCfTab       ? "#t-cf:checked      ~ #cf-content      { display: block; }\n" : "") +
                (withVerdictsTab ? "#t-verdicts:checked ~ #verdicts-content { display: block; }\n" : "");

        String cfRadio = withCfTab
                ? "  <input type=\"radio\" name=\"tab\" id=\"t-cf\">\n"
                + "  <label for=\"t-cf\">\uD83C\uDFA8 Base vs CF</label>\n" : "";

        String verdictsRadio = withVerdictsTab
                ? "  <input type=\"radio\" name=\"tab\" id=\"t-verdicts\">\n"
                + "  <label for=\"t-verdicts\">\uD83C\uDFAF Verdicts</label>\n" : "";

        String cfContent = withCfTab
                ? "  <div class=\"tab-content\" id=\"cf-content\">\n" + cfTab + "\n  </div>\n" : "";

        String verdictsContent = withVerdictsTab
                ? "  <div class=\"tab-content\" id=\"verdicts-content\">\n" + verdictsTab + "\n  </div>\n" : "";

        return "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
             + "<meta charset=\"UTF-8\">\n"
             + "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
             + "<title>" + esc(techniqueName) + " \u2013 Pattern Matching Report</title>\n"
             + "<style>\n" + CSS + tabActivationCss + "\n</style>\n"
             + "</head>\n<body>\n"
             + "<h1>Pattern Matching Analysis: " + esc(techniqueName) + "</h1>\n"
             + "<p class=\"subtitle\">Generated " + new java.util.Date() + " \u00B7 "
             + results.size() + " results \u00B7 " + profiles.size() + " variants"
             + " \u00B7 <em>Click any scene image to enlarge</em></p>\n"

             // Lightbox overlay (hidden until an image is clicked)
             + "<div id=\"lb\" class=\"lb\" onclick=\"lbClose()\" role=\"dialog\" "
             + "aria-label=\"Image lightbox\" aria-modal=\"true\">\n"
             + "  <button class=\"lb-close\" onclick=\"lbClose()\" "
             + "aria-label=\"Close\">\u2715</button>\n"
             + "  <img id=\"lb-img\" src=\"\" alt=\"Enlarged scene\">\n"
             + "  <div id=\"lb-cap\" class=\"lb-cap\"></div>\n"
             // Cluster explorer panel — hidden until a 🔬 button is clicked
             + "  <div id=\"lb-clusters\" class=\"lb-clusters\" onclick=\"event.stopPropagation()\">\n"
             + "    <div class=\"lb-cl-header\">\n"
             + "      <span class=\"lb-cl-title\">\uD83D\uDD2C Colour Cluster Explorer</span>\n"
             + "      <span id=\"lb-cl-counter\" class=\"lb-cl-counter\"></span>\n"
             + "    </div>\n"
             + "    <div id=\"lb-cl-status\" class=\"lb-cl-status\"></div>\n"
             + "    <div class=\"lb-cl-body\">\n"
             + "      <div class=\"lb-cl-img-wrap\">\n"
             + "        <img id=\"lb-cl-img\" src=\"\" alt=\"Scene base\">\n"
             + "        <svg id=\"lb-cl-svg\" class=\"lb-cl-svg\" "
             + "xmlns=\"http://www.w3.org/2000/svg\"></svg>\n"
             + "      </div>\n"
             + "    </div>\n"
             + "    <div class=\"lb-cl-controls\">\n"
             + "      <button class=\"lb-cl-btn\" onclick=\"lbClusterStep(-1)\">&#9664; Prev</button>\n"
             + "      <div class=\"lb-cl-info\">\n"
             + "        <span id=\"lb-cl-label\" class=\"lb-cl-label\"></span>\n"
             + "        <span id=\"lb-cl-cnt\" class=\"lb-cl-cnt\"></span>\n"
             + "      </div>\n"
             + "      <button class=\"lb-cl-btn\" onclick=\"lbClusterStep(+1)\">Next &#9654;</button>\n"
             + "    </div>\n"
             + "    <div id=\"lb-cl-cap\" class=\"lb-cap\"></div>\n"
             + "  </div>\n"
             + "</div>\n"

             // Tab buttons + content
             + "<div class=\"tabs\">\n"
             + "  <input type=\"radio\" name=\"tab\" id=\"t-results\" checked>\n"
             + "  <label for=\"t-results\">\uD83D\uDCCA Results</label>\n"
             + "  <input type=\"radio\" name=\"tab\" id=\"t-perf\">\n"
             + "  <label for=\"t-perf\">\u26A1 Performance</label>\n"
             + cfRadio
             + verdictsRadio
             + "  <div class=\"tab-content\" id=\"results-content\">\n"
             + resultsTab + "\n"
             + "  </div>\n"
             + "  <div class=\"tab-content\" id=\"perf-content\">\n"
             + performanceTab + "\n"
             + "  </div>\n"
             + cfContent
             + verdictsContent
             + "</div>\n"
             + "<script>\n"
             + "var _lbClData = null, _lbClIdx = 0;\n"
             + "\n"
             + "function lbOpen(src, caption) {\n"
             + "  var lb = document.getElementById('lb');\n"
             + "  document.getElementById('lb-img').src = src;\n"
             + "  document.getElementById('lb-img').style.display = '';\n"
             + "  document.getElementById('lb-cap').textContent = caption;\n"
             + "  document.getElementById('lb-cap').style.display = '';\n"
             + "  document.getElementById('lb-clusters').style.display = 'none';\n"
             + "  lb.classList.add('lb-visible');\n"
             + "  lb.focus();\n"
             + "}\n"
             + "\n"
             + "function lbOpenClusters(btn) {\n"
             + "  var imgSrc  = btn.getAttribute('data-img');\n"
             + "  var jsUrl   = btn.getAttribute('data-js');\n"
             + "  var winKey  = btn.getAttribute('data-key');\n"
             + "  var caption = btn.getAttribute('data-cap');\n"
             + "  var lb = document.getElementById('lb');\n"
             + "  document.getElementById('lb-img').style.display  = 'none';\n"
             + "  document.getElementById('lb-cap').style.display  = 'none';\n"
             + "  document.getElementById('lb-cl-cap').textContent = caption || '';\n"
             + "  document.getElementById('lb-cl-img').src         = imgSrc  || '';\n"
             + "  document.getElementById('lb-clusters').style.display = 'flex';\n"
             + "  lb.classList.add('lb-visible');\n"
             + "  lb.focus();\n"
             + "  _lbClData = null;\n"
             + "  _lbClIdx  = 0;\n"
             + "  document.getElementById('lb-cl-status').textContent  = 'Loading\u2026';\n"
             + "  document.getElementById('lb-cl-label').textContent   = '';\n"
             + "  document.getElementById('lb-cl-cnt').textContent     = '';\n"
             + "  document.getElementById('lb-cl-counter').textContent = '';\n"
             + "  document.getElementById('lb-cl-svg').innerHTML = '';\n"
             + "  if (window[winKey]) {\n"
             + "    _lbClData = window[winKey]; _lbRenderCluster(0); return;\n"
             + "  }\n"
             + "  var s = document.createElement('script');\n"
             + "  s.src = jsUrl;\n"
             + "  s.onload = function() {\n"
             + "    if (window[winKey]) { _lbClData = window[winKey]; _lbRenderCluster(0); }\n"
             + "    else document.getElementById('lb-cl-status').textContent ="
             + " 'Key not found: ' + winKey;\n"
             + "  };\n"
             + "  s.onerror = function() {\n"
             + "    document.getElementById('lb-cl-status').textContent ="
             + " 'Failed to load: ' + jsUrl;\n"
             + "  };\n"
             + "  document.head.appendChild(s);\n"
             + "}\n"
             + "\n"
             + "function lbClose() {\n"
             + "  document.getElementById('lb').classList.remove('lb-visible');\n"
             + "  document.getElementById('lb-img').src = '';\n"
             + "  document.getElementById('lb-img').style.display = '';\n"
             + "  document.getElementById('lb-cap').style.display = '';\n"
             + "  document.getElementById('lb-clusters').style.display = 'none';\n"
             + "  document.getElementById('lb-cl-svg').innerHTML = '';\n"
             + "  _lbClData = null;\n"
             + "}\n"
             + "\n"
             + "function lbClusterStep(dir) {\n"
             + "  if (!_lbClData || !_lbClData.clusters.length) return;\n"
             + "  var n = _lbClData.clusters.length;\n"
             + "  _lbRenderCluster(((_lbClIdx + dir) % n + n) % n);\n"
             + "}\n"
             + "\n"
             + "function _lbRenderCluster(idx) {\n"
             + "  if (!_lbClData) return;\n"
             + "  var cls = _lbClData.clusters;\n"
             + "  if (!cls || !cls.length) {\n"
             + "    document.getElementById('lb-cl-status').textContent = 'No clusters found.';\n"
             + "    return;\n"
             + "  }\n"
             + "  idx = ((idx % cls.length) + cls.length) % cls.length;\n"
             + "  _lbClIdx = idx;\n"
             + "  var c = cls[idx];\n"
             + "  document.getElementById('lb-cl-status').textContent  = '';\n"
             + "  document.getElementById('lb-cl-counter').textContent = (idx + 1) + ' / ' + cls.length;\n"
             + "  document.getElementById('lb-cl-label').textContent   = c.label;\n"
             + "  document.getElementById('lb-cl-cnt').textContent     = c.contours.length + ' contour(s)';\n"
             + "  var svg = document.getElementById('lb-cl-svg');\n"
             + "  svg.setAttribute('viewBox', '0 0 ' + _lbClData.imageW + ' ' + _lbClData.imageH);\n"
             + "  svg.setAttribute('preserveAspectRatio', 'none');\n"
             + "  svg.innerHTML = '';\n"
             + "  c.contours.forEach(function(pts) {\n"
             + "    if (!pts.length) return;\n"
             + "    var el = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');\n"
             + "    el.setAttribute('points', pts.map(function(p){return p[0]+','+p[1];}).join(' '));\n"
             + "    el.setAttribute('fill',         c.colour + '33');\n"
             + "    el.setAttribute('stroke',       c.colour);\n"
             + "    el.setAttribute('stroke-width', '1.5');\n"
             + "    svg.appendChild(el);\n"
             + "  });\n"
             + "}\n"
             + "\n"
             + "document.addEventListener('keydown', function(e) {\n"
             + "  if (e.key === 'Escape') lbClose();\n"
             + "  if (document.getElementById('lb-clusters').style.display !== 'none') {\n"
             + "    if (e.key === 'ArrowLeft')  lbClusterStep(-1);\n"
             + "    if (e.key === 'ArrowRight') lbClusterStep(+1);\n"
             + "  }\n"
             + "});\n"
             + "</script>\n"
             + "</body>\n</html>\n";
    }

    // =========================================================================
    // Results tab
    // =========================================================================

    private static String buildResultsTab(List<AnalysisResult> results,
                                           Map<AnalysisResult, DetectionVerdict> verdicts,
                                           Map<AnalysisResult, ClusterRef> clusterRefs,
                                           Path reportPath) {
        // Group by referenceId
        Map<ReferenceId, List<AnalysisResult>> byRef = results.stream()
                .collect(Collectors.groupingBy(AnalysisResult::referenceId,
                        LinkedHashMap::new, Collectors.toList()));

        // Collect unique method names and categories
        List<String> methods = results.stream()
                .map(AnalysisResult::methodName).distinct().sorted().toList();
        SceneCategory[] cats = SceneCategory.values();

        StringBuilder sb = new StringBuilder();
        boolean hasSummaryVerdicts = !verdicts.isEmpty();

        sb.append("<h2>Results Summary</h2>\n");
        if (hasSummaryVerdicts) {
            sb.append("<p>Avg match score <em>on correctly-located detections only</em> (Score%) "
                    + "and overall reliability (Accuracy%) per reference per category. ");
        } else {
            sb.append("<p>Average raw match score per reference per category. ");
        }
        sb.append("\uD83D\uDFE2 \u226570%  \uD83D\uDFE1 \u226540%  \uD83D\uDD34 &lt;40%  \u26A0\uFE0F gradient background ref</p>\n");

        // ---- Definitions panel ----
        sb.append("<details class=\"defs-panel\">\n");
        sb.append("<summary>&#x1F4D6; Definitions &amp; How to Read This Report</summary>\n");
        sb.append("<div class=\"defs-body\">\n");
        sb.append("<h3>Score %</h3>\n");
        if (hasSummaryVerdicts) {
            sb.append("<p>The match score is normalised to 0\u2013100% for every method so results are comparable "
                    + "across techniques. It represents how confident the matcher is that the reference shape is "
                    + "present at the reported location.</p>"
                    + "<p><b>In the summary table, Score% shows the average score computed only over "
                    + "<em>correctly-located detections</em></b> &mdash; scenes where the matcher both exceeded "
                    + "the score threshold <em>and</em> placed its bounding box on the right region of the image "
                    + "(verdict = \u2705 Correct). "
                    + "This filters out cases where the matcher scored high but pointed at the wrong area, "
                    + "giving a truer picture of per-category confidence when the technique is actually working. "
                    + "A \u2013 means the technique never correctly located the shape in that category.</p>"
                    + "<p>Scores are colour-coded: "
                    + "<span class=\"score-g\">\u2265 70% (good)</span>, "
                    + "<span class=\"score-y\">\u2265 40% (marginal)</span>, "
                    + "<span class=\"score-r\">&lt; 40% (poor)</span>.</p>\n");
            sb.append("<h3>Accuracy %</h3>\n");
            sb.append("<p><b>Accuracy = (Correct + Correctly&thinsp;Rejected) / Total results</b>. "
                    + "This is the primary reliability indicator: it measures how often the technique "
                    + "makes the <em>right call</em> regardless of direction &mdash; correctly finding the shape "
                    + "when it is present, and correctly staying silent when it is absent. "
                    + "A high Score% with a low Accuracy% means the matcher is overconfident: it finds the shape "
                    + "but also fires false alarms on negative scenes. "
                    + "Colour thresholds: "
                    + "<span class=\"score-g\">\u2265 80% (reliable)</span>, "
                    + "<span class=\"score-y\">\u2265 60% (moderate)</span>, "
                    + "<span class=\"score-r\">&lt; 60% (unreliable)</span>.</p>\n");
        } else {
            sb.append("<p>The match score is normalised to 0\u2013100% for every method so results are comparable "
                    + "across techniques. It represents how confident the matcher is that the reference shape is "
                    + "present at the reported location. "
                    + "Scores are colour-coded: "
                    + "<span class=\"score-g\">\u2265 70% (good)</span>, "
                    + "<span class=\"score-y\">\u2265 40% (marginal)</span>, "
                    + "<span class=\"score-r\">&lt; 40% (poor)</span>.</p>\n");
        }
        sb.append("<h3>Categories</h3>\n");
        sb.append("<dl>\n");
        sb.append("  <dt>A &mdash; Clean</dt><dd>Reference placed on a plain or lightly textured background with no transformation. Baseline difficulty.</dd>\n");
        sb.append("  <dt>B &mdash; Transformed</dt><dd>Reference has been rotated, scaled, offset, or a combination of these. Tests geometric robustness.</dd>\n");
        sb.append("  <dt>C &mdash; Degraded</dt><dd>Reference is present but the scene has been degraded: blur, noise, contrast reduction, occlusion, or hue shift. Tests noise robustness.</dd>\n");
        sb.append("  <dt>D &mdash; Negative</dt><dd>The reference shape is <em>not</em> present in the scene. Used to measure false-positive rate.</dd>\n");
        sb.append("</dl>\n");
        sb.append("<h3>Verdict Labels</h3>\n");
        sb.append("<dl>\n");
        sb.append("  <dt><span class=\"vbadge tp\">&#x2705; Correct</span></dt>"
                + "<dd>The queried shape <em>is</em> in this scene, the score exceeded the threshold, "
                + "and the predicted bounding-box centre landed on the shape. True Positive.</dd>\n");
        sb.append("  <dt><span class=\"vbadge fp\">&#x1F4CD; Wrong Location</span></dt>"
                + "<dd>The queried shape is in this scene and the score was high enough, but the "
                + "predicted bbox centre did not overlap the ground-truth region. Detected the wrong area.</dd>\n");
        sb.append("  <dt><span class=\"vbadge fn\">&#x274C; Missed</span></dt>"
                + "<dd>The queried shape is in this scene but the score was below the threshold. False Negative &mdash; the shape was not found.</dd>\n");
        sb.append("  <dt><span class=\"vbadge fp\">&#x26A0;&#xFE0F; False Alarm</span></dt>"
                + "<dd>The queried shape is <em>not</em> in this scene (Cat D, or a different shape's scene) "
                + "but the score exceeded the threshold. False Positive &mdash; the detector hallucinated a match.</dd>\n");
        sb.append("  <dt><span class=\"vbadge tn\">&#x2713; Correctly Rejected</span></dt>"
                + "<dd>The queried shape is not in this scene and the score was below the threshold. "
                + "True Negative &mdash; the detector correctly stayed silent.</dd>\n");
        sb.append("</dl>\n");
        sb.append("<h3>Localisation Check</h3>\n");
        sb.append("<p>Localisation is checked by testing whether the <em>centre</em> of the predicted "
                + "bounding box falls within the ground-truth placed rect expanded by &plusmn;"
                + DetectionVerdict.CENTRE_TOLERANCE_PX + "&thinsp;px on each side. "
                + "This is intentionally scale- and rotation-invariant: even if the matcher returns a "
                + "fixed-size bbox (e.g. always 128&times;128), the centre will land on the shape if the "
                + "match is correct, regardless of the placed shape's actual size in the scene.</p>\n");
        sb.append("<h3>CF Variants</h3>\n");
        sb.append("<p><b>CF_LOOSE / CF_TIGHT</b> &mdash; Colour-Filter pre-processing. Before matching, "
                + "pixels whose colour is far from the expected foreground colour are masked out. "
                + "LOOSE uses a wider colour tolerance; TIGHT uses a narrow tolerance. "
                + "This can dramatically improve scores on busy backgrounds but may hurt performance "
                + "when the placed shape has been hue-shifted.</p>\n");
        sb.append("</div>\n</details>\n\n");

        // Summary table: one row per reference, columns = method × category
        // Each method gets two sub-columns when verdicts are available:
        //   Score%    = avg score over CORRECT detections only
        //   Accuracy% = (CORRECT + CORRECTLY_REJECTED) / total
        sb.append("<table class=\"summary\">\n<thead><tr>");
        sb.append("<th>Reference</th>");
        for (String m : methods) {
            for (SceneCategory c : cats) {
                if (hasSummaryVerdicts) {
                    sb.append("<th colspan=\"2\">").append(esc(shortMethod(m)))
                      .append("<br><small>").append(c.name().charAt(0)).append("</small></th>");
                } else {
                    sb.append("<th>").append(esc(shortMethod(m)))
                      .append("<br><small>").append(c.name().charAt(0)).append("</small></th>");
                }
            }
        }
        sb.append("</tr>\n");
        if (hasSummaryVerdicts) {
            sb.append("<tr><th></th>");
            for (String m : methods) {
                for (SceneCategory c : cats) {
                    sb.append("<th class=\"sub-hdr\">Score%</th><th class=\"sub-hdr\">Acc%</th>");
                }
            }
            sb.append("</tr>\n");
        }
        sb.append("</thead>\n<tbody>\n");

        // Gradient-background reference IDs (slots 2 & 3 of the 4-cycle in ReferenceImageFactory)
        Set<ReferenceId> gradientRefs = gradientBackgroundRefs();

        for (Map.Entry<ReferenceId, List<AnalysisResult>> entry : byRef.entrySet()) {
            ReferenceId ref = entry.getKey();
            List<AnalysisResult> refResults = entry.getValue();
            boolean isGradient = gradientRefs.contains(ref);
            String gradientFlag = isGradient
                ? " title=\"Gradient canvas background \u2013 pixel-sensitive match scores may be lower\"" : "";

            sb.append("<tr>");
            sb.append("<td").append(gradientFlag).append(">")
              .append(isGradient ? "\u26A0\uFE0F " : "").append(esc(ref.name())).append("</td>");

            for (String m : methods) {
                for (SceneCategory c : cats) {
                    final String method = m;
                    final SceneCategory cat = c;
                    List<AnalysisResult> cell = refResults.stream()
                            .filter(r -> r.methodName().equals(method) && r.category() == cat
                                         && !r.isError())
                            .toList();

                    if (hasSummaryVerdicts) {
                        // Score% = avg over CORRECT detections only
                        OptionalDouble correctAvg = cell.stream()
                                .filter(r -> verdicts.get(r) == DetectionVerdict.CORRECT)
                                .mapToDouble(AnalysisResult::matchScorePercent)
                                .average();
                        // Accuracy% = (CORRECT + CORRECTLY_REJECTED) / total with verdict
                        long withV = cell.stream().filter(verdicts::containsKey).count();
                        long good  = cell.stream().filter(r -> {
                            DetectionVerdict v = verdicts.get(r);
                            return v == DetectionVerdict.CORRECT
                                || v == DetectionVerdict.CORRECTLY_REJECTED;
                        }).count();
                        double acc = withV > 0 ? good * 100.0 / withV : Double.NaN;

                        // Score cell
                        if (correctAvg.isPresent()) {
                            double v = correctAvg.getAsDouble();
                            String cls = v >= 70 ? "g" : v >= 40 ? "y" : "r";
                            sb.append("<td class=\"").append(cls).append("\">")
                              .append(String.format("%.0f%%", v)).append("</td>");
                        } else {
                            sb.append("<td class=\"na\">\u2013</td>");
                        }
                        // Accuracy cell
                        if (!Double.isNaN(acc)) {
                            String cls = acc >= 80 ? "g" : acc >= 60 ? "y" : "r";
                            sb.append("<td class=\"").append(cls).append("\">")
                              .append(String.format("%.0f%%", acc)).append("</td>");
                        } else {
                            sb.append("<td class=\"na\">\u2013</td>");
                        }
                    } else {
                        OptionalDouble avg = cell.stream()
                                .mapToDouble(AnalysisResult::matchScorePercent)
                                .average();
                        if (avg.isPresent()) {
                            double v = avg.getAsDouble();
                            String cls = v >= 70 ? "g" : v >= 40 ? "y" : "r";
                            sb.append("<td class=\"").append(cls).append("\">")
                              .append(String.format("%.0f%%", v)).append("</td>");
                        } else {
                            sb.append("<td class=\"na\">\u2013</td>");
                        }
                    }
                }
            }
            sb.append("</tr>\n");
        }
        sb.append("</tbody></table>\n");

        // Accordion: per-reference scene thumbnails
        sb.append("<h2>Per-Reference Detail</h2>\n");
        for (Map.Entry<ReferenceId, List<AnalysisResult>> entry : byRef.entrySet()) {
            ReferenceId ref = entry.getKey();
            List<AnalysisResult> refResults = entry.getValue();
            boolean isGradient = gradientRefs.contains(ref);

            sb.append("<details>\n<summary>")
              .append(isGradient ? "\u26A0\uFE0F " : "").append(esc(ref.name()))
              .append(" (").append(refResults.size()).append(" results)</summary>\n");

            // Reference image — shown once at the top of the expandable section.
            // report.html lives at test_output/<technique>/report.html, so
            // ../references/<ID>.png resolves to test_output/references/<ID>.png
            String refImgPath = "../references/" + ref.name() + ".png";
            sb.append("<div class=\"ref-header\">\n");
            sb.append("  <img src=\"").append(refImgPath).append("\" width=\"96\" height=\"96\"")
              .append(" class=\"ref-thumb lb-trigger\" role=\"button\" tabindex=\"0\"")
              .append(" title=\"Reference: ").append(esc(ref.name())).append("\"")
              .append(" onclick=\"lbOpen(this.src,'Reference: ").append(esc(ref.name())).append("')\"")
              .append(" onkeydown=\"if(event.key==='Enter'||event.key===' ')lbOpen(this.src,'Reference: ")
              .append(esc(ref.name())).append("')\"")
              .append(" onerror=\"this.style.display='none'\">\n");
            sb.append("  <span class=\"ref-name\">").append(esc(ref.name())).append("</span>\n");
            sb.append("</div>\n");

            // Group results by base method (strip _CF_LOOSE / _CF_TIGHT suffixes)
            // so each method family gets its own collapsible sub-section.
            LinkedHashMap<String, List<AnalysisResult>> byMethod = new LinkedHashMap<>();
            for (AnalysisResult r : refResults) {
                String base = r.methodName()
                        .replaceAll("_CF1?_(LOOSE|TIGHT)$", "");
                byMethod.computeIfAbsent(base, k -> new ArrayList<>()).add(r);
            }

            for (Map.Entry<String, List<AnalysisResult>> mEntry : byMethod.entrySet()) {
                String baseMethod = mEntry.getKey();
                List<AnalysisResult> mResults = mEntry.getValue();

                // Build accuracy badge for the method group header
                String avgBadge = accuracyBadge(mResults, verdicts);

                sb.append("<details class=\"method-group\">\n");
                sb.append("<summary class=\"method-group-summary\">")
                  .append(esc(baseMethod)).append(avgBadge)
                  .append(" <span class=\"method-count\">(")
                  .append(mResults.size()).append(" results)</span>")
                  .append("</summary>\n");

                // Third level: group by CF variant (Base / CF Loose / CF Tight)
                LinkedHashMap<String, List<AnalysisResult>> byCf = new LinkedHashMap<>();
                for (AnalysisResult r : mResults) {
                    String cfKey = r.methodName().matches(".*_CF1?_TIGHT$") ? "CF Tight"
                            : r.methodName().matches(".*_CF1?_LOOSE$") ? "CF Loose" : "Base";
                    byCf.computeIfAbsent(cfKey, k -> new ArrayList<>()).add(r);
                }

                for (Map.Entry<String, List<AnalysisResult>> cfEntry : byCf.entrySet()) {
                    String cfLabel = cfEntry.getKey();
                    List<AnalysisResult> cfResults = cfEntry.getValue();
                    String cfAccBadge = accuracyBadge(cfResults, verdicts);

                    sb.append("<details class=\"cf-group\">\n");
                    sb.append("<summary class=\"cf-group-summary\">")
                      .append("<span class=\"cf-badge\">").append(esc(cfLabel)).append("</span>")
                      .append(cfAccBadge)
                      .append(" <span class=\"method-count\">(").append(cfResults.size()).append(" results)</span>")
                      .append("</summary>\n");
                    sb.append("<div class=\"scene-grid\">\n");

                    for (AnalysisResult r : cfResults) {
                        String imgSrc = null;
                        if (r.annotatedPath() != null) {
                            imgSrc = r.annotatedPath().toString().replace('\\', '/');
                        }
                        String scoreLabel = r.isError() ? "ERR" : String.format("%.1f%%", r.matchScorePercent());
                        String cls = r.isError() ? "err"
                                : r.matchScorePercent() >= 70 ? "good"
                                : r.matchScorePercent() >= 40 ? "warn" : "bad";
                        DetectionVerdict verdict = verdicts.get(r);
                        String verdictBadge = verdict != null
                                ? "<span class=\"vbadge " + verdict.cssClass() + "\">"
                                  + verdict.emoji() + " " + verdict.label() + "</span>"
                                : "";
                        sb.append("<div class=\"scene-thumb ").append(cls).append("\">\n");
                        if (imgSrc != null) {
                            String caption = esc(r.referenceId().name()) + " | "
                                    + esc(r.methodName()) + " | "
                                    + esc(r.variantLabel()) + " | "
                                    + "Score: " + scoreLabel + " " + r.matchScoreEmoji()
                                    + (verdict != null ? " | " + verdict.emoji() + " " + verdict.label() : "")
                                    + " | " + r.elapsedMs() + " ms";
                            sb.append("  <img src=\"").append(imgSrc)
                              .append("\" width=\"160\" loading=\"lazy\" class=\"lb-trigger\""
                                    + " role=\"button\" tabindex=\"0\" title=\"Click to enlarge\""
                                    + " onclick=\"lbOpen(this.src,'").append(caption).append("')\"")
                              .append(" onkeydown=\"if(event.key==='Enter'||event.key===' ')"
                                    + "lbOpen(this.src,'").append(caption).append("')\"")
                              .append(">\n");
                        } else {
                            sb.append("  <div class=\"no-img\">no image</div>\n");
                        }
                        sb.append("  <div class=\"scene-label\">")
                          .append(esc(r.variantLabel())).append("<br>")
                          .append("<b>").append(scoreLabel).append("</b>")
                          .append(" ").append(r.matchScoreEmoji())
                          .append(" ").append(r.elapsedMs()).append("ms")
                          .append(verdictBadge.isEmpty() ? "" : "<br>" + verdictBadge)
                          .append("</div>\n");
                        // Cluster explorer button — only when a sidecar was written
                        ClusterRef cr = clusterRefs != null ? clusterRefs.get(r) : null;
                        if (cr != null) {
                            String safeImg = imgSrc != null ? imgSrc : "";
                            String clCap   = (r.referenceId() != null
                                    ? esc(r.referenceId().name()) : "D")
                                    + " | " + esc(r.variantLabel()) + " \u2014 Colour Clusters";
                            sb.append("  <button class=\"cl-btn\""
                                    + " data-img=\"").append(safeImg).append("\""
                                    + " data-js=\"").append(cr.jsUrl()).append("\""
                                    + " data-key=\"").append(cr.winKey()).append("\""
                                    + " data-cap=\"").append(clCap).append("\""
                                    + " onclick=\"lbOpenClusters(this)\""
                                    + " title=\"Explore colour clusters for this scene\">"
                                    + "\uD83D\uDD2C Clusters</button>\n");
                        }
                        sb.append("</div>\n"); // close scene-thumb
                    }
                    sb.append("</div>\n</details>\n"); // cf-group
                }
                sb.append("</details>\n"); // method-group
            }
            sb.append("</details>\n");
        }
        return sb.toString();
    }

    // =========================================================================
    // Performance tab
    // =========================================================================

    private static String buildPerformanceTab(List<PerformanceProfile> profiles) {
        if (profiles.isEmpty()) return "<p>No performance data available.</p>\n";

        StringBuilder sb = new StringBuilder();
        sb.append("<h2>Timing Summary (measured at 640Ã—480)</h2>\n");

        // Timing table
        sb.append("<table class=\"perf\">\n<thead><tr>");
        sb.append("<th>Variant</th><th>Min ms</th><th>Max ms</th><th>Avg ms</th>");
        sb.append("<th>P95 ms</th><th>ms/MP</th></tr></thead>\n<tbody>\n");
        for (PerformanceProfile p : profiles) {
            sb.append("<tr><td>").append(esc(p.methodVariant())).append("</td>")
              .append(td(p.minMs())).append(td(p.maxMs()))
              .append(tdf(p.avgMs())).append(td(p.p95Ms()))
              .append(tdf(p.msPerMp()))
              .append("</tr>\n");
        }
        sb.append("</tbody></table>\n");

        // Resolution projection table
        sb.append("<h2>Resolution Projection</h2>\n");
        sb.append("<p>Linear (L) and quadratic (Q) projection of avg ms to higher resolutions.</p>\n");
        String[] resLabels = {"640x480", "720p", "1080p", "1440p", "4K"};
        sb.append("<table class=\"perf\">\n<thead><tr><th>Variant</th>");
        for (String r : resLabels) {
            sb.append("<th>").append(r).append(" L</th><th>").append(r).append(" Q</th>");
        }
        sb.append("</tr></thead>\n<tbody>\n");
        for (PerformanceProfile p : profiles) {
            sb.append("<tr><td>").append(esc(p.methodVariant())).append("</td>");
            for (String r : resLabels) {
                double[] v = p.projectedMs().get(r);
                if (v != null) {
                    sb.append(tdf(v[0])).append(tdf(v[1]));
                } else {
                    sb.append("<td>\u2013</td><td>\u2013</td>");
                }
            }
            sb.append("</tr>\n");
        }
        sb.append("</tbody></table>\n");

        // Memory estimation table
        sb.append("<h2>Estimated Working Memory</h2>\n");
        sb.append("<table class=\"perf\">\n<thead><tr><th>Variant</th>");
        for (String r : resLabels) sb.append("<th>").append(r).append(" MB</th>");
        sb.append("</tr></thead>\n<tbody>\n");
        for (PerformanceProfile p : profiles) {
            sb.append("<tr><td>").append(esc(p.methodVariant())).append("</td>");
            for (String r : resLabels) {
                Double mb = p.estimatedHeapMb().get(r);
                sb.append(mb != null ? tdf(mb) : "<td>\u2013</td>");
            }
            sb.append("</tr>\n");
        }
        sb.append("</tbody></table>\n");

        // CSS bar chart – avg ms at each resolution (linear projection)
        sb.append("<h2>Projected Avg Time (Linear) \u2013 Bar Chart</h2>\n");
        sb.append("<div class=\"chart\">\n");
        double globalMax = profiles.stream()
                .flatMap(p -> p.projectedMs().values().stream())
                .mapToDouble(v -> v[0]).max().orElse(1);
        for (PerformanceProfile p : profiles) {
            sb.append("<div class=\"chart-row\">\n");
            sb.append("  <div class=\"chart-label\">").append(esc(shortMethod(p.methodVariant())))
              .append("</div>\n");
            for (String r : new String[]{"640x480", "720p", "1080p", "1440p", "4K"}) {
                double[] v = p.projectedMs().get(r);
                double val = v != null ? v[0] : 0;
                int pct = (int) Math.min(100, (val / globalMax) * 100);
                sb.append("  <div class=\"chart-bar-wrap\">")
                  .append("<div class=\"chart-bar\" style=\"width:").append(pct).append("%\">")
                  .append(String.format("%.0f", val)).append("ms")
                  .append("</div></div>\n");
            }
            sb.append("</div>\n");
        }
        sb.append("</div>\n");

        // Interpretation notes
        sb.append("<h2>Interpretation Notes</h2>\n<ul>\n");
        for (PerformanceProfile p : profiles) {
            sb.append("<li>").append(esc(PerformanceProfiler.interpretationNote(p)))
              .append("</li>\n");
        }
        sb.append("</ul>\n");

        return sb.toString();
    }

    // =========================================================================
    // Base vs CF Comparison tab
    // =========================================================================

    /**
     * Builds the Base vs CF Comparison tab.
     *
     * <p>For each base method (e.g. TM_CCOEFF_NORMED), shows the average match score for
     * base, CF_LOOSE, and CF_TIGHT side by side per scene category. Delta cells are
     * colour-coded: ðŸŸ¢ CF improved (delta â‰¥ +2%), â¬œ neutral, ðŸ”´ CF degraded (delta â‰¤ -2%).
     * A summary row shows which mode wins most often across all references.
     */
    private static String buildCfComparisonTab(List<AnalysisResult> results) {
        if (results.isEmpty()) return "<p>No results.</p>";

        // Identify base method names (those without _CF_ suffix)
        List<String> baseMethods = results.stream()
                .map(AnalysisResult::methodName)
                .filter(n -> !n.contains("_CF_"))
                .distinct().sorted().toList();

        if (baseMethods.isEmpty()) return "<p>No base variants found \u2013 CF comparison unavailable.</p>";

        SceneCategory[] cats = SceneCategory.values();
        StringBuilder sb = new StringBuilder();

        sb.append("<h2>Base vs CF Comparison</h2>\n");
        sb.append("<p>Average match score per base method per scene category. ")
          .append("Delta = CF score \u2212 base score. ")
          .append("\uD83D\uDFE2 \u2265+2% improvement \u00B7 \uD83D\uDD34 \u2264\u22122% degradation \u00B7 \u2B1C within \u00B12%</p>\n");

        // One table per base method
        for (String base : baseMethods) {
            String loose = base + "_CF_LOOSE";
            String tight = base + "_CF_TIGHT";

            sb.append("<h3>").append(esc(base)).append("</h3>\n");
            sb.append("<table class=\"perf\">\n<thead><tr>");
            sb.append("<th>Category</th>");
            sb.append("<th>Base avg%</th>");
            sb.append("<th>CF_LOOSE avg%</th><th>LOOSE Î”</th>");
            sb.append("<th>CF_TIGHT avg%</th><th>TIGHT Î”</th>");
            sb.append("</tr></thead>\n<tbody>\n");

            double totalDeltaLoose = 0, totalDeltaTight = 0;
            int rowCount = 0;

            for (SceneCategory cat : cats) {
                OptionalDouble baseAvg  = avgScore(results, base,  cat);
                OptionalDouble looseAvg = avgScore(results, loose, cat);
                OptionalDouble tightAvg = avgScore(results, tight, cat);

                if (baseAvg.isEmpty()) continue;
                double b = baseAvg.getAsDouble();
                double l = looseAvg.isPresent() ? looseAvg.getAsDouble() : Double.NaN;
                double t = tightAvg.isPresent() ? tightAvg.getAsDouble() : Double.NaN;

                double dL = Double.isNaN(l) ? Double.NaN : l - b;
                double dT = Double.isNaN(t) ? Double.NaN : t - b;
                if (!Double.isNaN(dL)) { totalDeltaLoose += dL; rowCount++; }
                if (!Double.isNaN(dT)) { totalDeltaTight += dT; }

                sb.append("<tr>")
                  .append("<td>").append(cat.name()).append("</td>")
                  .append(String.format("<td>%.1f%%</td>", b))
                  .append(Double.isNaN(l) ? "<td>\u2013</td><td>\u2013</td>"
                          : String.format("<td>%.1f%%</td>%s", l, deltaCell(dL)))
                  .append(Double.isNaN(t) ? "<td>\u2013</td><td>\u2013</td>"
                          : String.format("<td>%.1f%%</td>%s", t, deltaCell(dT)))
                  .append("</tr>\n");
            }

            // Summary row
            if (rowCount > 0) {
                double avgDL = totalDeltaLoose / rowCount;
                double avgDT = totalDeltaTight / rowCount;
                sb.append("<tr class=\"summary-row\">")
                  .append("<td><b>Overall avg \u0394</b></td><td>\u2013</td>")
                  .append(String.format("<td>\u2013</td>%s", deltaCell(avgDL)))
                  .append(String.format("<td>\u2013</td>%s", deltaCell(avgDT)))
                  .append("</tr>\n");
            }
            sb.append("</tbody></table>\n");
        }

        // Grand summary – which mode wins most across all methods
        sb.append("<h3>Grand Summary</h3>\n");
        sb.append("<table class=\"perf\">\n<thead><tr>")
          .append("<th>Base Method</th><th>LOOSE avg \u0394</th><th>TIGHT avg \u0394</th><th>Winner</th>")
          .append("</tr></thead>\n<tbody>\n");

        for (String base : baseMethods) {
            String loose = base + "_CF_LOOSE";
            String tight = base + "_CF_TIGHT";
            double sumL = 0, sumT = 0;
            int n = 0;
            for (SceneCategory cat : cats) {
                OptionalDouble bAvg = avgScore(results, base,  cat);
                OptionalDouble lAvg = avgScore(results, loose, cat);
                OptionalDouble tAvg = avgScore(results, tight, cat);
                if (bAvg.isEmpty()) continue;
                if (lAvg.isPresent()) { sumL += lAvg.getAsDouble() - bAvg.getAsDouble(); n++; }
                if (tAvg.isPresent())   sumT += tAvg.getAsDouble() - bAvg.getAsDouble();
            }
            if (n == 0) continue;
            double avgDL = sumL / n, avgDT = sumT / n;
            String winner = avgDL >= avgDT && avgDL > 0.5 ? "\uD83D\uDFE2 CF_LOOSE"
                    : avgDT > avgDL && avgDT > 0.5        ? "\uD83D\uDFE2 CF_TIGHT"
                    : avgDL < -0.5 && avgDT < -0.5        ? "\uD83D\uDD34 Base"
                    :                                        "\u2B1C Neutral";
            sb.append("<tr>")
              .append("<td>").append(esc(base)).append("</td>")
              .append(deltaCell(avgDL)).append(deltaCell(avgDT))
              .append("<td>").append(winner).append("</td>")
              .append("</tr>\n");
        }
        sb.append("</tbody></table>\n");
        return sb.toString();
    }

    private static OptionalDouble avgScore(List<AnalysisResult> results,
                                           String method, SceneCategory cat) {
        return results.stream()
                .filter(r -> r.methodName().equals(method) && r.category() == cat && !r.isError())
                .mapToDouble(AnalysisResult::matchScorePercent)
                .average();
    }

    private static String deltaCell(double delta) {
        if (Double.isNaN(delta)) return "<td>\u2013</td>";
        String cls  = delta >= 2.0 ? "g" : delta <= -2.0 ? "r" : "na";
        String sign = delta >= 0 ? "+" : "";
        return String.format("<td class=\"%s\">%s%.1f%%</td>", cls, sign, delta);
    }

    // =========================================================================
    // Verdicts tab
    // =========================================================================

    /**
     * Builds the Verdicts tab: per-method breakdown counts, precision, recall, F1,
     * plus a full per-scene breakdown table.
     */
    private static String buildVerdictsTab(List<AnalysisResult> results,
                                            Map<AnalysisResult, DetectionVerdict> verdicts,
                                            Map<AnalysisResult, SceneEntry> sceneMap) {
        if (verdicts.isEmpty()) return "<p>No verdict data available.</p>";

        List<String> methods = results.stream()
                .map(AnalysisResult::methodName).distinct().sorted().toList();

        StringBuilder sb = new StringBuilder();

        // --- Legend ---
        sb.append("<h2>Detection Verdict Summary</h2>\n");
        sb.append("<p>Score threshold: <b>").append(DetectionVerdict.DEFAULT_SCORE_THRESHOLD)
          .append("%</b> &nbsp;|&nbsp; ")
          .append("Localisation: predicted bbox centre must fall within ground-truth rect ")
          .append("&plusmn;").append(DetectionVerdict.CENTRE_TOLERANCE_PX).append("px</p>\n");
        sb.append("<p class=\"legend\">"
                + "<span class=\"vbadge tp\">&#x2705; Correct</span> &mdash; found at the right location &nbsp;&nbsp;"
                + "<span class=\"vbadge fp\">&#x1F4CD; Wrong Location</span> &mdash; detected but bbox is off &nbsp;&nbsp;"
                + "<span class=\"vbadge fn\">&#x274C; Missed</span> &mdash; shape present but not found &nbsp;&nbsp;"
                + "<span class=\"vbadge fp\">&#x26A0;&#xFE0F; False Alarm</span> &mdash; no shape but detector fired &nbsp;&nbsp;"
                + "<span class=\"vbadge tn\">&#x2713; Correctly Rejected</span> &mdash; no shape and detector silent"
                + "</p>\n");

        // --- Summary table per method ---
        sb.append("<table class=\"perf\">\n<thead><tr>");
        sb.append("<th>Method</th>");
        sb.append("<th>&#x2705; Correct</th>");
        sb.append("<th>&#x1F4CD; Wrong Location</th>");
        sb.append("<th>&#x274C; Missed</th>");
        sb.append("<th>&#x26A0;&#xFE0F; False Alarm</th>");
        sb.append("<th>&#x2713; Correctly Rejected</th>");
        sb.append("<th>Precision%</th><th>Recall%</th><th>F1%</th>");
        sb.append("</tr></thead>\n<tbody>\n");

        for (String method : methods) {
            long correct  = 0, wrongLoc = 0, missed = 0, falseAlarm = 0, rejected = 0;
            for (AnalysisResult r : results) {
                if (!r.methodName().equals(method)) continue;
                DetectionVerdict v = verdicts.get(r);
                if (v == null) continue;
                switch (v) {
                    case CORRECT             -> correct++;
                    case WRONG_LOCATION      -> wrongLoc++;
                    case MISSED              -> missed++;
                    case FALSE_ALARM         -> falseAlarm++;
                    case CORRECTLY_REJECTED  -> rejected++;
                }
            }
            long tp = correct, fp = wrongLoc + falseAlarm, fn = missed;
            double prec = (tp + fp) > 0 ? 100.0 * tp / (tp + fp) : 0;
            double rec  = (tp + fn) > 0 ? 100.0 * tp / (tp + fn) : 0;
            double f1   = (prec + rec) > 0 ? 2 * prec * rec / (prec + rec) : 0;
            sb.append("<tr>")
              .append("<td>").append(esc(method)).append("</td>")
              .append("<td class=\"tp\">").append(correct).append("</td>")
              .append("<td class=\"fp\">").append(wrongLoc).append("</td>")
              .append("<td class=\"fn\">").append(missed).append("</td>")
              .append("<td class=\"fp\">").append(falseAlarm).append("</td>")
              .append("<td class=\"tn\">").append(rejected).append("</td>")
              .append(String.format("<td>%.1f%%</td><td>%.1f%%</td><td>%.1f%%</td>",
                      prec, rec, f1))
              .append("</tr>\n");
        }
        sb.append("</tbody></table>\n");

        // --- Per-scene breakdown ---
        sb.append("<h2>Per-Scene Breakdown</h2>\n");
        sb.append("<details><summary>Show all ").append(verdicts.size())
          .append(" rows</summary>\n");
        sb.append("<table class=\"perf\">\n<thead><tr>");
        sb.append("<th>Method</th><th>Reference</th><th>Scene Variant</th><th>Category</th>");
        sb.append("<th>Score%</th><th>Verdict</th><th>GT Rect</th><th>Pred Centre</th>");
        sb.append("</tr></thead>\n<tbody>\n");

        for (AnalysisResult r : results) {
            DetectionVerdict v = verdicts.get(r);
            if (v == null) continue;
            SceneEntry scene = sceneMap.get(r);

            String gtRect  = "&mdash;";
            String predCtr = "&mdash;";
            if (scene != null && !scene.placements().isEmpty()) {
                org.opencv.core.Rect gt = scene.placements().stream()
                        .filter(p -> p.referenceId() == r.referenceId())
                        .map(SceneShapePlacement::placedRect)
                        .findFirst().orElse(null);
                if (gt != null) {
                    gtRect = gt.x + "," + gt.y + " " + gt.width + "&times;" + gt.height;
                }
            }
            if (r.boundingRect() != null) {
                org.opencv.core.Rect pred = r.boundingRect();
                int cx = pred.x + pred.width  / 2;
                int cy = pred.y + pred.height / 2;
                predCtr = cx + "," + cy;
            }

            sb.append("<tr>")
              .append("<td>").append(esc(shortMethod(r.methodName()))).append("</td>")
              .append("<td>").append(r.referenceId() != null ? esc(r.referenceId().name()) : "&mdash;").append("</td>")
              .append("<td>").append(esc(r.variantLabel())).append("</td>")
              .append("<td>").append(r.category().name()).append("</td>")
              .append(String.format("<td>%.1f%%</td>", r.matchScorePercent()))
              .append("<td class=\"").append(v.cssClass()).append("\">")
                  .append(v.emoji()).append(" ").append(v.label()).append("</td>")
              .append("<td>").append(gtRect).append("</td>")
              .append("<td>").append(predCtr).append("</td>")
              .append("</tr>\n");
        }
        sb.append("</tbody></table>\n</details>\n");
        return sb.toString();
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    /** Escapes HTML special characters. */
    private static Set<ReferenceId> gradientBackgroundRefs() {
        Set<ReferenceId> set = new HashSet<>();
        ReferenceId[] ids = ReferenceId.values();
        for (int i = 0; i < ids.length; i++) {
            int slot = i % 4;
            if (slot == 2 || slot == 3) set.add(ids[i]); // H colour gradient + radial
        }
        return set;
    }

    private static String accuracyBadge(List<AnalysisResult> results,
                                         Map<AnalysisResult, DetectionVerdict> verdicts) {
        long withVerdicts = results.stream().filter(verdicts::containsKey).count();
        long total        = results.stream().filter(r -> !r.isError()).count();

        if (withVerdicts > 0 && total > 0) {
            long posTotal = results.stream().filter(r -> {
                DetectionVerdict v = verdicts.get(r);
                return v == DetectionVerdict.CORRECT
                    || v == DetectionVerdict.WRONG_LOCATION
                    || v == DetectionVerdict.MISSED;
            }).count();
            long correct  = results.stream()
                    .filter(r -> verdicts.get(r) == DetectionVerdict.CORRECT).count();
            long negTotal = results.stream().filter(r -> {
                DetectionVerdict v = verdicts.get(r);
                return v == DetectionVerdict.CORRECTLY_REJECTED
                    || v == DetectionVerdict.FALSE_ALARM;
            }).count();
            long rejected = results.stream()
                    .filter(r -> verdicts.get(r) == DetectionVerdict.CORRECTLY_REJECTED).count();

            double pctFound    = posTotal > 0 ? correct  * 100.0 / posTotal  : 0;
            double pctRejected = negTotal > 0 ? rejected * 100.0 / negTotal  : 0;
            double combined    = (correct + rejected) * 100.0 / Math.max(1, posTotal + negTotal);
            String accCls      = combined >= 80 ? "g" : combined >= 60 ? "y" : "r";

            String tip = String.format(
                    "Correctly Found: %d of %d positive scenes (%.0f%%) \u2014 "
                  + "shape was present, detected, and bbox was on target  |  "
                  + "Correctly Rejected: %d of %d negative scenes (%.0f%%) \u2014 "
                  + "shape was absent and detector stayed silent",
                    correct, posTotal, pctFound, rejected, negTotal, pctRejected);

            return String.format(
                    " <span class=\"method-avg %s acc-badge\" title=\"%s\">"
                  + "<span class=\"acc-found\">%.0f%%&#x2705;</span>"
                  + " <span class=\"acc-sep\">/</span>"
                  + " <span class=\"acc-rej\">%.0f%%&#x2713;</span>"
                  + "</span>",
                    accCls, esc(tip), pctFound, pctRejected);
        } else {
            OptionalDouble avg = results.stream()
                    .filter(r -> !r.isError())
                    .mapToDouble(AnalysisResult::matchScorePercent)
                    .average();
            return avg.isPresent()
                    ? String.format(
                            " <span class=\"method-avg %s\" title=\"Average raw match score\">avg %.0f%%</span>",
                            avg.getAsDouble() >= 70 ? "g" : avg.getAsDouble() >= 40 ? "y" : "r",
                            avg.getAsDouble())
                    : "";
        }
    }

    private static String shortMethod(String m) {
        if (m == null) return "";
        // Strip common prefixes to keep table columns narrow
        return m.replace("TM_", "").replace("CONTOURS_MATCH_", "").replace("HISTCMP_", "");
    }

    private static String esc(String s) {
        if (s == null) return "";
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace("\"", "&quot;");
    }

    private static String td(long v) {
        return "<td>" + v + "</td>";
    }

    private static String tdf(double v) {
        return String.format("<td>%.1f</td>", v);
    }

    // =========================================================================
    // Cluster sidecar writing
    // =========================================================================

    /**
     * For each unique {@link SceneEntry} in {@code sceneMap}, writes a small JS file to
     * {@code <reportDir>/clusters/<key>.js} containing the raw colour-cluster contour
     * data from its {@link SceneDescriptor}.  The JS file assigns a single property on
     * {@code window} so it can be loaded lazily via a dynamic {@code <script>} tag even
     * when the report is served from a {@code file://} URL.
     *
     * @return map from each {@link AnalysisResult} to its {@link ClusterRef} (jsUrl +
     *         winKey), or an empty map when no descriptors are available.
     */
    private static Map<AnalysisResult, ClusterRef> writeClusterSidecars(
            List<AnalysisResult> results,
            Map<AnalysisResult, SceneEntry> sceneMap,
            Path reportPath) {

        if (sceneMap == null || sceneMap.isEmpty()) return Collections.emptyMap();

        Path clustersDir = reportPath.getParent().resolve("clusters");

        // One JS file per unique SceneEntry instance (identity, not equality).
        IdentityHashMap<SceneEntry, ClusterRef> sceneToRef = new IdentityHashMap<>();

        for (Map.Entry<AnalysisResult, SceneEntry> e : sceneMap.entrySet()) {
            SceneEntry scene = e.getValue();
            if (sceneToRef.containsKey(scene)) continue;
            if (scene == null || scene.descriptor() == null) {
                sceneToRef.put(scene, null);
                continue;
            }

            // Build a safe filename key from scene identity fields.
            String refPart = scene.primaryReferenceId() != null
                    ? scene.primaryReferenceId().name() : "D";
            String bgPart  = scene.backgroundId() != null
                    ? scene.backgroundId().name() : "NONE";
            String varPart = scene.variantLabel() != null
                    ? scene.variantLabel() : "default";
            String safeKey = (refPart + "_" + bgPart + "_" + varPart)
                    .replaceAll("[^A-Za-z0-9_]", "_");

            String jsFilename = safeKey + ".js";
            String winKey     = "__cluster_" + safeKey;

            int W = 640, H = 480;
            if (scene.sceneMat() != null && !scene.sceneMat().empty()) {
                W = scene.sceneMat().cols();
                H = scene.sceneMat().rows();
            }

            try {
                Files.createDirectories(clustersDir);
                String js = buildClusterJs(winKey, scene.descriptor(), W, H);
                Files.writeString(clustersDir.resolve(jsFilename), js,
                        StandardCharsets.UTF_8);
                sceneToRef.put(scene, new ClusterRef("clusters/" + jsFilename, winKey));
            } catch (IOException ex) {
                sceneToRef.put(scene, null);
            }
        }

        // Map every result back to its scene's ClusterRef.
        Map<AnalysisResult, ClusterRef> out = new HashMap<>(results.size() * 2);
        for (AnalysisResult r : results) {
            SceneEntry scene = sceneMap.get(r);
            if (scene != null) {
                ClusterRef ref = sceneToRef.get(scene);
                if (ref != null) out.put(r, ref);
            }
        }
        return out;
    }

    /**
     * Builds the content of a {@code clusters/<key>.js} sidecar file.
     *
     * <p>The file assigns a single window property:
     * <pre>
     *   window["__cluster_<key>"] = { imageW, imageH, clusters: [...] };
     * </pre>
     * Each cluster entry contains the raw contour points exactly as produced by
     * {@link SceneDescriptor} — no filtering or simplification is applied.
     */
    private static String buildClusterJs(String winKey, SceneDescriptor desc,
                                          int imageW, int imageH) {
        StringBuilder sb = new StringBuilder(4096);
        sb.append("window[\"").append(winKey).append("\"] = ");
        sb.append("{\"imageW\":").append(imageW)
          .append(",\"imageH\":").append(imageH)
          .append(",\"clusters\":[");

        List<SceneDescriptor.ClusterContours> clusters = desc.clusters();
        for (int i = 0; i < clusters.size(); i++) {
            if (i > 0) sb.append(",");
            SceneDescriptor.ClusterContours cc = clusters.get(i);

            sb.append("{\"label\":\"").append(clusterLabel(cc).replace("\"", "\\\"")).append("\"")
              .append(",\"hue\":").append(cc.hue)
              .append(",\"achromatic\":").append(cc.achromatic)
              .append(",\"brightAchromatic\":").append(cc.brightAchromatic)
              .append(",\"colour\":\"").append(clusterSvgColour(cc)).append("\"")
              .append(",\"contours\":[");

            List<MatOfPoint> contours = cc.contours;
            for (int j = 0; j < contours.size(); j++) {
                if (j > 0) sb.append(",");
                sb.append("[");
                Point[] pts = contours.get(j).toArray();
                for (int k = 0; k < pts.length; k++) {
                    if (k > 0) sb.append(",");
                    sb.append("[").append((int) pts[k].x)
                      .append(",").append((int) pts[k].y).append("]");
                }
                sb.append("]");
            }
            sb.append("]}");
        }
        sb.append("]};");
        return sb.toString();
    }

    /** Human-readable cluster label for the explorer panel. */
    private static String clusterLabel(SceneDescriptor.ClusterContours cc) {
        if (cc.envelope)        return "Envelope (Combined Chromatic)";
        if (cc.achromatic)      return cc.brightAchromatic
                ? "Bright Achromatic (White / Light-Grey)"
                : "Dark Achromatic (Black / Dark-Grey)";
        return String.format("Chromatic H=%.0f\u00B0 (%s)",
                cc.hue * 2.0, hueNameApprox(cc.hue));
    }

    /** SVG stroke/fill colour derived from the cluster's hue. */
    private static String clusterSvgColour(SceneDescriptor.ClusterContours cc) {
        if (cc.envelope)   return "#ffffff";
        if (cc.achromatic) return cc.brightAchromatic ? "#e8e8e8" : "#888888";
        // OpenCV hue is half-degrees (0-179); CSS hsl() expects full degrees (0-359).
        int cssDeg = ((int) Math.round(cc.hue * 2.0)) % 360;
        return String.format("hsl(%d,100%%,60%%)", cssDeg);
    }

    /** Approximate colour name from an OpenCV half-degree hue value. */
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

    // =========================================================================
    // CSS
    // =========================================================================

    private static final String CSS = """
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: system-ui, sans-serif; background: #0f0f14;
               color: #d0d0d8; padding: 20px; }
        h1 { font-size: 1.6rem; color: #a0c0ff; margin-bottom: 4px; }
        h2 { font-size: 1.1rem; color: #80a0e0; margin: 20px 0 8px; }
        .subtitle { color: #666; font-size: 0.85rem; margin-bottom: 16px; }

        /* Tabs */
        .tabs { display: flex; flex-wrap: wrap; }
        .tabs input[type=radio] { display: none; }
        .tabs label {
          display: inline-block; padding: 8px 18px; background: #1a1a24;
          border: 1px solid #333; cursor: pointer; font-size: 0.9rem;
          border-radius: 4px 4px 0 0; margin-right: 4px; color: #a0a8c0;
        }
        .tabs input:checked + label {
          background: #1e2840; color: #a0c0ff; border-bottom-color: #1e2840;
        }
        .tab-content { display: none; width: 100%; border: 1px solid #333;
                        background: #13131c; padding: 16px; order: 99; }
        /* Tab activation rules injected per-report (see buildHtml) */

        /* Tables */
        table { border-collapse: collapse; font-size: 0.82rem;
                width: 100%; margin-bottom: 12px; overflow-x: auto; display: block; }
        th, td { border: 1px solid #2a2a40; padding: 4px 8px; white-space: nowrap; }
        th { background: #1a1a2e; color: #8090c0; text-align: center; }
        td { text-align: center; }
        td.g  { background: #0d2d14; color: #60d080; }
        td.y  { background: #2a2510; color: #d0b040; }
        td.r  { background: #2d0d0d; color: #d06060; }
        td.na { color: #444; }

        /* Summary table first column */
        table.summary td:first-child { text-align: left; min-width: 160px; }
        /* Sub-header row for Score%/Acc% columns */
        th.sub-hdr { font-size: 0.72rem; color: #6070a0; background: #141428;
                     padding: 2px 4px; font-weight: normal; }

        /* Scene grid */
        .scene-grid { display: flex; flex-wrap: wrap; gap: 8px; padding: 8px 0; }
        .scene-thumb { border: 2px solid #333; border-radius: 4px; width: 164px;
                        overflow: hidden; background: #0d0d14; }
        .scene-thumb.good { border-color: #1a6030; }
        .scene-thumb.warn { border-color: #605010; }
        .scene-thumb.bad  { border-color: #601010; }
        .scene-thumb.err  { border-color: #501060; }
        .scene-thumb img  { display: block; width: 160px; cursor: pointer; }
        .scene-thumb img:hover { opacity: 0.85; outline: 2px solid #4080c0; }
        .lb-trigger:focus { outline: 2px solid #4080c0; }
        .no-img { width: 160px; height: 90px; display: flex; align-items: center;
                  justify-content: center; color: #444; font-size: 0.75rem; }
        .scene-label { font-size: 0.72rem; padding: 4px 6px; color: #9090a8;
                       line-height: 1.4; }
        .scene-label b { color: #d0d0d8; }

        /* Performance table */
        table.perf td:first-child { text-align: left; font-size: 0.78rem; }

        /* Bar chart */
        .chart { font-size: 0.8rem; }
        .chart-row { display: flex; align-items: center; margin-bottom: 4px; gap: 6px; }
        .chart-label { width: 160px; text-align: right; color: #8090b0;
                        font-size: 0.75rem; flex-shrink: 0; }
        .chart-bar-wrap { flex: 1; background: #1a1a2e; border-radius: 2px; height: 20px;
                          min-width: 60px; }
        .chart-bar { height: 20px; background: linear-gradient(90deg, #1a4080, #4070c0);
                      border-radius: 2px; font-size: 0.7rem; color: #c0d0f0;
                      padding: 2px 4px; white-space: nowrap; overflow: hidden; }

        details summary { cursor: pointer; padding: 6px 8px; background: #1a1a2e;
                          border-radius: 4px; margin: 4px 0; color: #a0b0d0; }
        details summary:hover { background: #1e2840; }

        /* Method sub-group inside a reference accordion */
        .method-group { margin: 6px 0 10px 0; border: 1px solid #252535;
                         border-radius: 4px; overflow: hidden; }
        .method-group-summary { cursor: pointer; padding: 5px 10px;
                                 background: #161628; color: #90a8cc;
                                 font-size: 0.85rem; font-weight: bold;
                                 list-style: none; display: flex;
                                 align-items: center; gap: 8px; }
        .method-group-summary:hover { background: #1c1c38; }
        .method-group-summary::before { content: "▶"; font-size: 0.7rem;
                                         transition: transform 0.15s; }
        details.method-group[open] > .method-group-summary::before { transform: rotate(90deg); }

        /* CF variant sub-group inside a method group */
        .cf-group { margin: 4px 6px 6px 6px; border: 1px solid #1e1e30;
                     border-radius: 3px; }
        .cf-group-summary { cursor: pointer; padding: 4px 10px;
                             background: #111120; color: #7090b8;
                             font-size: 0.8rem; list-style: none;
                             display: flex; align-items: center; gap: 6px; }
        .cf-group-summary:hover { background: #161628; }
        .cf-group-summary::before { content: "▶"; font-size: 0.65rem;
                                      transition: transform 0.15s; }
        details.cf-group[open] > .cf-group-summary::before { transform: rotate(90deg); }
        .method-avg { border-radius: 3px; padding: 1px 6px; font-size: 0.75rem;
                       font-weight: bold; }
        .method-avg.g { background: #0a3a18; color: #4ddf80; }
        .method-avg.y { background: #2a2510; color: #d0b040; }
        .method-avg.r { background: #2d0d0d; color: #d06060; }
        .method-count { font-size: 0.72rem; color: #5060a0; font-weight: normal; }
        .cf-badge { display: inline-block; font-size: 0.68rem; border-radius: 3px;
                     padding: 1px 4px; background: #1e243c; color: #7090cc;
                     font-weight: bold; }
        /* Accuracy badge sub-parts */
        .acc-badge { cursor: help; display: inline-flex; align-items: center;
                      gap: 2px; padding: 1px 5px; }
        .acc-found  { color: #4ddf80; }
        .acc-rej    { color: #6699aa; }
        .acc-sep    { color: #445; font-weight: normal; }

        /* CF comparison table summary row */
        tr.summary-row td { background: #1a1a2e; font-style: italic; }

        /* Verdict badges */
        .vbadge { display: inline-block; border-radius: 3px; padding: 1px 5px;
                  font-size: 0.72rem; font-weight: bold; margin-left: 3px; }
        .vbadge.tp, td.tp { background: #0a3a18; color: #4ddf80; }
        .vbadge.fp, td.fp { background: #3a2000; color: #ffaa33; }
        .vbadge.fn, td.fn { background: #3a0a0a; color: #ff5555; }
        .vbadge.tn, td.tn { background: #101820; color: #6699aa; }
        p.legend { font-size: 0.85rem; color: #8090a8; }

        /* Definitions panel */
        .defs-panel { margin-bottom: 16px; }
        .defs-panel > summary { font-weight: bold; font-size: 0.92rem; }
        .defs-body { padding: 12px 16px; background: #10101a; border: 1px solid #2a2a40;
                     border-radius: 0 0 4px 4px; font-size: 0.84rem; color: #9090b0; }
        .defs-body h3 { color: #a0b0d0; margin: 14px 0 4px; font-size: 0.9rem; }
        .defs-body dl { margin: 0 0 8px 12px; }
        .defs-body dt { color: #c0c8e0; font-weight: bold; margin-top: 8px; }
        .defs-body dd { margin: 2px 0 0 16px; }
        .defs-body p  { margin: 4px 0; }
        .score-g { color: #60d080; font-weight: bold; }
        .score-y { color: #d0b040; font-weight: bold; }
        .score-r { color: #d06060; font-weight: bold; }

        /* Per-reference header inside accordion */
        .ref-header { display: flex; align-items: center; gap: 14px;
                      padding: 10px 8px 6px; background: #0e0e18;
                      border-bottom: 1px solid #2a2a40; margin-bottom: 8px; }
        .ref-thumb { display: block; border: 2px solid #2a4060; border-radius: 4px;
                     cursor: pointer; flex-shrink: 0;
                     background: #1a1a2e; /* visible backdrop for black-canvas refs */ }
        .ref-thumb:hover { border-color: #4080c0; opacity: 0.9; }
        .ref-name { font-size: 1rem; font-weight: bold; color: #a0c0e0;
                    letter-spacing: 0.04em; }

        /* Third tab activation (CF tab) \u2013 added inline when withCfTab=true */
        #t-results:checked ~ #results-content { display: block; }
        #t-perf:checked    ~ #perf-content    { display: block; }

        /* Lightbox overlay */
        .lb { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.88);
               z-index: 1000; align-items: center; justify-content: center;
               flex-direction: column; gap: 12px; cursor: zoom-out; }
        .lb.lb-visible { display: flex; }
        .lb img { max-width: 92vw; max-height: 80vh; object-fit: contain;
                   border: 2px solid #334; border-radius: 4px;
                   box-shadow: 0 0 40px rgba(0,0,0,0.8); cursor: default; }
        .lb-cap { color: #c0c8e0; font-size: 0.85rem; text-align: center;
                   max-width: 90vw; padding: 4px 12px; background: rgba(0,0,0,0.5);
                   border-radius: 4px; pointer-events: none; }
        .lb-close { position: fixed; top: 16px; right: 20px; background: #222;
                     border: 1px solid #555; color: #ccc; font-size: 1.4rem;
                     width: 36px; height: 36px; border-radius: 50%; cursor: pointer;
                     display: flex; align-items: center; justify-content: center;
                     z-index: 1001; line-height: 1; }
        .lb-close:hover { background: #444; color: #fff; }

        /* Cluster explorer panel (inside lightbox) */
        #lb-clusters { display: none; flex-direction: column; align-items: center;
                        gap: 10px; max-height: 90vh; cursor: default; width: 100%; }
        .lb-cl-header { display: flex; align-items: center; gap: 16px; width: 100%;
                         justify-content: center; }
        .lb-cl-title  { font-weight: bold; color: #a0c0ff; font-size: 0.9rem; }
        .lb-cl-counter { color: #6080a0; font-size: 0.85rem; background: #1a1a2e;
                          padding: 2px 10px; border-radius: 10px; min-width: 60px;
                          text-align: center; }
        .lb-cl-status { color: #8090b0; font-size: 0.8rem; min-height: 1.2em;
                         text-align: center; }
        .lb-cl-body   { overflow: auto; }
        .lb-cl-img-wrap { position: relative; display: inline-block; }
        .lb-cl-img-wrap img { display: block; max-width: 88vw; max-height: 62vh;
                               background: #050508; }
        .lb-cl-svg  { position: absolute; top: 0; left: 0; width: 100%; height: 100%;
                       pointer-events: none; }
        .lb-cl-controls { display: flex; align-items: center; gap: 20px; }
        .lb-cl-btn  { background: #1e2840; border: 1px solid #4060a0; color: #a0c0ff;
                       padding: 6px 16px; border-radius: 4px; cursor: pointer;
                       font-size: 0.85rem; }
        .lb-cl-btn:hover { background: #253050; }
        .lb-cl-info { text-align: center; min-width: 220px; }
        .lb-cl-label { display: block; color: #d0d8f0; font-weight: bold;
                        font-size: 0.9rem; }
        .lb-cl-cnt  { display: block; color: #6080a0; font-size: 0.78rem; }

        /* Scene-card cluster button */
        .cl-btn { display: block; width: 100%; background: #141828;
                   border: 0; border-top: 1px solid #252540;
                   color: #6080b0; font-size: 0.7rem; padding: 4px 0;
                   cursor: pointer; text-align: center; }
        .cl-btn:hover { background: #1c2238; color: #90b0e0; }
    """;
}

