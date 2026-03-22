package org.example.analytics;

import org.example.factories.BackgroundId;
import org.example.factories.ReferenceId;
import org.example.scene.SceneCategory;
import org.example.scene.SceneEntry;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Stream;

/**
 * Writes and reads JSON sidecar files alongside annotated PNGs so that
 * completed scene-pair results can be reloaded without re-running the matcher.
 *
 * <h2>File layout</h2>
 * <pre>
 *  annotated/
 *    {variant}/
 *      {ref}_vs_{sceneRef}_{variantLabel}.png   ← annotated image (written by matcher)
 *      {ref}_vs_{sceneRef}_{variantLabel}.json  ← sidecar written by this class
 * </pre>
 *
 * <h2>JSON format (hand-rolled, no dependencies)</h2>
 * <pre>
 * {
 *   "methodName":        "TM_CCOEFF_NORMED",
 *   "referenceId":       "LINE_H",
 *   "variantLabel":      "rot_45",
 *   "category":          "B_TRANSFORMED",
 *   "backgroundId":      "BG_SOLID_WHITE",
 *   "matchScorePercent": 87.3,
 *   "boundingRect":      {"x":10,"y":20,"w":64,"h":64},
 *   "elapsedMs":         12,
 *   "preFilterMs":       3,
 *   "scenePx":           307200,
 *   "annotatedPath":     "annotated/TM_CCOEFF_NORMED/line_h_vs_line_h_rot_45.png",
 *   "isError":           false,
 *   "errorMessage":      null
 * }
 * </pre>
 *
 * <p>If both the {@code .png} and {@code .json} exist for a given work item,
 * the result is loaded from JSON and the matcher is skipped for that pair.
 */
public final class ResultMetadataStore {

    private ResultMetadataStore() {}

    // =========================================================================
    //  Write
    // =========================================================================

    /**
     * Writes a JSON sidecar for {@code result} next to its annotated PNG.
     * The sidecar is placed at the same path as {@code annotatedPath} but
     * with a {@code .json} extension.
     *
     * <p>If {@code result.annotatedPath()} is null (image not saved), the JSON
     * is written using the standard naming convention derived from the result
     * fields so it can still be used to skip re-computation.
     *
     * @param result    the result to serialise
     * @param scene     the scene the result was computed against (for ground-truth)
     * @param outputDir the matcher's root output dir (used to resolve paths)
     */
    public static void write(AnalysisResult result, SceneEntry scene, Path outputDir) {
        try {
            Path jsonPath = resolveJsonPath(result, scene, outputDir);
            if (jsonPath == null) return;
            Files.createDirectories(jsonPath.getParent());
            Files.writeString(jsonPath, toJson(result, scene), StandardCharsets.UTF_8);
        } catch (Exception ignored) {}
    }

    // =========================================================================
    //  Read — load all existing sidecars from a variant directory
    // =========================================================================

    /**
     * Loads all {@code .json} sidecar files from {@code annotated/} under
     * {@code outputDir} and reconstructs the corresponding {@link AnalysisResult}
     * and {@link SceneEntry} ground-truth data.
     *
     * <p>Only entries where the paired {@code .png} also exists are returned
     * (a lone JSON without a PNG is ignored).
     *
     * @return map from reconstructed result → partial SceneEntry (ground-truth fields only)
     */
    public static Map<AnalysisResult, SceneEntry> loadAll(Path outputDir) {
        Map<AnalysisResult, SceneEntry> loaded = new LinkedHashMap<>();
        Path annotatedRoot = outputDir.resolve("annotated");
        if (!Files.isDirectory(annotatedRoot)) return loaded;

        try (Stream<Path> varDirs = Files.list(annotatedRoot)) {
            varDirs.filter(Files::isDirectory).forEach(varDir -> {
                try (Stream<Path> files = Files.list(varDir)) {
                    files.filter(p -> p.toString().endsWith(".json")).forEach(jsonPath -> {
                        Path pngPath = jsonPath.resolveSibling(
                                jsonPath.getFileName().toString().replace(".json", ".png"));
                        if (!Files.exists(pngPath)) return; // PNG missing — skip
                        try {
                            String json = Files.readString(jsonPath, StandardCharsets.UTF_8);
                            ParsedEntry entry = fromJson(json, outputDir);
                            if (entry != null) loaded.put(entry.result(), entry.scene());
                        } catch (Exception ignored) {}
                    });
                } catch (IOException ignored) {}
            });
        } catch (IOException ignored) {}
        return loaded;
    }

    /**
     * Returns a set of "skip keys" — one per existing (variant, refId, sceneVariantLabel,
     * backgroundId) tuple that already has both a PNG and a JSON on disk.
     * Used by the parallel loop to avoid recomputing finished pairs.
     */
    public static Set<String> existingKeys(Path outputDir) {
        Set<String> keys = new HashSet<>();
        loadAll(outputDir).keySet().forEach(r -> keys.add(skipKey(r)));
        return keys;
    }

    /**
     * Builds the skip key for a result — uniquely identifies a
     * (methodName, referenceId, sceneVariantLabel, backgroundId) tuple.
     */
    public static String skipKey(AnalysisResult r) {
        return r.methodName() + "|"
             + (r.referenceId() != null ? r.referenceId().name() : "null") + "|"
             + r.variantLabel() + "|"
             + (r.backgroundId() != null ? r.backgroundId().name() : "null");
    }

    /**
     * Builds the skip key from the raw fields — used before a result exists
     * to check whether computation can be skipped.
     */
    public static String skipKey(String methodName, ReferenceId refId,
                                  String variantLabel, BackgroundId bgId) {
        return methodName + "|"
             + (refId != null ? refId.name() : "null") + "|"
             + variantLabel + "|"
             + (bgId  != null ? bgId.name()  : "null");
    }

    // =========================================================================
    //  JSON serialisation (hand-rolled — no external library)
    // =========================================================================

    static String toJson(AnalysisResult r, SceneEntry scene) {
        StringBuilder sb = new StringBuilder("{\n");
        appendStr(sb,  "methodName",        r.methodName());
        appendStr(sb,  "referenceId",       r.referenceId()  != null ? r.referenceId().name()  : null);
        appendStr(sb,  "variantLabel",      r.variantLabel());
        appendStr(sb,  "category",          r.category()     != null ? r.category().name()      : null);
        appendStr(sb,  "backgroundId",      r.backgroundId() != null ? r.backgroundId().name()  : null);
        appendDbl(sb,  "matchScorePercent", r.matchScorePercent());
        if (r.boundingRect() != null) {
            sb.append("  \"boundingRect\": {")
              .append("\"x\":").append(r.boundingRect().x).append(",")
              .append("\"y\":").append(r.boundingRect().y).append(",")
              .append("\"w\":").append(r.boundingRect().width).append(",")
              .append("\"h\":").append(r.boundingRect().height)
              .append("},\n");
        } else {
            sb.append("  \"boundingRect\": null,\n");
        }
        appendLng(sb, "elapsedMs",    r.elapsedMs());
        appendLng(sb, "preFilterMs",  r.preFilterMs());
        appendInt(sb, "scenePx",      r.scenePx());
        appendStr(sb, "annotatedPath",
                r.annotatedPath() != null ? r.annotatedPath().toString().replace('\\', '/') : null);
        appendBool(sb, "isError",    r.isError());
        appendStr(sb,  "errorMessage", r.errorMessage());
        // The scene's actual primary reference (may differ from the query ref for cross-ref runs)
        if (scene != null) {
            appendStr(sb, "sceneRefId",
                    scene.primaryReferenceId() != null ? scene.primaryReferenceId().name() : null);
        } else {
            appendStr(sb, "sceneRefId", null);
        }
        // Ground-truth from SceneEntry (for verdict reconstruction)
        if (scene != null && scene.groundTruthRect() != null) {
            sb.append("  \"gtRect\": {")
              .append("\"x\":").append(scene.groundTruthRect().x).append(",")
              .append("\"y\":").append(scene.groundTruthRect().y).append(",")
              .append("\"w\":").append(scene.groundTruthRect().width).append(",")
              .append("\"h\":").append(scene.groundTruthRect().height)
              .append("}\n");
        } else {
            sb.append("  \"gtRect\": null\n");
        }
        sb.append("}");
        return sb.toString();
    }

    // =========================================================================
    //  JSON deserialisation
    // =========================================================================

    record ParsedEntry(AnalysisResult result, SceneEntry scene) {}

    static ParsedEntry fromJson(String json, Path outputDir) {
        try {
            Map<String, String> m = parseFlat(json);

            String methodName   = m.get("methodName");
            String refIdStr     = m.get("referenceId");
            String variantLabel = m.get("variantLabel");
            String categoryStr  = m.get("category");
            String bgIdStr      = m.get("backgroundId");
            double score        = parseDouble(m.get("matchScorePercent"), 0.0);
            long   elapsedMs    = parseLong(m.get("elapsedMs"), 0L);
            long   preFilterMs  = parseLong(m.get("preFilterMs"), 0L);
            int    scenePx      = parseInt(m.get("scenePx"), 0);
            String pathStr      = m.get("annotatedPath");
            boolean isError     = "true".equalsIgnoreCase(m.get("isError"));
            String errorMsg     = m.get("errorMessage");
            // sceneRefId = the shape the scene actually contains (may differ from refId)
            String sceneRefIdStr = m.get("sceneRefId");

            ReferenceId   refId      = refIdStr     != null ? ReferenceId.valueOf(refIdStr)     : null;
            ReferenceId   sceneRefId = sceneRefIdStr != null && !sceneRefIdStr.equals("null")
                                       ? ReferenceId.valueOf(sceneRefIdStr) : null;
            SceneCategory cat    = categoryStr != null ? SceneCategory.valueOf(categoryStr)   : null;
            BackgroundId  bgId   = bgIdStr     != null ? BackgroundId.valueOf(bgIdStr)        : null;

            org.opencv.core.Rect bbox  = parseRect(m.get("boundingRect"));
            org.opencv.core.Rect gtBox = parseRect(m.get("gtRect"));

            Path annotatedPath = pathStr != null && !pathStr.equals("null")
                    ? outputDir.resolve(pathStr) : null;

            AnalysisResult result = new AnalysisResult(
                    methodName, refId, variantLabel, cat, bgId,
                    score, bbox, elapsedMs, preFilterMs, scenePx,
                    annotatedPath, isError, "null".equals(errorMsg) ? null : errorMsg,
                    AnalysisResult.ScoringLayers.ZERO);

            // Reconstruct stub using sceneRefId (the shape in the scene), NOT refId (the query).
            // For cross-ref runs these differ: e.g. querying CIRCLE against a GRID scene.
            // Using refId here incorrectly makes hasThisRef=true in DetectionVerdict.evaluate.
            SceneEntry scene = gtBox != null
                    ? SceneEntry.stub(sceneRefId, cat, bgId, variantLabel, gtBox)
                    : (sceneRefId != null || cat != null
                        ? SceneEntry.stub(sceneRefId, cat, bgId, variantLabel, null)
                        : null);

            return new ParsedEntry(result, scene);
        } catch (Exception e) {
            return null;
        }
    }

    // =========================================================================
    //  Private helpers
    // =========================================================================

    /** Resolves the .json sidecar path for a result. */
    private static Path resolveJsonPath(AnalysisResult r, SceneEntry scene, Path outputDir) {
        if (r.annotatedPath() != null) {
            // Derive from the PNG path that was already written
            String pngStr = r.annotatedPath().toString();
            String jsonStr = pngStr.endsWith(".png")
                    ? pngStr.substring(0, pngStr.length() - 4) + ".json"
                    : pngStr + ".json";
            // annotatedPath is relative to outputDir
            return outputDir.resolve(jsonStr);
        }
        // No annotated image was saved — derive path from fields
        if (r.methodName() == null || r.referenceId() == null || scene == null) return null;
        String variant  = sanitise(r.methodName());
        String sceneRef = scene.primaryReferenceId() != null
                ? sanitise(scene.primaryReferenceId().name()) : "neg";
        String fname    = sanitise(r.referenceId().name()) + "_vs_" + sceneRef
                + "_" + sanitise(r.variantLabel()) + ".json";
        return outputDir.resolve("annotated").resolve(variant).resolve(fname);
    }

    /** Minimal flat JSON parser — handles one level of key/value pairs. */
    private static Map<String, String> parseFlat(String json) {
        Map<String, String> map = new LinkedHashMap<>();
        // Strip outer braces
        String body = json.trim();
        if (body.startsWith("{")) body = body.substring(1);
        if (body.endsWith("}"))  body = body.substring(0, body.lastIndexOf('}'));

        // Match "key": value pairs — handles strings, numbers, booleans, null,
        // and the single-level object for boundingRect/gtRect.
        java.util.regex.Matcher m = java.util.regex.Pattern
                .compile("\"([^\"]+)\"\\s*:\\s*(\\{[^}]*\\}|\"[^\"]*\"|[^,}\\s]+)")
                .matcher(body);
        while (m.find()) {
            String key = m.group(1);
            String val = m.group(2).trim();
            // Strip surrounding quotes from strings
            if (val.startsWith("\"") && val.endsWith("\""))
                val = val.substring(1, val.length() - 1);
            map.put(key, val);
        }
        return map;
    }

    private static org.opencv.core.Rect parseRect(String raw) {
        if (raw == null || raw.equals("null")) return null;
        try {
            Map<String, String> m = parseFlat(raw);
            int x = parseInt(m.get("x"), 0), y = parseInt(m.get("y"), 0);
            int w = parseInt(m.get("w"), 1), h = parseInt(m.get("h"), 1);
            return new org.opencv.core.Rect(x, y, w, h);
        } catch (Exception e) { return null; }
    }

    private static void appendStr(StringBuilder sb, String k, String v) {
        sb.append("  \"").append(k).append("\": ");
        if (v == null) sb.append("null");
        else           sb.append('"').append(v.replace("\\", "\\\\").replace("\"", "\\\"")).append('"');
        sb.append(",\n");
    }
    private static void appendDbl (StringBuilder sb, String k, double v) { sb.append("  \"").append(k).append("\": ").append(String.format(Locale.ROOT, "%.6f", v)).append(",\n"); }
    private static void appendLng (StringBuilder sb, String k, long   v) { sb.append("  \"").append(k).append("\": ").append(v).append(",\n"); }
    private static void appendInt (StringBuilder sb, String k, int    v) { sb.append("  \"").append(k).append("\": ").append(v).append(",\n"); }
    private static void appendBool(StringBuilder sb, String k, boolean v){ sb.append("  \"").append(k).append("\": ").append(v).append(",\n"); }

    private static double  parseDouble(String s, double  def) { try { return s != null && !s.equals("null") ? Double.parseDouble(s)  : def; } catch(Exception e){ return def; } }
    private static long    parseLong  (String s, long    def) { try { return s != null && !s.equals("null") ? Long.parseLong(s)      : def; } catch(Exception e){ return def; } }
    private static int     parseInt   (String s, int     def) { try { return s != null && !s.equals("null") ? Integer.parseInt(s)    : def; } catch(Exception e){ return def; } }

    private static String sanitise(String s) {
        return s == null ? "unknown" : s.replaceAll("[^A-Za-z0-9_\\-]", "_");
    }
}

