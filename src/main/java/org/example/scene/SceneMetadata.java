package org.example.scene;

import java.util.List;

/**
 * Serialises a {@link SceneEntry} to a self-contained JSON string.
 *
 * <p>No external JSON library — hand-written to keep the dependency footprint minimal.
 * The output is a single JSON object saved as {@code <sampleName>.json} alongside the
 * corresponding PNG, e.g.:
 *
 * <pre>{@code
 * {
 *   "sampleName": "b_rot45_LINE_H",
 *   "category":   "B_TRANSFORMED",
 *   "variant":    "rot_45",
 *   "background": "BG_RANDOM_MIXED",
 *   "sceneW": 640, "sceneH": 480,
 *   "placements": [
 *     {
 *       "referenceId":       "LINE_H",
 *       "placedRect":        { "x":256, "y":176, "w":128, "h":128 },
 *       "scaleFactor":       1.0,
 *       "rotationDeg":       45.0,
 *       "offsetX":           0,
 *       "offsetY":           0,
 *       "colourShifted":     false,
 *       "occluded":          false,
 *       "occlusionFraction": 0.0
 *     }
 *   ]
 * }
 * }</pre>
 */
public final class SceneMetadata {

    private SceneMetadata() {}

    /**
     * Converts a {@link SceneEntry} to a formatted JSON string.
     *
     * @param sampleName the base filename (no extension) used for both the PNG and the JSON
     * @param entry      the scene entry to serialise
     * @return pretty-printed JSON string
     */
    public static String toJson(String sampleName, SceneEntry entry) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append(kv("sampleName", sampleName)).append(",\n");
        sb.append(kv("category",   entry.category().name())).append(",\n");
        sb.append(kv("variant",    entry.variantLabel())).append(",\n");
        sb.append(kv("background", entry.backgroundId().name())).append(",\n");

        // primaryReferenceId may be null for Cat D
        if (entry.primaryReferenceId() != null) {
            sb.append(kv("primaryReferenceId", entry.primaryReferenceId().name())).append(",\n");
        } else {
            sb.append("  \"primaryReferenceId\": null,\n");
        }

        sb.append("  \"sceneW\": ").append(entry.sceneMat().cols()).append(",\n");
        sb.append("  \"sceneH\": ").append(entry.sceneMat().rows()).append(",\n");

        sb.append("  \"placements\": [\n");
        List<SceneShapePlacement> placements = entry.placements();
        for (int i = 0; i < placements.size(); i++) {
            sb.append(placementJson(placements.get(i)));
            if (i < placements.size() - 1) sb.append(",");
            sb.append("\n");
        }
        sb.append("  ]\n");
        sb.append("}\n");
        return sb.toString();
    }

    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    private static String kv(String key, String value) {
        return "  \"" + key + "\": \"" + escape(value) + "\"";
    }

    private static String placementJson(SceneShapePlacement p) {
        return "    {\n" +
               "      \"referenceId\":       \"" + p.referenceId().name() + "\",\n" +
               "      \"placedRect\":        { \"x\":" + p.placedRect().x +
                                             ", \"y\":" + p.placedRect().y +
                                             ", \"w\":" + p.placedRect().width +
                                             ", \"h\":" + p.placedRect().height + " },\n" +
               "      \"scaleFactor\":       " + p.scaleFactor() + ",\n" +
               "      \"rotationDeg\":       " + p.rotationDeg() + ",\n" +
               "      \"offsetX\":           " + p.offsetX() + ",\n" +
               "      \"offsetY\":           " + p.offsetY() + ",\n" +
               "      \"colourShifted\":     " + p.colourShifted() + ",\n" +
               "      \"occluded\":          " + p.occluded() + ",\n" +
               "      \"occlusionFraction\": " + p.occlusionFraction() + "\n" +
               "    }";
    }

    private static String escape(String s) {
        return s == null ? "" : s.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}


