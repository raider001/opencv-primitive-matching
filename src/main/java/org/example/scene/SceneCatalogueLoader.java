package org.example.scene;

import org.example.factories.BackgroundId;
import org.example.factories.ReferenceId;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Loads the pre-generated scene catalogue from {@code test_output/catalogue_samples/}.
 *
 * <p>Each scene is stored as a pair of files:
 * <ul>
 *   <li>{@code <name>.png}  — the rendered 640×480 scene image</li>
 *   <li>{@code <name>.json} — the {@link SceneMetadata} sidecar</li>
 * </ul>
 *
 * <p>Loading is parallelised across all available processors.  Progress is printed
 * every 5% of files processed.  A detailed summary (counts by category, skipped/failed
 * files) is printed on completion.
 *
 * <p>Two optional filters control which scenes are included:
 * <ul>
 *   <li><b>Reference filter</b> — restrict by {@link ReferenceId}; Cat D negatives
 *       are always included regardless of this filter.</li>
 *   <li><b>Category filter</b> — restrict by {@link SceneCategory}; if empty, all
 *       categories are included.</li>
 * </ul>
 *
 * <p>Falls back to in-memory generation via {@link SceneCatalogue#build()} if the
 * catalogue directory does not exist or yields no valid entries.
 */
public final class SceneCatalogueLoader {

    public static final Path CATALOGUE_DIR =
            Paths.get("test_output", "catalogue_samples");

    private SceneCatalogueLoader() {}

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /**
     * Loads scenes from disk filtered by reference ID only.
     *
     * @param filterRefs zero or more {@link ReferenceId}s; empty = load all references
     */
    public static List<SceneEntry> load(ReferenceId... filterRefs) throws IOException {
        return load(
                filterRefs == null ? new ReferenceId[0] : filterRefs,
                new SceneCategory[0]);
    }

    /**
     * Loads scenes from disk with both a reference filter and a category filter.
     *
     * @param filterRefs  zero or more {@link ReferenceId}s; empty = all references
     * @param filterCats  zero or more {@link SceneCategory}s; empty = all categories
     * @return unmodifiable list of loaded {@link SceneEntry} objects
     */
    public static List<SceneEntry> load(ReferenceId[]   filterRefs,
                                        SceneCategory[] filterCats) throws IOException {
        if (!Files.isDirectory(CATALOGUE_DIR)) {
            System.err.printf("[CatalogueLoader] %s not found — falling back to in-memory generation%n",
                    CATALOGUE_DIR);
            return SceneCatalogue.build();
        }

        Set<ReferenceId>   refFilter = toSet(filterRefs);
        Set<SceneCategory> catFilter = toSet(filterCats);

        // Collect all JSON paths up-front so we know the total for progress reporting
        List<Path> jsonPaths;
        try (Stream<Path> files = Files.list(CATALOGUE_DIR)) {
            jsonPaths = files
                    .filter(p -> p.getFileName().toString().endsWith(".json"))
                    .sorted()
                    .collect(Collectors.toList());
        }

        int total = jsonPaths.size();
        if (total == 0) {
            System.err.printf("[CatalogueLoader] No JSON files in %s — falling back%n", CATALOGUE_DIR);
            return SceneCatalogue.build();
        }

        // ---- Report filters and estimate the expected load count ----------
        // Filenames encode category and referenceId, e.g.:
        //   a_clean__<variant>__<refId>__<bgId>.json
        //   d_negative__<variant>__negative__<bgId>.json
        // We scan filenames (not file contents) to estimate how many will pass
        // the active filters — giving accurate progress reporting.
        int estimated = estimateFilteredCount(jsonPaths, refFilter, catFilter);

        System.out.printf("[CatalogueLoader] Catalogue: %,d files on disk", total);
        if (!refFilter.isEmpty() || !catFilter.isEmpty()) {
            System.out.printf("  →  ~%,d expected after filters", estimated);
        }
        System.out.println();
        if (!refFilter.isEmpty()) {
            System.out.printf("[CatalogueLoader] Reference filter (%d): %s%n",
                    refFilter.size(), refFilter);
        }
        if (!catFilter.isEmpty()) {
            System.out.printf("[CatalogueLoader] Category filter  (%d): %s%n",
                    catFilter.size(), catFilter);
        }

        // ---- Parallel load ------------------------------------------------
        long tStart = System.currentTimeMillis();
        // Progress is tracked against the estimated filtered count so that the
        // percentage reflects actual loading work, not files skipped by filters.
        int  progressTotal = Math.max(1, estimated > 0 ? estimated : total);
        int  reportEvery   = Math.max(1, progressTotal / 20);   // ~5% steps

        ConcurrentLinkedQueue<SceneEntry> bag      = new ConcurrentLinkedQueue<>();
        AtomicInteger processed = new AtomicInteger(0);
        AtomicInteger skipped   = new AtomicInteger(0);
        AtomicInteger failed    = new AtomicInteger(0);

        jsonPaths.parallelStream().forEach(jsonPath -> {
            try {
                LoadResult lr = loadOne(jsonPath, refFilter, catFilter);
                switch (lr.status()) {
                    case OK      -> bag.add(lr.entry());
                    case SKIPPED -> { skipped.incrementAndGet(); return; }  // don't count toward progress
                    case FAILED  -> failed.incrementAndGet();
                }
            } catch (Exception e) {
                failed.incrementAndGet();
                System.err.printf("[CatalogueLoader] Error loading %s: %s%n",
                        jsonPath.getFileName(), e.getMessage());
            }

            // Progress counts only non-skipped files (attempted loads)
            int n = processed.incrementAndGet();
            if (n % reportEvery == 0 || n == progressTotal) {
                long elapsed = System.currentTimeMillis() - tStart;
                System.out.printf("[CatalogueLoader] %5.1f%%  %,d/~%,d  loaded=%,d  failed=%,d  %ds%n",
                        Math.min(100.0, n * 100.0 / progressTotal),
                        n, progressTotal,
                        bag.size(), failed.get(),
                        elapsed / 1000);
            }
        });

        // ---- Sort for deterministic ordering ------------------------------
        List<SceneEntry> entries = new ArrayList<>(bag);
        entries.sort(Comparator.comparing(e -> e.category().name()
                + "|" + (e.primaryReferenceId() != null ? e.primaryReferenceId().name() : "~")
                + "|" + e.variantLabel()));

        if (entries.isEmpty()) {
            System.err.printf("[CatalogueLoader] No entries passed filters — falling back%n");
            return SceneCatalogue.build();
        }

        // ---- Detailed summary ---------------------------------------------
        printSummary(entries, total, skipped.get(), failed.get(),
                System.currentTimeMillis() - tStart);

        return Collections.unmodifiableList(entries);
    }

    // -------------------------------------------------------------------------
    // Filter estimation  (filename-based, no I/O)
    // -------------------------------------------------------------------------

    /**
     * Estimates how many files will pass the active filters by inspecting filenames only.
     *
     * <p>Filename format: {@code <cat>__<variant>__<refIdOrNegative>__<bgId>.json}
     * where {@code <cat>} is the lower-cased category prefix, e.g. {@code a_clean},
     * {@code b_transformed}, {@code c_degraded}, {@code d_negative}.
     *
     * <p>Category D negatives always pass the reference filter.
     */
    private static int estimateFilteredCount(List<Path> jsonPaths,
                                             Set<ReferenceId>   refFilter,
                                             Set<SceneCategory> catFilter) {
        int count = 0;
        for (Path p : jsonPaths) {
            String name = p.getFileName().toString(); // e.g. a_clean__rot_45__circle_outline__bg_random_mixed.json

            // Derive category from filename prefix
            SceneCategory cat = null;
            if      (name.startsWith("a_clean__"))      cat = SceneCategory.A_CLEAN;
            else if (name.startsWith("b_transformed__")) cat = SceneCategory.B_TRANSFORMED;
            else if (name.startsWith("c_degraded__"))   cat = SceneCategory.C_DEGRADED;
            else if (name.startsWith("d_negative__"))   cat = SceneCategory.D_NEGATIVE;

            if (cat == null) continue;
            if (!catFilter.isEmpty() && !catFilter.contains(cat)) continue;

            // Negatives always pass the ref filter
            if (cat == SceneCategory.D_NEGATIVE) { count++; continue; }

            if (refFilter.isEmpty()) { count++; continue; }

            // Extract the refId segment — third __ delimited token
            // format: <cat>__<variant>__<refId>__<bgId>.json
            // variant may contain __ so we split from the right
            String withoutExt = name.substring(0, name.length() - 5); // strip .json
            String[] parts = withoutExt.split("__");
            if (parts.length < 3) { count++; continue; } // can't tell — include pessimistically

            // The refId is the third-to-last token (bgId is last, refId second-to-last,
            // variant is everything in between — but we stored it as cat__variant__ref__bg)
            // so refId = parts[parts.length - 2]
            String refName = parts[parts.length - 2].toUpperCase();
            try {
                ReferenceId ref = ReferenceId.valueOf(refName);
                if (refFilter.contains(ref)) count++;
            } catch (IllegalArgumentException ignored) {
                count++; // unknown token — include pessimistically
            }
        }
        return count;
    }

    // -------------------------------------------------------------------------
    // Single-file loader  →  LoadResult
    // -------------------------------------------------------------------------

    private enum Status { OK, SKIPPED, FAILED }

    private record LoadResult(Status status, SceneEntry entry) {
        static LoadResult ok(SceneEntry e) { return new LoadResult(Status.OK,      e); }
        static LoadResult skipped()        { return new LoadResult(Status.SKIPPED, null); }
        static LoadResult failed()         { return new LoadResult(Status.FAILED,  null); }
    }

    private static LoadResult loadOne(Path jsonPath,
                                      Set<ReferenceId>   refFilter,
                                      Set<SceneCategory> catFilter) throws IOException {
        String json = Files.readString(jsonPath);

        // Fast pre-filter on primaryReferenceId from JSON text before loading the PNG
        ReferenceId primaryRef = parseEnumOrNull(json, "primaryReferenceId", ReferenceId.class);
        if (!refFilter.isEmpty() && primaryRef != null && !refFilter.contains(primaryRef)) {
            return LoadResult.skipped();
        }

        SceneCategory category = parseEnum(json, "category", SceneCategory.class);
        if (category == null) return LoadResult.failed();

        // Category D negatives always pass the ref filter; apply cat filter to all
        if (!catFilter.isEmpty() && !catFilter.contains(category)) {
            return LoadResult.skipped();
        }

        // Load PNG
        String pngName  = jsonPath.getFileName().toString().replace(".json", ".png");
        Path   pngPath  = jsonPath.getParent().resolve(pngName);
        if (!Files.exists(pngPath)) {
            System.err.printf("[CatalogueLoader] Missing PNG for %s%n", jsonPath.getFileName());
            return LoadResult.failed();
        }

        Mat sceneMat = Imgcodecs.imread(pngPath.toAbsolutePath().toString());
        if (sceneMat == null || sceneMat.empty()) {
            System.err.printf("[CatalogueLoader] Could not decode %s%n", pngPath.getFileName());
            return LoadResult.failed();
        }

        // Validate dimensions against JSON metadata
        int expectedW = parseInt(json, "sceneW");
        int expectedH = parseInt(json, "sceneH");
        if (expectedW > 0 && expectedH > 0
                && (sceneMat.cols() != expectedW || sceneMat.rows() != expectedH)) {
            System.err.printf("[CatalogueLoader] Dimension mismatch in %s: expected %dx%d got %dx%d%n",
                    pngPath.getFileName(), expectedW, expectedH,
                    sceneMat.cols(), sceneMat.rows());
            // Still usable — just warn; don't fail
        }

        String       variant    = parseStr(json, "variant");
        BackgroundId background = parseEnum(json, "background", BackgroundId.class);
        List<SceneShapePlacement> placements = parsePlacements(json);

        return LoadResult.ok(new SceneEntry(primaryRef, category, variant,
                background, placements, sceneMat));
    }

    // -------------------------------------------------------------------------
    // Summary printer
    // -------------------------------------------------------------------------

    private static void printSummary(List<SceneEntry> entries, int total,
                                     int skipped, int failed, long elapsedMs) {
        Map<SceneCategory, Long> byCat = entries.stream()
                .collect(Collectors.groupingBy(SceneEntry::category, Collectors.counting()));

        System.out.printf("%n[CatalogueLoader] --- Load complete in %.1f s ---%n",
                elapsedMs / 1000.0);
        System.out.printf("  Total files:  %,d%n", total);
        System.out.printf("  Loaded:       %,d%n", entries.size());
        System.out.printf("  Skipped:      %,d  (did not match filters)%n", skipped);
        System.out.printf("  Failed:       %,d  (bad/missing files)%n", failed);
        System.out.printf("  By category:%n");
        for (SceneCategory cat : SceneCategory.values()) {
            long count = byCat.getOrDefault(cat, 0L);
            if (count > 0) {
                System.out.printf("    %-20s %,d%n", cat.name(), count);
            }
        }
        // Per-reference breakdown (only if ≤ 10 distinct refs — avoids wall of text)
        Map<ReferenceId, Long> byRef = entries.stream()
                .filter(e -> e.primaryReferenceId() != null)
                .collect(Collectors.groupingBy(SceneEntry::primaryReferenceId, Collectors.counting()));
        if (byRef.size() <= 10) {
            System.out.printf("  By reference:%n");
            byRef.entrySet().stream()
                 .sorted(Map.Entry.comparingByKey(Comparator.comparing(Enum::name)))
                 .forEach(e -> System.out.printf("    %-30s %,d scenes%n",
                         e.getKey().name(), e.getValue()));
        } else {
            System.out.printf("  Distinct references: %d%n", byRef.size());
        }
        System.out.println();
    }

    // -------------------------------------------------------------------------
    // Minimal JSON parsers (no external library)
    // -------------------------------------------------------------------------

    private static String parseStr(String json, String key) {
        Pattern p = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*\"([^\"]+)\"");
        Matcher m = p.matcher(json);
        return m.find() ? m.group(1) : "";
    }

    private static <E extends Enum<E>> E parseEnum(String json, String key, Class<E> cls) {
        String val = parseStr(json, key);
        try { return Enum.valueOf(cls, val); }
        catch (IllegalArgumentException | NullPointerException e) { return null; }
    }

    private static <E extends Enum<E>> E parseEnumOrNull(String json, String key, Class<E> cls) {
        Pattern nullP = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*null");
        if (nullP.matcher(json).find()) return null;
        return parseEnum(json, key, cls);
    }

    private static List<SceneShapePlacement> parsePlacements(String json) {
        List<SceneShapePlacement> list = new ArrayList<>();
        int start = json.indexOf("\"placements\"");
        if (start < 0) return list;
        int arrStart = json.indexOf('[', start);
        int arrEnd   = json.lastIndexOf(']');
        if (arrStart < 0 || arrEnd < 0 || arrEnd < arrStart) return list;
        String arr = json.substring(arrStart, arrEnd + 1);

        int depth = 0, blockStart = -1;
        for (int i = 0; i < arr.length(); i++) {
            char c = arr.charAt(i);
            if (c == '{') {
                if (depth++ == 0) blockStart = i;
            } else if (c == '}') {
                if (--depth == 0 && blockStart >= 0) {
                    SceneShapePlacement p = parsePlacement(arr.substring(blockStart, i + 1));
                    if (p != null) list.add(p);
                    blockStart = -1;
                }
            }
        }
        return list;
    }

    private static SceneShapePlacement parsePlacement(String block) {
        try {
            ReferenceId ref = parseEnum(block, "referenceId", ReferenceId.class);
            if (ref == null) return null;

            // Parse rect fields from the nested placedRect object precisely
            int x = parseRectField(block, "x");
            int y = parseRectField(block, "y");
            int w = parseRectField(block, "w");
            int h = parseRectField(block, "h");

            double  scale    = parseDouble (block, "scaleFactor");
            double  rot      = parseDouble (block, "rotationDeg");
            int     offX     = parseInt    (block, "offsetX");
            int     offY     = parseInt    (block, "offsetY");
            boolean shifted  = parseBoolean(block, "colourShifted");
            boolean occluded = parseBoolean(block, "occluded");
            double  occFrac  = parseDouble (block, "occlusionFraction");

            return new SceneShapePlacement(ref, new Rect(x, y, w, h),
                    scale, rot, offX, offY, shifted, occluded, occFrac);
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * Parses a single integer field from inside the {@code placedRect} object,
     * e.g. {@code "x":256}.  Uses a word-boundary anchor so "x" doesn't match "offsetX".
     */
    private static int parseRectField(String block, String key) {
        // Restrict search to the placedRect sub-object
        int rectStart = block.indexOf("\"placedRect\"");
        int rectOpen  = rectStart >= 0 ? block.indexOf('{', rectStart) : -1;
        int rectClose = rectOpen  >= 0 ? block.indexOf('}', rectOpen)  : -1;
        String scope  = (rectOpen >= 0 && rectClose > rectOpen)
                ? block.substring(rectOpen, rectClose + 1) : block;

        Pattern p = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*(-?\\d+)");
        Matcher m = p.matcher(scope);
        return m.find() ? Integer.parseInt(m.group(1)) : 0;
    }

    private static int parseInt(String json, String key) {
        Pattern p = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*(-?\\d+)");
        Matcher m = p.matcher(json);
        return m.find() ? Integer.parseInt(m.group(1)) : 0;
    }

    private static double parseDouble(String json, String key) {
        Pattern p = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*(-?[\\d.Ee+]+)");
        Matcher m = p.matcher(json);
        return m.find() ? Double.parseDouble(m.group(1)) : 0.0;
    }

    private static boolean parseBoolean(String json, String key) {
        Pattern p = Pattern.compile("\"" + Pattern.quote(key) + "\"\\s*:\\s*(true|false)");
        Matcher m = p.matcher(json);
        return m.find() && m.group(1).equals("true");
    }

    @SafeVarargs
    private static <E extends Enum<E>> Set<E> toSet(E... values) {
        if (values == null || values.length == 0) return Collections.emptySet();
        return EnumSet.copyOf(Arrays.asList(values));
    }
}





