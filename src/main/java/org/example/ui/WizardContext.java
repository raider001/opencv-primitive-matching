package org.example.ui;

import org.example.*;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;


/**
 * Shared state and helper methods used by all five wizard-step panels.
 *
 * <p>A single instance is created by {@link BenchmarkLauncher} and passed to
 * each panel on construction.  All catalogue-file scanning and count arithmetic
 * lives here so the panels stay thin.
 */
public final class WizardContext {

    // ── Catalogue file cache ───────────────────────────────────────────────
    /** Lower-cased filenames of every file in {@code test_output/catalogue_samples/}. */
    private Set<String> catFileNames = Collections.emptySet();

    // ── Matcher list ───────────────────────────────────────────────────────
    public final List<MatcherDescriptor> matcherList = new ArrayList<>(MatcherRegistry.ALL);

    // ── Fixed dimension constants ──────────────────────────────────────────
    /** Total non-negative SceneVariant count. */
    public static final int N_SCENE_VARIANTS = (int) Arrays.stream(SceneVariant.values())
            .filter(sv -> sv.category() != SceneCategory.D_NEGATIVE).count();

    /** Total BackgroundId count. */
    public static final int N_BACKGROUNDS = BackgroundId.values().length;

    /** Total ReferenceId count. */
    public static final int N_REFERENCES = ReferenceId.values().length;

    // ── Expected totals (all expressed as scene pairs) ─────────────────────

    /** Expected scene pairs for one variant = References × Backgrounds × SceneVariants. */
    public static int expectedVariantTotal() {
        return N_REFERENCES * N_BACKGROUNDS * N_SCENE_VARIANTS;
    }

    /** Expected scene pairs for one reference = Backgrounds × SceneVariants. */
    public static int expectedRefTotal() {
        return N_BACKGROUNDS * N_SCENE_VARIANTS;
    }

    /**
     * Expected scene pairs for one matcher = Variants × References × Backgrounds × SceneVariants.
     */
    public static int expectedMatcherTotal(int variantCount) {
        return variantCount * N_REFERENCES * N_BACKGROUNDS * N_SCENE_VARIANTS;
    }

    /**
     * Expected scene pairs for one background = References × SceneVariants.
     * (How many catalogue entries this background should have across all refs and scene variants.)
     */
    public static int expectedBackgroundTotal() {
        return N_REFERENCES * N_SCENE_VARIANTS;
    }

    /**
     * Expected scene pairs for one scene variant = References × Backgrounds.
     * (How many catalogue entries this scene variant should have across all refs and backgrounds.)
     */
    public static int expectedSceneTotal() {
        return N_REFERENCES * N_BACKGROUNDS;
    }

    /**
     * Counts all generated annotated PNGs for a matcher across every variant dir:
     * sum of {@code annotated/{variant}/*.png} for all variants.
     */
    public static int countGeneratedMatcherFiles(MatcherDescriptor md) {
        int total = 0;
        for (String vn : md.variantNames()) {
            total += countGeneratedVariantFiles(md, vn);
        }
        return total;
    }

    // ── Catalogue cache ────────────────────────────────────────────────────

    /** Reloads the catalogue filename cache from disk. */
    public void reloadCatalogueFileNames() {
        Set<String> files = new HashSet<>();
        try {
            Path catDir = Paths.get("test_output", "catalogue_samples");
            if (Files.isDirectory(catDir)) {
                try (var s = Files.list(catDir)) {
                    s.map(p -> p.getFileName().toString().toLowerCase()).forEach(files::add);
                }
            }
        } catch (IOException ignored) {}
        catFileNames = files;
    }

    public Set<String> catFileNames() { return catFileNames; }

    // ── Generation-count helpers ───────────────────────────────────────────

    /**
     * Counts generated PNGs for a specific matcher variant by scanning:
     * {@code outputDir/annotated/{sanitised(variant)}/}.
     */
    public static int countGeneratedVariantFiles(MatcherDescriptor md, String variant) {
        Path varDir = md.outputDir().resolve("annotated").resolve(sanitiseVariant(variant));
        if (!Files.isDirectory(varDir)) return 0;
        try (var s = Files.list(varDir)) {
            return (int) s.filter(p -> p.toString().endsWith(".png")).count();
        } catch (IOException e) { return 0; }
    }

    /**
     * Counts generated PNGs for a specific (matcher variant, reference) pair by
     * scanning: {@code outputDir/annotated/{sanitised(variant)}/{RID}_vs_*.png}.
     */
    public static int countGeneratedForRefInVariant(MatcherDescriptor md,
                                                     String variant, ReferenceId rid) {
        Path varDir = md.outputDir().resolve("annotated").resolve(sanitiseVariant(variant));
        if (!Files.isDirectory(varDir)) return 0;
        String prefix = rid.name().toLowerCase() + "_vs_";
        try (var s = Files.list(varDir)) {
            return (int) s.filter(p -> {
                String name = p.getFileName().toString().toLowerCase();
                return name.startsWith(prefix) && name.endsWith(".png");
            }).count();
        } catch (IOException e) { return 0; }
    }

    /**
     * Counts how many catalogue scene pairs (PNG entries) exist for the given background.
     * Each catalogue entry = one scene pair (one ref × one scene variant for this bg).
     * Total possible = {@link #expectedBackgroundTotal()}.
     */
    public int countGeneratedForBackground(BackgroundId bg) {
        if (catFileNames.isEmpty()) return 0;
        String bgSeg = "__" + bg.name().toLowerCase();
        return (int) catFileNames.stream()
                .filter(n -> n.endsWith(".png") && n.contains(bgSeg))
                .count();
    }

    /**
     * Counts how many catalogue scene pairs (PNG entries) exist for the given scene variant.
     * Each catalogue entry = one scene pair (one ref × one background for this scene variant).
     * Total possible = {@link #expectedSceneTotal()}.
     */
    public int countGeneratedForScene(SceneVariant sv) {
        if (catFileNames.isEmpty()) return 0;
        String key = "__" + sv.label().replace(".", "-").replace(" ", "-").toLowerCase() + "__";
        return (int) catFileNames.stream()
                .filter(n -> n.endsWith(".png") && n.contains(key))
                .count();
    }

    /**
     * Returns whether at least one catalogue entry exists for the given scene variant.
     */
    public boolean isCatalogueSceneGenerated(SceneVariant sv) {
        return countGeneratedForScene(sv) > 0;
    }

    /** Counts PNG files in a directory (non-recursive). */
    public static int countPngs(Path dir) {
        try (var s = Files.list(dir)) {
            return (int) s.filter(p -> p.toString().endsWith(".png")).count();
        } catch (IOException e) { return 0; }
    }

    /**
     * Mirrors the {@code sanitise()} logic used inside every matcher to name variant dirs:
     * replaces any character outside {@code [A-Za-z0-9_-]} with {@code _}.
     */
    public static String sanitiseVariant(String name) {
        return name.replaceAll("[^A-Za-z0-9_\\-]", "_");
    }

    /** Mirrors the filename logic in SceneCatalogueTest so names are consistent. */
    public static String buildCatalogueName(SceneEntry scene) {
        String cat     = scene.category().name().toLowerCase();
        String ref     = scene.primaryReferenceId() != null
                ? scene.primaryReferenceId().name().toLowerCase() : "negative";
        String variant = scene.variantLabel().toLowerCase().replaceAll("[^a-z0-9_]", "_");
        String bg      = scene.backgroundId().name().toLowerCase();
        return cat + "__" + variant + "__" + ref + "__" + bg;
    }

    // ── Domain grouping helpers ────────────────────────────────────────────

    public static String groupOf(ReferenceId rid) {
        String n = rid.name();
        if (n.startsWith("LINE_"))        return "Lines";
        if (n.startsWith("CIRCLE_") || n.startsWith("ELLIPSE_")) return "Circles & Ellipses";
        if (n.startsWith("RECT_ROTATED")) return "Rotated Rectangles";
        if (n.startsWith("RECT_"))        return "Rectangles";
        if (n.startsWith("TRIANGLE_") || n.startsWith("PENTAGON_") || n.startsWith("HEXAGON_") ||
            n.startsWith("HEPTAGON_")  || n.startsWith("OCTAGON_")  || n.startsWith("STAR_"))
                                           return "Polygons & Stars";
        if (n.startsWith("POLYLINE_"))    return "Polylines";
        if (n.startsWith("ARC_"))         return "Arcs";
        if (n.startsWith("CONCAVE_") || n.startsWith("IRREGULAR_")) return "Concave / Irregular";
        if (n.startsWith("COMPOUND_"))    return "Compound Shapes";
        if (n.startsWith("GRID_") || n.startsWith("CHECKER") || n.equals("CROSSHAIR"))
                                           return "Grids & Patterns";
        if (n.startsWith("TEXT_"))         return "Text";
        if (n.startsWith("BICOLOUR_") || n.startsWith("TRICOLOUR_")) return "Multi-Colour";
        return "Other";
    }

    public static String bgGroupOf(BackgroundId bg) {
        String n = bg.name();
        if (n.startsWith("BG_SOLID"))    return "Solid";
        if (n.startsWith("BG_GRADIENT")) return "Gradient";
        if (n.startsWith("BG_NOISE"))    return "Noise";
        if (n.startsWith("BG_GRID"))     return "Grid";
        if (n.startsWith("BG_RANDOM"))   return "Random";
        return "Other";
    }
}

