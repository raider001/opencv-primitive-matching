package org.example;

import org.junit.jupiter.api.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Milestone 4 — Scene Catalogue analytical test.
 *
 * Builds the full catalogue, verifies counts, prints a metadata spot-check,
 * saves sample scenes as clean PNG + JSON sidecar (no overlay baked into image),
 * and writes a contact sheet.  No assertions — always passes.
 */
@DisplayName("Milestone 4 — Scene Catalogue")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class SceneCatalogueTest {

    private static final Path OUT = Paths.get("test_output", "catalogue_samples");
    private static List<SceneEntry>   catalogue;
    private static List<SceneEntry>   multiShapeScenes;

    @BeforeAll
    static void setup() throws IOException {
        OpenCvLoader.load();
        Files.createDirectories(OUT.toAbsolutePath());
        System.out.println("\nBuilding scene catalogue — this may take a few seconds...");
        long t0 = System.currentTimeMillis();
        catalogue = SceneCatalogue.build();
        multiShapeScenes = SceneGenerator.buildMultiShape();
        long elapsed = System.currentTimeMillis() - t0;
        System.out.printf("Catalogue built: %d scenes + %d multi-shape samples in %d ms%n%n",
                catalogue.size(), multiShapeScenes.size(), elapsed);
    }

    // -------------------------------------------------------------------------
    // Test 1 — counts
    // -------------------------------------------------------------------------

    @Test @Order(1)
    @DisplayName("Verify scene counts per category")
    void verifyCounts() {
        long a = catalogue.stream().filter(e -> e.category() == SceneCategory.A_CLEAN).count();
        long b = catalogue.stream().filter(e -> e.category() == SceneCategory.B_TRANSFORMED).count();
        long c = catalogue.stream().filter(e -> e.category() == SceneCategory.C_DEGRADED).count();
        long d = catalogue.stream().filter(e -> e.category() == SceneCategory.D_NEGATIVE).count();

        System.out.println("=== Scene counts ===");
        System.out.printf("  A (clean)       : %4d  (expected 352)%n",  a);
        System.out.printf("  B (transformed) : %4d  (expected 1320)%n", b);
        System.out.printf("  C (degraded)    : %4d  (expected 616)%n",  c);
        System.out.printf("  D (negative)    : %4d  (expected 44)%n",   d);
        System.out.printf("  Total           : %4d  (expected 2332)%n%n", catalogue.size());
        System.out.printf("  Multi-shape     : %4d%n%n", multiShapeScenes.size());
    }

    // -------------------------------------------------------------------------
    // Test 2 — metadata spot-check
    // -------------------------------------------------------------------------

    @Test @Order(2)
    @DisplayName("Print metadata spot-check for 12 random scenes")
    void printMetadataSpotCheck() {
        Random rng = new Random(77L);
        List<SceneEntry> sample = new ArrayList<>(catalogue);
        Collections.shuffle(sample, rng);
        sample = sample.subList(0, Math.min(12, sample.size()));

        String fmt = "%-6s  %-24s  %-26s  %3s  %s%n";
        String sep = "-".repeat(92);
        System.out.println("=== Metadata spot-check (12 random scenes) ===");
        System.out.println(sep);
        System.out.printf(fmt, "Cat", "Variant", "Background", "Pl#", "Placement summary");
        System.out.println(sep);
        for (SceneEntry e : sample) {
            String placement = e.placements().isEmpty() ? "(none)"
                    : e.placements().stream()
                        .map(p -> p.referenceId().name() + " " + p.summary())
                        .reduce((a, b) -> a + " | " + b).orElse("(none)");
            System.out.printf(fmt,
                    e.category().name().charAt(0) + "",
                    trunc(e.variantLabel(), 24),
                    trunc(e.backgroundId().name(), 26),
                    e.placements().size(),
                    trunc(placement, 60));
        }
        System.out.println(sep);
        System.out.println();
    }

    // -------------------------------------------------------------------------
    // Test 3 — save ALL catalogue scenes as PNG + JSON sidecar
    // -------------------------------------------------------------------------

    @Test @Order(3)
    @DisplayName("Save all catalogue scenes: clean PNG + JSON sidecar (no overlay in image)")
    void saveSamples() throws IOException {
        int saved = 0;
        int total = catalogue.size();
        System.out.printf("Saving all %d catalogue scenes to %s ...%n", total, OUT.toAbsolutePath());
        for (SceneEntry scene : catalogue) {
            String baseName = buildSampleName(scene);
            saveSample(scene, baseName);
            saved++;
            if (saved % 200 == 0 || saved == total) {
                System.out.printf("  %d / %d saved%n", saved, total);
            }
        }
        // Also save multi-shape scenes
        for (SceneEntry scene : multiShapeScenes) {
            String baseName = buildSampleName(scene);
            saveSample(scene, baseName);
            saved++;
        }
        System.out.printf("Saved %d scenes total (PNG + JSON) to %s%n%n",
                saved, OUT.toAbsolutePath());
    }

    // -------------------------------------------------------------------------
    // Test 4 — annotated overlays (separate directory, for visual debugging)
    // -------------------------------------------------------------------------

    @Test @Order(4)
    @DisplayName("Save annotated overlays with ground-truth rect drawn (separate dir)")
    void saveAnnotatedOverlays() throws IOException {
        Path annoDir = OUT.toAbsolutePath().resolve("annotated");
        Files.createDirectories(annoDir);

        int saved = 0;
        for (SceneEntry scene : catalogue) {
            String baseName = buildSampleName(scene);
            Mat annotated = drawOverlay(scene);
            Imgcodecs.imwrite(annoDir.resolve(baseName + ".png").toString(), annotated);
            annotated.release();
            saved++;
        }
        for (SceneEntry scene : multiShapeScenes) {
            String baseName = buildSampleName(scene);
            Mat annotated = drawOverlay(scene);
            Imgcodecs.imwrite(annoDir.resolve(baseName + ".png").toString(), annotated);
            annotated.release();
            saved++;
        }
        System.out.printf("Saved %d annotated overlays to %s%n%n", saved, annoDir);
    }

    // -------------------------------------------------------------------------
    // Test 5 — contact sheet
    // -------------------------------------------------------------------------

    @Test @Order(5)
    @DisplayName("Save 6x6 contact sheet")
    void saveContactSheet() {
        int cols = 6, rows = 6, thumbW = 128, thumbH = 96;
        Mat sheet = Mat.zeros(rows * thumbH, cols * thumbW, CvType.CV_8UC3);

        int total = catalogue.size();
        int step  = Math.max(1, total / (cols * rows));
        int pos   = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int idx = Math.min(pos * step, total - 1);
                Mat thumb = new Mat();
                Imgproc.resize(catalogue.get(idx).sceneMat(), thumb,
                        new Size(thumbW, thumbH));
                Mat roi = sheet.submat(new Rect(c * thumbW, r * thumbH, thumbW, thumbH));
                thumb.copyTo(roi);
                thumb.release();
                roi.release();
                pos++;
            }
        }
        Path outPath = OUT.toAbsolutePath().resolve("contact_sheet.png");
        Imgcodecs.imwrite(outPath.toString(), sheet);
        sheet.release();
        System.out.printf("Contact sheet saved to %s%n%n", outPath);
    }


    // -------------------------------------------------------------------------
    // I/O helpers
    // -------------------------------------------------------------------------

    /**
     * Saves a scene as:
     *  - {@code <baseName>.png}   — clean scene image, no annotations
     *  - {@code <baseName>.json}  — JSON sidecar with full placement metadata
     */
    private static void saveSample(SceneEntry scene, String baseName) throws IOException {
        // Clean PNG — exactly what the matcher will receive (no overlays)
        Path pngPath = OUT.toAbsolutePath().resolve(baseName + ".png");
        Imgcodecs.imwrite(pngPath.toString(), scene.sceneMat());

        // JSON sidecar
        Path jsonPath = OUT.toAbsolutePath().resolve(baseName + ".json");
        String json = SceneMetadata.toJson(baseName, scene);
        Files.write(jsonPath, json.getBytes(StandardCharsets.UTF_8));
    }

    /**
     * Builds a consistent filename from scene metadata, safe for all OS filesystems.
     * Format: {@code <cat>_<variant>_<refId>_<bgId>}
     */
    private static String buildSampleName(SceneEntry scene) {
        String cat = scene.category().name().toLowerCase();
        String ref = scene.primaryReferenceId() != null
                ? scene.primaryReferenceId().name().toLowerCase() : "negative";
        String variant = scene.variantLabel().toLowerCase().replaceAll("[^a-z0-9_]", "_");
        String bg  = scene.backgroundId().name().toLowerCase();
        // Truncate to keep filenames reasonable
        return cat + "__" + variant + "__" + ref + "__" + bg;
    }

    /**
     * Returns a clone of the scene Mat with green ground-truth bounding boxes
     * and reference labels drawn over it — for visual debugging only.
     */
    private static Mat drawOverlay(SceneEntry scene) {
        Mat m = scene.sceneMat().clone();
        for (SceneShapePlacement p : scene.placements()) {
            Imgproc.rectangle(m,
                    new Point(p.placedRect().x, p.placedRect().y),
                    new Point(p.placedRect().x + p.placedRect().width,
                              p.placedRect().y + p.placedRect().height),
                    new Scalar(0, 255, 0), 2);
            Imgproc.putText(m,
                    p.referenceId().name() + (p.rotationDeg() != 0
                            ? String.format(" r=%.0f", p.rotationDeg()) : ""),
                    new Point(p.placedRect().x + 2, Math.max(14, p.placedRect().y - 4)),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.35, new Scalar(0, 255, 0), 1);
        }
        Imgproc.putText(m,
                scene.category().name() + " | " + scene.variantLabel()
                + " | shapes=" + scene.placements().size(),
                new Point(4, 14), Imgproc.FONT_HERSHEY_SIMPLEX, 0.38,
                new Scalar(255, 255, 255), 1);
        return m;
    }

    private static String trunc(String s, int max) {
        return s == null ? "" : s.length() <= max ? s : s.substring(0, max - 1) + "~";
    }
}
