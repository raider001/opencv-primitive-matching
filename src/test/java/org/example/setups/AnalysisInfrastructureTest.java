package org.example.setups;

import org.example.*;
import org.example.analytics.AnalysisOutputWriter;
import org.example.analytics.AnalysisResult;
import org.example.analytics.PerformanceProfile;
import org.example.analytics.PerformanceProfiler;
import org.example.factories.ReferenceId;
import org.example.factories.ReferenceImageFactory;
import org.example.scene.SceneCatalogue;
import org.example.scene.SceneCategory;
import org.example.scene.SceneEntry;
import org.junit.jupiter.api.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Milestone 5 — Analysis Infrastructure analytical test.
 *
 * Validates the full pipeline (AnalysisResult → PerformanceProfiler → AnalysisOutputWriter
 * → HtmlReportWriter) using a deterministic dummy matcher that returns predictable scores.
 *
 * No real matching is performed. No assertions — always passes.
 */
@DisplayName("Milestone 5 — Analysis Infrastructure")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class AnalysisInfrastructureTest {

    private static final Path OUT = Paths.get("test_output", "infrastructure_test");

    // Dummy method variant names
    private static final String[] VARIANTS = {
        "DummyMatcher_FAST",
        "DummyMatcher_MEDIUM",
        "DummyMatcher_SLOW",
        "DummyMatcher_FAST_CF_LOOSE",
        "DummyMatcher_FAST_CF_TIGHT",
    };

    private static List<AnalysisResult>   results;
    private static List<PerformanceProfile> profiles;

    // -------------------------------------------------------------------------
    // Setup — run dummy matcher
    // -------------------------------------------------------------------------

    @BeforeAll
    static void runDummyMatcher() throws IOException {
        OpenCvLoader.load();
        Files.createDirectories(OUT.toAbsolutePath());
        Files.createDirectories(OUT.resolve("annotated").toAbsolutePath());

        // Pick 5 reference IDs for the test
        ReferenceId[] refs = Arrays.copyOf(ReferenceId.values(), 5);

        // Pick 10 scenes from the catalogue (2 per category)
        List<SceneEntry> catalogue = SceneCatalogue.build();
        List<SceneEntry> scenes = selectSampleScenes(catalogue, 10);

        results = new ArrayList<>();
        Random rng = new Random(42L);

        System.out.printf("%nRunning dummy matcher: %d refs × %d scenes × %d variants = %d results%n%n",
                refs.length, scenes.size(), VARIANTS.length,
                refs.length * scenes.size() * VARIANTS.length);

        for (ReferenceId ref : refs) {
            Mat refMat = ReferenceImageFactory.build(ref);
            for (SceneEntry scene : scenes) {
                for (String variant : VARIANTS) {
                    long t0 = System.currentTimeMillis();

                    // Simulate work: delay proportional to variant speed tier
                    simulateWork(variant, rng);

                    long elapsed = System.currentTimeMillis() - t0;
                    long preFilter = variant.contains("_CF_") ? (rng.nextInt(3) + 1) : 0L;

                    // Dummy score: base depends on category, with noise
                    double score = dummyScore(variant, scene.category(), rng);

                    // Annotated mat: clone scene, draw a dummy rect + score label
                    Mat annotated = buildAnnotatedMat(scene.sceneMat(), ref, score);

                    // Dummy bounding rect: centre of scene
                    Rect bbox = new Rect(240, 160, 160, 120);

                    results.add(new AnalysisResult(
                            variant, ref, scene.variantLabel(), scene.category(),
                            scene.backgroundId(), score, bbox,
                            elapsed, preFilter, scene.sceneMat().cols() * scene.sceneMat().rows(),
                            null, false, null));
                }
            }
            refMat.release();
        }

        System.out.printf("Dummy matcher done: %d results collected.%n%n", results.size());
    }

    // -------------------------------------------------------------------------
    // Test 1 — ASCII table
    // -------------------------------------------------------------------------

    @Test @Order(1)
    @DisplayName("Print ASCII results table")
    void printAsciiResults() {
        System.out.println("=== ASCII Results Table (first 30 rows) ===");
        AnalysisOutputWriter.printAsciiTable(results.subList(0, Math.min(30, results.size())));
    }

    // -------------------------------------------------------------------------
    // Test 2 — Annotated images + reference grids
    // -------------------------------------------------------------------------

    @Test @Order(2)
    @DisplayName("Save annotated images")
    void saveImages() {
        AnalysisOutputWriter.saveAnnotatedImages(results, OUT.resolve("annotated"));
        System.out.printf("Annotated images saved to %s%n%n", OUT.toAbsolutePath());
    }

    // -------------------------------------------------------------------------
    // Test 3 — Performance profiling + ASCII table
    // -------------------------------------------------------------------------

    @Test @Order(3)
    @DisplayName("Compute performance profiles and print table")
    void computeProfiles() {
        profiles = new ArrayList<>();
        for (String variant : VARIANTS) {
            profiles.add(PerformanceProfiler.profile(results, variant));
        }
        System.out.println("=== Performance Table ===");
        AnalysisOutputWriter.printPerformanceTable(profiles);

        System.out.println("=== Interpretation Notes ===");
        for (PerformanceProfile p : profiles) {
            System.out.println("  " + PerformanceProfiler.interpretationNote(p));
        }
        System.out.println();
    }

    // -------------------------------------------------------------------------
    // Test 4 — HTML report
    // -------------------------------------------------------------------------

    @Test @Order(4)
    @DisplayName("Write HTML report")
    void writeHtmlReport() throws IOException {
        Path reportPath = OUT.toAbsolutePath().resolve("report.html");
        HtmlReportWriter.write(results, profiles, "Dummy Matcher (Infrastructure Test)", reportPath);
        System.out.printf("HTML report written to %s%n%n", reportPath);
        System.out.printf("  → Open in a browser to verify Results and Performance tabs.%n%n");
    }

    // -------------------------------------------------------------------------
    // Dummy matcher helpers
    // -------------------------------------------------------------------------

    private static void simulateWork(String variant, Random rng) {
        // Simulate varying execution times by spinning
        int baseUs;
        if (variant.contains("SLOW"))        baseUs = 8;
        else if (variant.contains("MEDIUM"))  baseUs = 3;
        else                                  baseUs = 1;
        // Busy-wait for a tiny amount — keeps elapsed times in realistic range
        long deadline = System.nanoTime() + (baseUs + rng.nextInt(baseUs)) * 1_000_000L;
        while (System.nanoTime() < deadline) { /* spin */ }
    }

    private static double dummyScore(String variant, SceneCategory cat, Random rng) {
        // Base score by category (A = easiest, D = lowest)
        double base = switch (cat) {
            case A_CLEAN      -> 80.0;
            case B_TRANSFORMED -> 60.0;
            case C_DEGRADED    -> 40.0;
            case D_NEGATIVE    -> 15.0;
        };
        // CF variants slightly better on A/B, slightly worse on C
        if (variant.contains("_CF_")) {
            base += (cat == SceneCategory.C_DEGRADED) ? -5 : 5;
        }
        // Add ±15% noise
        return Math.max(0, Math.min(100, base + (rng.nextDouble() - 0.5) * 30));
    }

    private static Mat buildAnnotatedMat(Mat scene, ReferenceId ref, double score) {
        Mat m = scene.clone();
        // Draw a dummy result rectangle
        Imgproc.rectangle(m, new Point(240, 160), new Point(400, 280),
                new Scalar(0, 200, 0), 2);
        // Score label
        Imgproc.putText(m, String.format("%.1f%%", score),
                new Point(242, 156), Imgproc.FONT_HERSHEY_SIMPLEX,
                0.5, new Scalar(0, 255, 0), 1);
        // Reference name
        Imgproc.putText(m, ref.name(),
                new Point(4, 14), Imgproc.FONT_HERSHEY_SIMPLEX,
                0.38, new Scalar(255, 255, 255), 1);
        return m;
    }

    /** Picks up to {@code n} scenes spread across all four categories. */
    private static List<SceneEntry> selectSampleScenes(List<SceneEntry> catalogue, int n) {
        List<SceneEntry> selected = new ArrayList<>();
        int perCat = Math.max(1, n / SceneCategory.values().length);
        for (SceneCategory cat : SceneCategory.values()) {
            catalogue.stream()
                    .filter(e -> e.category() == cat)
                    .limit(perCat)
                    .forEach(selected::add);
        }
        return selected.subList(0, Math.min(n, selected.size()));
    }
}

