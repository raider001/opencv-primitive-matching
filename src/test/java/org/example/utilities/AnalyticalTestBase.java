package org.example.utilities;

import org.example.*;
import org.junit.jupiter.api.*;
import org.opencv.core.Mat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Shared base for all analytical technique tests (Milestones 7–16).
 *
 * <h2>Catalogue source</h2>
 * Scenes are loaded from the pre-generated files in
 * {@code test_output/catalogue_samples/} (PNG + JSON sidecar).
 * Each result is evaluated against its scene's ground-truth placement to
 * produce a {@link DetectionVerdict} (TP / FP / FN / TN).
 *
 * <h2>Reference filter</h2>
 * Override {@link #referenceFilter()} to return a non-empty array of
 * {@link ReferenceId}s — only those references (plus all Cat D negatives)
 * will be loaded.  Return an empty array (default) to run all 88 references.
 *
 * <h2>Debug mode</h2>
 * When {@link #debugMode()} returns {@code true}, a 3-scene in-memory debug
 * catalogue is used instead of loading from disk (fast iteration during dev).
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public abstract class AnalyticalTestBase {

    // -------------------------------------------------------------------------
    // Abstract configuration — subclasses fill these in
    // -------------------------------------------------------------------------

    /** Short uppercase tag used in log lines, e.g. {@code "TM"}, {@code "FM"}. */
    protected abstract String tag();

    /** Human-readable technique name for the HTML report title. */
    protected abstract String techniqueName();

    /** Root output directory, e.g. {@code Paths.get("test_output","template_matching")}. */
    protected abstract Path outputDir();

    /**
     * When {@code true}, only {@link #debugRef()} is used and the slim debug catalogue
     * (3 scenes) is loaded instead of the full slim catalogue.
     */
    protected abstract boolean debugMode();

    /** The single {@link ReferenceId} used when {@link #debugMode()} is {@code true}. */
    protected abstract ReferenceId debugRef();

    /**
     * Variant names whose annotated PNG should be saved to disk.
     * Pass an empty set to skip all image saving.
     */
    protected abstract Set<String> saveVariants();

    /**
     * Override to restrict which reference IDs are loaded from the catalogue.
     * Return an empty array (default) to run all references.
     * Category D (negative) scenes are always included regardless of this filter.
     *
     * <p>Example — run only circle references:
     * <pre>{@code
     * \@Override protected ReferenceId[] referenceFilter() {
     *     return new ReferenceId[]{ ReferenceId.CIRCLE_OUTLINE, ReferenceId.CIRCLE_FILLED };
     * }
     * }</pre>
     */
    protected ReferenceId[] referenceFilter() { return new ReferenceId[0]; }

    /**
     * Override to skip individual scenes from the loaded catalogue.
     * Return {@code true} to include the scene, {@code false} to exclude it.
     * Default accepts all scenes.
     *
     * <p>Example — only run clean and negative scenes:
     * <pre>{@code
     * \@Override protected boolean sceneFilter(SceneEntry scene) {
     *     return scene.category() == SceneCategory.A_CLEAN
     *         || scene.category() == SceneCategory.D_NEGATIVE;
     * }
     * }</pre>
     */
    protected boolean sceneFilter(SceneEntry scene) { return true; }

    /**
     * Runs the technique-specific matcher for one (reference, scene) pair.
     *
     * @param refId        the reference being searched for
     * @param refMat       128×128 BGR reference Mat — caller retains ownership
     * @param scene        the scene to search in
     * @param saveVariants variant names that should produce an annotated PNG on disk
     * @param outputDir    root output directory
     * @return one {@link AnalysisResult} per variant
     */
    protected abstract List<AnalysisResult> runMatcher(ReferenceId refId,
                                                        Mat refMat,
                                                        SceneEntry scene,
                                                        Set<String> saveVariants,
                                                        Path outputDir);

    // -------------------------------------------------------------------------
    // Shared state
    // -------------------------------------------------------------------------

    private List<SceneEntry>                   catalogue;
    private List<AnalysisResult>               results;
    private Map<AnalysisResult, SceneEntry>    resultSceneMap;
    private Path                               absOutputDir;
    private ProgressDisplay                    progress;

    // -------------------------------------------------------------------------
    // @BeforeAll — catalogue loading
    // -------------------------------------------------------------------------

    @BeforeAll
    void setup() throws IOException {
        OpenCvLoader.load();
        absOutputDir = outputDir().toAbsolutePath().normalize();

        // Clear the output directory so stale results from a previous run
        // are never mixed with the current one.
        if (Files.isDirectory(absOutputDir)) {
            System.out.printf("[%s] Clearing output directory: %s%n", tag(), absOutputDir);
            clearDirectory(absOutputDir);
        }
        Files.createDirectories(absOutputDir);

        if (debugMode()) {
            System.out.printf("%n[%s] *** DEBUG MODE — ref: %s, 3-scene debug catalogue ***%n",
                    tag(), debugRef());
            catalogue = SceneCatalogue.buildDebug(debugRef());
        } else {
            System.out.printf("%n[%s] Loading catalogue from disk...%n", tag());
            long t0 = System.currentTimeMillis();
            List<SceneEntry> raw = SceneCatalogueLoader.load(referenceFilter());

            // Built-in guard: for A/B/C scenes, only keep those whose primaryReferenceId
            // is in the referenceFilter set.  D_NEGATIVE scenes (primaryRef==null) always
            // pass.  This prevents scenes for unrequested shapes flooding the results and
            // making it look like the matcher found the wrong image.
            Set<ReferenceId> refSet = referenceFilter().length > 0
                    ? Set.of(referenceFilter()) : Set.of();
            List<SceneEntry> refFiltered = raw.stream().filter(s -> {
                if (s.category() == SceneCategory.D_NEGATIVE) return true;
                if (refSet.isEmpty()) return true;
                return s.primaryReferenceId() != null && refSet.contains(s.primaryReferenceId());
            }).toList();

            // Apply subclass scene filter (variant-label / category restrictions)
            catalogue = refFiltered.stream().filter(this::sceneFilter).toList();
            int excluded = raw.size() - catalogue.size();

            System.out.printf("[%s] Catalogue: %d scenes  (A=%d B=%d C=%d D=%d)%s  in %d ms%n%n",
                    tag(), catalogue.size(),
                    catalogue.stream().filter(s -> s.category() == SceneCategory.A_CLEAN).count(),
                    catalogue.stream().filter(s -> s.category() == SceneCategory.B_TRANSFORMED).count(),
                    catalogue.stream().filter(s -> s.category() == SceneCategory.C_DEGRADED).count(),
                    catalogue.stream().filter(s -> s.category() == SceneCategory.D_NEGATIVE).count(),
                    excluded > 0 ? "  [" + excluded + " excluded by scene filter]" : "",
                    System.currentTimeMillis() - t0);
        }
    }

    // -------------------------------------------------------------------------
    // @Test — parallel double loop: refs × scenes
    // -------------------------------------------------------------------------

    @Test @Order(1)
    @DisplayName("Analyse all scenes against all references")
    void analyseAllScenesAgainstAllReferences() {
        // In debug mode use the single debug ref; otherwise derive from filter or all
        ReferenceId[] refs;
        if (debugMode()) {
            refs = new ReferenceId[]{ debugRef() };
        } else if (referenceFilter().length > 0) {
            refs = referenceFilter();
        } else {
            // Collect the distinct refs actually present in the loaded catalogue
            refs = catalogue.stream()
                    .map(SceneEntry::primaryReferenceId)
                    .filter(Objects::nonNull)
                    .distinct()
                    .toArray(ReferenceId[]::new);
        }

        int           totalPairs  = refs.length * catalogue.size();
        AtomicInteger done        = new AtomicInteger(0);
        long          tStart      = System.currentTimeMillis();
        int           numThreads  = 8;

        progress = new ProgressDisplay(
                tag(), refs, totalPairs, catalogue.size(), numThreads, tStart);
        progress.start();

        ConcurrentLinkedQueue<AnalysisResult> bag      = new ConcurrentLinkedQueue<>();
        Map<AnalysisResult, SceneEntry>       sceneMap = new ConcurrentHashMap<>(totalPairs * 2);

        ForkJoinPool pool = new ForkJoinPool(numThreads);
        try {
            pool.submit(() ->
                Arrays.stream(refs).parallel().forEach(refId -> {
                    Mat refMat = ReferenceImageFactory.build(refId);
                    try {
                        for (SceneEntry scene : catalogue) {
                            List<AnalysisResult> matched =
                                    runMatcher(refId, refMat, scene, saveVariants(), absOutputDir);
                            for (AnalysisResult r : matched) {
                                bag.add(r);
                                sceneMap.put(r, scene);
                            }
                            int n = done.incrementAndGet();
                            progress.update(refId, n, bag.size());
                        }
                    } finally {
                        refMat.release();
                    }
                })
            ).get();
        } catch (Exception e) {
            throw new RuntimeException("Parallel matcher loop failed", e);
        } finally {
            pool.shutdown();
        }

        results        = new ArrayList<>(bag);
        resultSceneMap = sceneMap;
        progress.finish(done.get(), results.size());
    }

    // -------------------------------------------------------------------------
    // @AfterAll — output writing
    // -------------------------------------------------------------------------

    @AfterAll
    void writeOutput() {
        try {
            writeOutputInternal();
        } catch (Exception | Error e) {
            if (progress != null) progress.status("ERROR: " + e.getMessage());
            System.err.printf("[%s] writeOutput failed: %s%n", tag(), e);
            e.printStackTrace(System.err);
        }
    }

    private void writeOutputInternal() throws IOException {
        if (results == null || results.isEmpty()) {
            System.out.printf("[%s] No results — skipping output.%n", tag());
            return;
        }

        // 1. Compute verdicts
        progress.status("Computing verdicts...");
        Map<AnalysisResult, DetectionVerdict> verdicts = new LinkedHashMap<>(results.size() * 2);
        for (AnalysisResult r : results) {
            SceneEntry scene = resultSceneMap.get(r);
            if (scene != null) {
                verdicts.put(r, DetectionVerdict.evaluate(r, scene));
            }
        }

        // 2. Print ASCII table with verdict column (saved variants, first 40 rows)
        progress.status("Printing sample results table...");
        System.out.printf("%n=== %s — Sample Results (first 40 rows) ===%n", techniqueName());
        printAsciiTableWithVerdicts(
                results.stream()
                       .filter(r -> saveVariants().contains(r.methodName()))
                       .limit(40)
                       .toList(),
                verdicts);

        // 3. Print verdict summary
        printVerdictSummary(verdicts);

        // 4. No-op for annotated images (already written by matcher)
        progress.status("Saving annotated images...");
        AnalysisOutputWriter.saveAnnotatedImages(results, absOutputDir.resolve("annotated"));

        // 5. Reference grids
        progress.status("Saving reference grids...");
        AnalysisOutputWriter.saveReferenceGrids(
                results.stream().filter(r -> r.annotatedPath() != null).toList(),
                absOutputDir);

        // 6. Performance profiles
        progress.status("Profiling performance...");
        List<PerformanceProfile> profiles = PerformanceProfiler.profileAll(results);
        System.out.printf("%n=== %s — Performance ===%n", techniqueName());
        AnalysisOutputWriter.printPerformanceTable(profiles);

        // 7. HTML report
        progress.status("Writing HTML report...");
        Path reportPath = absOutputDir.resolve("report.html");
        HtmlReportWriter.write(results, profiles, techniqueName(), reportPath,
                verdicts, resultSceneMap);

        progress.status("Done. Report: " + reportPath);
        System.out.printf("[%s] Report: %s%n%n", tag(), reportPath);
    }

    // -------------------------------------------------------------------------
    // Console output helpers
    // -------------------------------------------------------------------------

    private static void printAsciiTableWithVerdicts(List<AnalysisResult> rows,
                                                     Map<AnalysisResult, DetectionVerdict> verdicts) {
        String fmt = "%-30s %-20s %-16s %-14s %10s %6s %6s%n";
        String sep = "-".repeat(112);
        System.out.println(sep);
        System.out.printf(fmt, "Method", "Reference", "Variant", "Category", "Score%", "Ms", "Verdict");
        System.out.println(sep);
        for (AnalysisResult r : rows) {
            DetectionVerdict v = verdicts.get(r);
            String verdict = v != null ? v.emoji() + " " + v.label() : "—";
            System.out.printf(fmt,
                    trunc(r.methodName(),    30),
                    r.referenceId() != null ? trunc(r.referenceId().name(), 20) : "(none)",
                    trunc(r.variantLabel(),  16),
                    r.category().name(),
                    r.isError() ? "ERR" : String.format("%.1f", r.matchScorePercent()),
                    r.elapsedMs(),
                    verdict);
        }
        System.out.println(sep);
        System.out.printf("Total: %d results%n%n", rows.size());
    }

    private void printVerdictSummary(Map<AnalysisResult, DetectionVerdict> verdicts) {
        long correct   = verdicts.values().stream().filter(v -> v == DetectionVerdict.CORRECT).count();
        long wrongLoc  = verdicts.values().stream().filter(v -> v == DetectionVerdict.WRONG_LOCATION).count();
        long missed    = verdicts.values().stream().filter(v -> v == DetectionVerdict.MISSED).count();
        long falseAlarm= verdicts.values().stream().filter(v -> v == DetectionVerdict.FALSE_ALARM).count();
        long rejected  = verdicts.values().stream().filter(v -> v == DetectionVerdict.CORRECTLY_REJECTED).count();
        long total     = verdicts.size();
        long tp = correct;
        long fp = wrongLoc + falseAlarm;
        long fn = missed;
        double precision = (tp + fp) > 0 ? (100.0 * tp / (tp + fp)) : 0;
        double recall    = (tp + fn) > 0 ? (100.0 * tp / (tp + fn)) : 0;
        double f1        = (precision + recall) > 0
                ? (2 * precision * recall / (precision + recall)) : 0;

        System.out.printf("%n[%s] === Detection Verdict Summary ===%n", tag());
        System.out.printf("  ✅ Correct: %5d   📍 Wrong Location: %5d   ❌ Missed: %5d   ⚠️  False Alarm: %5d   ✓ Correctly Rejected: %5d   Total: %d%n",
                correct, wrongLoc, missed, falseAlarm, rejected, total);
        System.out.printf("  Precision: %.1f%%   Recall: %.1f%%   F1: %.1f%%%n%n",
                precision, recall, f1);
    }

    private static String trunc(String s, int max) {
        return s == null ? "" : s.length() <= max ? s : s.substring(0, max - 1) + "~";
    }

    /**
     * Recursively deletes all contents of {@code dir} without deleting the
     * directory itself.  Symlinks are deleted (not followed).
     */
    private static void clearDirectory(Path dir) throws IOException {
        try (var stream = Files.walk(dir)) {
            // Reverse order so children are deleted before parents
            stream.sorted(java.util.Comparator.reverseOrder())
                  .filter(p -> !p.equals(dir))   // keep the root
                  .forEach(p -> {
                      try { Files.deleteIfExists(p); }
                      catch (IOException e) {
                          System.err.printf("[clearDir] Could not delete %s: %s%n", p, e.getMessage());
                      }
                  });
        }
    }
}
