package org.example;

import org.junit.jupiter.api.*;
import org.opencv.core.Mat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Shared base for all analytical technique tests (Milestones 7–15).
 *
 * <p>Subclasses declare technique-specific configuration via abstract methods and
 * implement {@link #runMatcher} to call their specific matcher.  Everything else —
 * catalogue construction, the parallel double loop, progress logging, output writing,
 * and HTML report generation — lives here exactly once.
 *
 * <h2>Usage in a subclass</h2>
 * <pre>{@code
 * class TemplateMatchingTest extends AnalyticalTestBase {
 *     \@Override protected String tag()           { return "TM"; }
 *     \@Override protected String techniqueName() { return "Template Matching"; }
 *     \@Override protected Path  outputDir()      { return Paths.get("test_output","template_matching"); }
 *     \@Override protected boolean debugMode()    { return true; }
 *     \@Override protected ReferenceId debugRef() { return ReferenceId.CIRCLE_OUTLINE; }
 *     \@Override protected Set<String> saveVariants() { return Set.of("TM_CCOEFF_NORMED", ...); }
 *
 *     \@Override
 *     protected List<AnalysisResult> runMatcher(ReferenceId refId, Mat refMat,
 *                                               SceneEntry scene, Set<String> saveVariants,
 *                                               Path outputDir) {
 *         return TemplateMatcher.match(refId, refMat, scene, saveVariants, outputDir);
 *     }
 * }
 * }</pre>
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
abstract class AnalyticalTestBase {

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
    // Shared state — populated by @BeforeAll / @Test, consumed by @AfterAll
    // -------------------------------------------------------------------------

    private List<SceneEntry>     catalogue;
    private List<AnalysisResult> results;
    /** Absolute, canonical output directory — resolved once in setup(). */
    private Path absOutputDir;

    // -------------------------------------------------------------------------
    // @BeforeAll — catalogue construction
    // -------------------------------------------------------------------------

    @BeforeAll
    void setup() throws IOException {
        OpenCvLoader.load();
        // Resolve outputDir to absolute once so all subsequent calls are consistent
        absOutputDir = outputDir().toAbsolutePath().normalize();
        Files.createDirectories(absOutputDir);

        if (debugMode()) {
            System.out.printf("%n[%s] *** DEBUG MODE — ref: %s, building 3-scene debug catalogue ***%n",
                    tag(), debugRef());
            catalogue = SceneCatalogue.buildDebug(debugRef());
        } else {
            System.out.printf("%n[%s] Building slim scene catalogue...%n", tag());
            long t0 = System.currentTimeMillis();
            catalogue = SceneCatalogue.buildSlim();
            System.out.printf("[%s] Slim catalogue: %d scenes  (A=%d B=%d C=%d D=%d)  in %d ms%n%n",
                    tag(), catalogue.size(),
                    catalogue.stream().filter(s -> s.category() == SceneCategory.A_CLEAN).count(),
                    catalogue.stream().filter(s -> s.category() == SceneCategory.B_TRANSFORMED).count(),
                    catalogue.stream().filter(s -> s.category() == SceneCategory.C_DEGRADED).count(),
                    catalogue.stream().filter(s -> s.category() == SceneCategory.D_NEGATIVE).count(),
                    System.currentTimeMillis() - t0);
        }
    }

    // -------------------------------------------------------------------------
    // @Test — parallel double loop: refs × scenes
    // -------------------------------------------------------------------------

    @Test @Order(1)
    @DisplayName("Analyse all scenes against all references")
    void analyseAllScenesAgainstAllReferences() {
        ReferenceId[] refs = debugMode()
                ? new ReferenceId[]{ debugRef() }
                : ReferenceId.values();

        int           totalPairs  = refs.length * catalogue.size();
        AtomicInteger done        = new AtomicInteger(0);
        int           reportEvery = Math.max(1, totalPairs / 40); // ~2.5% steps
        long          tStart      = System.currentTimeMillis();

        System.out.printf("[%s] Starting: %d refs × %d scenes  |  threads: %d%n%n",
                tag(), refs.length, catalogue.size(),
                Runtime.getRuntime().availableProcessors());

        ConcurrentLinkedQueue<AnalysisResult> bag = new ConcurrentLinkedQueue<>();

        // Parallelise over refs; inner scene loop is sequential so refMat is
        // built once and shared across all scenes for that reference.
        Arrays.stream(refs).parallel().forEach(refId -> {
            Mat refMat = ReferenceImageFactory.build(refId);
            try {
                for (SceneEntry scene : catalogue) {
                    bag.addAll(runMatcher(refId, refMat, scene, saveVariants(), absOutputDir));

                    int n = done.incrementAndGet();
                    if (n % reportEvery == 0 || n == totalPairs) {
                        long   elapsed = System.currentTimeMillis() - tStart;
                        double pct     = (n * 100.0) / totalPairs;
                        long   eta     = elapsed > 0
                                ? (long) ((elapsed / pct) * (100.0 - pct) / 1000) : 0;
                        System.out.printf("[%s] %5.1f%%  %,d/%,d pairs  elapsed %ds  ETA ~%ds%n",
                                tag(), pct, n, totalPairs, elapsed / 1000, eta);
                    }
                }
            } finally {
                refMat.release();
            }
        });

        results = new ArrayList<>(bag);
        System.out.printf("%n[%s] Complete: %,d results in %.1f s%n%n",
                tag(), results.size(), (System.currentTimeMillis() - tStart) / 1000.0);
    }

    // -------------------------------------------------------------------------
    // @AfterAll — output writing
    // -------------------------------------------------------------------------

    @AfterAll
    void writeOutput() {
        try {
            writeOutputInternal();
        } catch (Exception | Error e) {
            System.err.printf("[%s] writeOutput failed: %s%n", tag(), e);
            e.printStackTrace(System.err);
        }
    }

    private void writeOutputInternal() throws IOException {
        if (results == null || results.isEmpty()) {
            System.out.printf("[%s] No results — skipping output.%n", tag());
            return;
        }

        System.out.printf("[%s] Writing output...%n", tag());

        // 1. ASCII sample table (saved variants only, first 40 rows)
        System.out.printf("%n=== %s — Sample Results (first 40 rows) ===%n", techniqueName());
        AnalysisOutputWriter.printAsciiTable(
                results.stream()
                       .filter(r -> saveVariants().contains(r.methodName()))
                       .limit(40)
                       .toList());

        // 2. saveAnnotatedImages is a no-op — images already on disk
        AnalysisOutputWriter.saveAnnotatedImages(results, absOutputDir.resolve("annotated"));

        // 3. Reference grids loaded from disk paths
        System.out.printf("[%s] Saving reference grids...%n", tag());
        AnalysisOutputWriter.saveReferenceGrids(
                results.stream().filter(r -> r.annotatedPath() != null).toList(),
                absOutputDir);

        // 4. Performance profiles
        List<PerformanceProfile> profiles = PerformanceProfiler.profileAll(results);

        System.out.printf("%n=== %s — Performance ===%n", techniqueName());
        AnalysisOutputWriter.printPerformanceTable(profiles);

        // 5. HTML report
        System.out.printf("[%s] Writing HTML report...%n", tag());
        Path reportPath = absOutputDir.resolve("report.html");
        HtmlReportWriter.write(results, profiles, techniqueName(), reportPath);
        System.out.printf("[%s] Report: %s%n%n", tag(), reportPath);
    }
}

