package org.example.utilities;

import org.example.*;

import java.io.IOException;
import java.nio.file.*;
import java.util.function.Consumer;

/**
 * Programmatic entry-point for generating the unified benchmark report.
 *
 * <p>Mirrors what {@link MatchingBenchmarkTest} does in its JUnit test, but
 * callable directly from the {@link BenchmarkLauncher} UI without needing
 * the JUnit test runner.
 */
public final class MatchingBenchmarkRunner {

    private MatchingBenchmarkRunner() {}

    /**
     * Collates all available per-technique {@code report.html} files and
     * writes the unified benchmark report to
     * {@code test_output/benchmark/report.html}.
     *
     * @param log callback for progress messages (called on the caller's thread)
     */
    public static void run(Consumer<String> log) throws IOException {
        Path benchmarkDir = Paths.get("test_output", "benchmark").toAbsolutePath().normalize();
        Files.createDirectories(benchmarkDir);

        // Re-use MatchingBenchmarkTest's collation logic by instantiating it
        // and calling its test method directly (no JUnit runner needed).
        MatchingBenchmarkTest test = new MatchingBenchmarkTest();
        try {
            test.setup();
            log.accept("Collating technique reports…");
            test.collateBenchmarkReport();
            log.accept("Written: " + benchmarkDir.resolve("report.html"));
        } catch (Exception e) {
            throw new IOException("Benchmark collation failed: " + e.getMessage(), e);
        }
    }
}


