package org.example.utilities;

import org.example.BenchmarkReportRunner;

import java.io.IOException;
import java.util.function.Consumer;

/**
 * Thin wrapper kept for backwards-compatibility.
 * All logic has been moved to {@link BenchmarkReportRunner} in main scope.
 */
public final class MatchingBenchmarkRunner {
    private MatchingBenchmarkRunner() {}

    public static void run(Consumer<String> log) throws IOException {
        BenchmarkReportRunner.run(log);
    }
}
