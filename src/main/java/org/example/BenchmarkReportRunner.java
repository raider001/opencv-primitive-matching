package org.example;

import java.io.IOException;
import java.lang.reflect.Method;
import java.nio.file.*;
import java.util.function.Consumer;

/**
 * Programmatic entry-point for generating the unified benchmark report.
 *
 * <p>Delegates to {@code org.example.utilities.MatchingBenchmarkRunner} via
 * reflection so that the main-scope UI can trigger the benchmark collation
 * without a hard compile-time dependency on test-scope classes.
 *
 * <p>When invoked from a context where the test classpath is present (e.g.
 * {@code mvn exec:java}, IDE run, or a fat-jar that includes test classes)
 * the collation runs normally.  If the test classes are not available an
 * {@link IOException} is thrown explaining the situation.
 */
public final class BenchmarkReportRunner {

    private BenchmarkReportRunner() {}

    /**
     * Collates all available per-technique {@code report.html} files and
     * writes the unified benchmark report to
     * {@code test_output/benchmark/report.html}.
     *
     * @param log callback for progress messages (called on the caller's thread)
     * @throws IOException if the test-scope runner cannot be located or the
     *                     collation itself fails
     */
    public static void run(Consumer<String> log) throws IOException {
        try {
            Class<?> cls = Class.forName("org.example.utilities.MatchingBenchmarkRunner");
            Method m = cls.getMethod("run", Consumer.class);
            m.invoke(null, log);
        } catch (ClassNotFoundException cnf) {
            throw new IOException(
                "Benchmark collation requires the test classpath to be present.\n" +
                "Run via 'mvn test' or from an IDE that includes test sources.", cnf);
        } catch (Exception ex) {
            Throwable cause = ex.getCause() != null ? ex.getCause() : ex;
            throw new IOException("Benchmark collation failed: " + cause.getMessage(), cause);
        }
    }
}


