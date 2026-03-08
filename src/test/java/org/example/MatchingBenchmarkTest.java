package org.example;

import org.junit.jupiter.api.*;

import java.io.IOException;
import java.nio.file.*;

/**
 * Milestone 20 — Unified Benchmark Report.
 *
 * <p>Delegates all collation logic to {@link BenchmarkReportRunner} which
 * lives in main scope and is therefore also available to the UI launcher
 * without requiring the test classpath.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
@DisplayName("Milestone 20 — Unified Benchmark Report (collator)")
public class MatchingBenchmarkTest {

    @BeforeAll
    public void setup() throws IOException {
        Files.createDirectories(
                Paths.get("test_output", "benchmark").toAbsolutePath().normalize());
    }

    @Test @Order(1)
    @DisplayName("Collate and analyse per-technique reports into unified benchmark HTML")
    public void collateBenchmarkReport() throws IOException {
        BenchmarkReportRunner.run(msg -> System.out.println("[BENCHMARK] " + msg));
    }
}
