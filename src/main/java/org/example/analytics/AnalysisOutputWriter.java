package org.example.analytics;

import java.nio.file.Path;
import java.util.*;


/**
 * Writes analysis output: ASCII tables to console.
 * Annotated images are written directly by the matcher (see TemplateMatcher),
 * so saveAnnotatedImages is a no-op (paths are already on disk).
 */
public final class AnalysisOutputWriter {

    private AnalysisOutputWriter() {}

    /**
     * No-op — annotated images are written directly by the matcher.
     * Kept for API compatibility; simply returns without doing anything.
     */
    public static void saveAnnotatedImages(List<AnalysisResult> results, Path outputDir) {
        // Images already written to disk by the matcher — nothing to do here.
    }


    // -------------------------------------------------------------------------
    // Console output
    // -------------------------------------------------------------------------

    /** Prints a compact ASCII results table to stdout. */
    public static void printAsciiTable(List<AnalysisResult> results) {
        String fmt = "%-30s %-20s %-16s %-14s %14s %8s%n";
        String sep = "-".repeat(108);
        System.out.println(sep);
        System.out.printf(fmt, "Method", "Reference", "Variant", "Category", "Match Score %", "Ms");
        System.out.println(sep);
        for (AnalysisResult r : results) {
            System.out.printf(fmt,
                    trunc(r.methodName(), 30),
                    trunc(r.referenceId().name(), 20),
                    trunc(r.variantLabel(), 16),
                    r.category().name(),
                    r.isError() ? "ERR" : String.format("%.1f", r.matchScorePercent()),
                    r.elapsedMs());
        }
        System.out.println(sep);
        System.out.printf("Total: %d results%n%n", results.size());
    }

    /** Prints a compact ASCII performance table to stdout. */
    public static void printPerformanceTable(List<PerformanceProfile> profiles) {
        String fmt = "%-32s %6s %6s %7s %6s %8s  %-36s%n";
        String sep = "-".repeat(110);
        System.out.println(sep);
        System.out.printf(fmt, "Variant", "Min", "Max", "Avg", "P95", "ms/MP", "4K proj (lin/quad)");
        System.out.println(sep);
        for (PerformanceProfile p : profiles) {
            double[] p4k = p.projectedMs().get("4K");
            String proj4k = p4k != null ? String.format("%.0f / %.0f ms", p4k[0], p4k[1]) : "n/a";
            System.out.printf(fmt,
                    trunc(p.methodVariant(), 32),
                    p.minMs(), p.maxMs(),
                    String.format("%.1f", p.avgMs()),
                    p.p95Ms(),
                    String.format("%.1f", p.msPerMp()),
                    proj4k);
        }
        System.out.println(sep);
        System.out.println();
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private static String sanitise(String s) {
        return s == null ? "unknown" : s.replaceAll("[^a-zA-Z0-9._-]", "_");
    }

    private static String trunc(String s, int max) {
        return s == null ? "" : s.length() <= max ? s : s.substring(0, max - 1) + "~";
    }
}
