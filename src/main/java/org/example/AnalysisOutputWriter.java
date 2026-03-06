package org.example;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Writes analysis output: reference grids and ASCII tables to console.
 * Annotated images are now written directly by the matcher (see TemplateMatcher),
 * so saveAnnotatedImages is a no-op (paths are already on disk).
 * saveReferenceGrids loads images from disk paths stored in AnalysisResult.
 */
public final class AnalysisOutputWriter {

    private AnalysisOutputWriter() {}

    // -------------------------------------------------------------------------
    // Image output
    // -------------------------------------------------------------------------

    /**
     * No-op — annotated images are written directly by the matcher.
     * Kept for API compatibility; simply returns without doing anything.
     */
    public static void saveAnnotatedImages(List<AnalysisResult> results, Path outputDir) {
        // Images already written to disk by the matcher — nothing to do here.
    }

    /**
     * For each unique {@link ReferenceId}, creates a grid image showing scene results
     * colour-coded by score tier, loading thumbnails from the on-disk paths stored in
     * {@link AnalysisResult#annotatedPath()}.
     * Saved to {@code outputDir/grids/<referenceId>_grid.png}.
     */
    public static void saveReferenceGrids(List<AnalysisResult> results, Path outputDir)
            throws IOException {
        Path gridDir = outputDir.resolve("grids");
        Files.createDirectories(gridDir);

        Map<ReferenceId, List<AnalysisResult>> byRef = results.stream()
                .filter(r -> r.annotatedPath() != null)
                .collect(Collectors.groupingBy(AnalysisResult::referenceId));

        int total = byRef.size(), done = 0;
        System.out.printf("[OUTPUT] Building reference grids: %d refs with saved images%n", total);
        long t0 = System.currentTimeMillis();

        for (Map.Entry<ReferenceId, List<AnalysisResult>> entry : byRef.entrySet()) {
            List<AnalysisResult> group = entry.getValue();
            int thumbW = 120, thumbH = 90;
            int cols = Math.min(10, group.size());
            int rows = (int) Math.ceil((double) group.size() / cols);

            Mat grid = Mat.zeros(rows * (thumbH + 20), cols * thumbW, CvType.CV_8UC3);

            for (int i = 0; i < group.size(); i++) {
                AnalysisResult r = group.get(i);
                Mat loaded = Imgcodecs.imread(outputDir.resolve(r.annotatedPath()).toString());
                if (loaded.empty()) { loaded.release(); continue; }

                int col = i % cols, row = i / cols;
                Mat thumb = new Mat();
                Imgproc.resize(loaded, thumb, new Size(thumbW, thumbH));
                loaded.release();

                Scalar border = r.isError()                    ? new Scalar(128, 0, 128)
                              : r.matchScorePercent() >= 70    ? new Scalar(0, 200, 0)
                              : r.matchScorePercent() >= 40    ? new Scalar(0, 200, 200)
                              :                                  new Scalar(0, 0, 200);
                Imgproc.rectangle(thumb, new Point(0,0), new Point(thumbW-1,thumbH-1), border, 2);

                int dstX = col * thumbW, dstY = row * (thumbH + 20);
                thumb.copyTo(grid.submat(new Rect(dstX, dstY, thumbW, thumbH)));

                String label = r.isError() ? "ERR" : String.format("%.0f%%", r.matchScorePercent());
                Imgproc.putText(grid, label, new Point(dstX + 2, dstY + thumbH + 14),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.38, new Scalar(200, 200, 200), 1);
                thumb.release();
            }

            Imgcodecs.imwrite(gridDir.resolve(entry.getKey().name() + "_grid.png").toString(), grid);
            grid.release();
            done++;
            if (done % 10 == 0 || done == total) {
                System.out.printf("[OUTPUT] Grids: %d/%d (%.0f%%)  %.1fs elapsed%n",
                        done, total, (done * 100.0 / Math.max(1, total)),
                        (System.currentTimeMillis() - t0) / 1000.0);
            }
        }
        System.out.printf("[OUTPUT] Grids done: %d written in %.1fs%n",
                total, (System.currentTimeMillis() - t0) / 1000.0);
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
