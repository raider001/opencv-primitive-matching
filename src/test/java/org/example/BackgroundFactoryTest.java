package org.example;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDouble;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Milestone 3 — Background Factory analytical test.
 *
 * Iterates every {@link BackgroundId}, builds the background via {@link BackgroundFactory},
 * saves it to {@code test_output/backgrounds/}, and prints a summary table showing
 * ID, tier, complexity label, and mean pixel intensity per channel (B/G/R).
 * No assertions — always passes.
 */
@DisplayName("Milestone 3 — Background Factory")
class BackgroundFactoryTest {

    private static final int W = 640;
    private static final int H = 480;
    private static final Path OUTPUT_DIR = Paths.get("test_output", "backgrounds");
    private static final List<String[]> TABLE_ROWS = new ArrayList<>();

    @BeforeAll
    static void setup() throws IOException {
        OpenCvLoader.load();
        Files.createDirectories(OUTPUT_DIR.toAbsolutePath());
    }

    @Test
    @DisplayName("Generate all 22 backgrounds and save to test_output/backgrounds/")
    void generateAllBackgrounds() {
        BackgroundId[] ids = BackgroundId.values();
        System.out.printf("%nGenerating %d backgrounds at %dx%d...%n%n", ids.length, W, H);

        for (BackgroundId id : ids) {
            Mat bg = null;
            String saveNote = "OK";
            String meanB = "0.0", meanG = "0.0", meanR = "0.0";
            try {
                bg = BackgroundFactory.build(id, W, H);

                if (bg == null || bg.empty()) {
                    saveNote = "EMPTY";
                } else {
                    String absPath = OUTPUT_DIR.toAbsolutePath()
                                               .resolve(id.name() + ".png")
                                               .toString();
                    Mat contiguous = bg.clone();
                    boolean saved = Imgcodecs.imwrite(absPath, contiguous);
                    contiguous.release();
                    saveNote = saved ? "OK" : "FAILED";

                    MatOfDouble mean   = new MatOfDouble();
                    MatOfDouble stddev = new MatOfDouble();
                    Core.meanStdDev(bg, mean, stddev);
                    double[] m = mean.toArray();
                    meanB = String.format("%.1f", m.length > 0 ? m[0] : 0);
                    meanG = String.format("%.1f", m.length > 1 ? m[1] : 0);
                    meanR = String.format("%.1f", m.length > 2 ? m[2] : 0);
                    mean.release();
                    stddev.release();
                }
            } catch (Exception e) {
                saveNote = "EX:" + e.getMessage();
            } finally {
                if (bg != null) bg.release();
            }

            TABLE_ROWS.add(new String[]{
                id.name(),
                String.valueOf(id.tier()),
                id.complexityLabel(),
                meanB, meanG, meanR,
                saveNote
            });
        }
    }

    @AfterAll
    static void printSummaryTable() {
        String fmt = "%-28s  %4s  %-8s  %6s  %6s  %6s  %s%n";
        String sep = "-".repeat(74);

        System.out.println();
        System.out.println(sep);
        System.out.printf(fmt, "ID", "Tier", "Complexity", "Mean B", "Mean G", "Mean R", "Saved");
        System.out.println(sep);
        for (String[] row : TABLE_ROWS) {
            System.out.printf(fmt, (Object[]) row);
        }
        System.out.println(sep);
        System.out.printf("Total: %d backgrounds written to %s%n",
                TABLE_ROWS.size(), OUTPUT_DIR.toAbsolutePath());

        long failed = TABLE_ROWS.stream().filter(r -> !r[6].equals("OK")).count();
        if (failed > 0) {
            System.out.printf("WARNING: %d background(s) had issues:%n", failed);
            TABLE_ROWS.stream()
                      .filter(r -> !r[6].equals("OK"))
                      .forEach(r -> System.out.printf("  %-28s -> %s%n", r[0], r[6]));
        } else {
            System.out.println("All backgrounds saved successfully.");
        }
        System.out.println();
    }
}






