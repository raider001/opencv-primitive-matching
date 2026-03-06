package org.example;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Milestone 2 — Reference Image Library analytical test.
 *
 * Iterates every {@link ReferenceId}, builds the reference image via
 * {@link ReferenceImageFactory}, saves it to {@code test_output/references/},
 * and prints a summary table. No assertions — always passes.
 */
@DisplayName("Milestone 2 — Reference Image Library")
class ReferenceImageLibraryTest {

    private static final Path OUTPUT_DIR = Paths.get("test_output", "references");
    private static final List<String[]> TABLE_ROWS = new ArrayList<>();

    @BeforeAll
    static void setup() throws IOException {
        OpenCvLoader.load();
        Files.createDirectories(OUTPUT_DIR);
    }

    @Test
    @DisplayName("Generate all reference images and save to test_output/references/")
    void generateAllReferenceImages() {
        ReferenceId[] ids = ReferenceId.values();
        System.out.printf("%nGenerating %d reference images...%n%n", ids.length);

        for (ReferenceId id : ids) {
            Mat image = ReferenceImageFactory.build(id);

            Path outPath = OUTPUT_DIR.toAbsolutePath().resolve(id.name() + ".png");
            MatOfByte buf = new MatOfByte();
            boolean encoded = Imgcodecs.imencode(".png", image, buf);
            boolean saved = false;
            if (encoded && buf.total() > 0) {
                try {
                    Files.write(outPath, buf.toArray());
                    saved = true;
                } catch (IOException e) {
                    System.err.println("Failed to write " + outPath + ": " + e.getMessage());
                }
            }
            buf.release();

            // Count non-zero pixels via greyscale conversion
            Mat grey = new Mat();
            Imgproc.cvtColor(image, grey, Imgproc.COLOR_BGR2GRAY);
            int nonZero = Core.countNonZero(grey);
            grey.release();

            TABLE_ROWS.add(new String[]{
                id.name(),
                ReferenceImageFactory.foregroundColourName(id),
                ReferenceImageFactory.backgroundFillName(id),
                String.valueOf(image.cols()),
                String.valueOf(image.rows()),
                String.valueOf(nonZero),
                saved ? "OK" : "FAILED"
            });

            image.release();
        }
    }

    @AfterAll
    static void printSummaryTable() {
        String fmt  = "%-32s %-10s %-22s %5s %5s %8s  %s%n";
        String sep  = "-".repeat(95);

        System.out.println();
        System.out.println(sep);
        System.out.printf(fmt, "ID", "FG Colour", "Background", "W", "H", "Non-zero", "Saved");
        System.out.println(sep);
        for (String[] row : TABLE_ROWS) {
            System.out.printf(fmt, (Object[]) row);
        }
        System.out.println(sep);
        System.out.printf("Total: %d images written to %s%n",
                TABLE_ROWS.size(), OUTPUT_DIR.toAbsolutePath());

        long failed = TABLE_ROWS.stream().filter(r -> r[6].equals("FAILED")).count();
        if (failed > 0) {
            System.out.printf("WARNING: %d image(s) failed to save!%n", failed);
        } else {
            System.out.println("All reference images saved successfully.");
        }
        System.out.println();
    }
}






