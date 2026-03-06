package org.example;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Milestone 1 — Smoke Test.
 *
 * Verifies that:
 *   1. The OpenCV native library loads without error.
 *   2. A basic {@link Mat} can be created and its dimensions read.
 *   3. The {@code test_output/} directory exists (or is created).
 *
 * This test never asserts pass/fail conditions beyond the implicit
 * "no exception thrown" guarantee. It is purely a sanity check.
 */
@DisplayName("Milestone 1 — OpenCV Smoke Test")
class SmokeTest {

    @BeforeAll
    static void loadOpenCv() {
        OpenCvLoader.load();
    }

    @Test
    @DisplayName("Native library loads and Mat can be constructed")
    void nativeLibraryLoadsAndMatCreates() throws IOException {
        // Create a small black Mat and verify its dimensions
        Mat mat = Mat.zeros(10, 10, CvType.CV_8UC3);

        int rows = mat.rows();
        int cols = mat.cols();
        int channels = mat.channels();

        System.out.printf("[SmokeTest] Mat: %dx%d, channels=%d, type=CV_8UC3%n",
                rows, cols, channels);

        // Ensure test_output directory exists
        Path testOutput = Paths.get("test_output");
        Files.createDirectories(testOutput);
        System.out.printf("[SmokeTest] test_output directory ready: %s%n",
                testOutput.toAbsolutePath());

        mat.release();

        System.out.println("[SmokeTest] ✅ Smoke test passed — OpenCV is operational.");
    }
}

