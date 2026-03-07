package org.example.setups;

import org.example.*;
import org.junit.jupiter.api.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Milestone 6 — Colour Pre-Filter analytical test.
 * Iterates all 88 ReferenceId values, applies LOOSE and TIGHT masks to the
 * reference itself and to a black scene, saves 4-panel PNGs, prints a table.
 * No assertions.
 */
@DisplayName("Milestone 6 — Colour Pre-Filter")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class ColourPreFilterTest {

    private static final Path OUT = Paths.get("test_output", "colour_prefilter");
    private static Mat BLACK_SCENE;

    @BeforeAll
    static void setup() throws IOException {
        OpenCvLoader.load();
        Files.createDirectories(OUT.toAbsolutePath());
        BLACK_SCENE = Mat.zeros(ReferenceImageFactory.SIZE, ReferenceImageFactory.SIZE,
                CvType.CV_8UC3);
    }

    @AfterAll
    static void teardown() {
        if (BLACK_SCENE != null) BLACK_SCENE.release();
    }

    @Test @Order(1)
    @DisplayName("Generate 4-panel mask images for all 88 references")
    void generateMaskPanels() throws IOException {
        String hdr = "%-34s  %-8s  %-10s  %-10s  %-10s  %-10s  %s%n";
        String sep = "-".repeat(95);
        System.out.println("\n=== Colour Pre-Filter Results ===");
        System.out.println(sep);
        System.out.printf(hdr, "ReferenceId", "Colour", "LooseRef%", "TightRef%",
                "LooseBlk%", "TightBlk%", "WrapHue?");
        System.out.println(sep);

        int saved = 0;
        for (ReferenceId id : ReferenceId.values()) {
            Mat refMat = ReferenceImageFactory.build(id);
            String colourName = ReferenceImageFactory.foregroundColourName(id);

            ColourRange looseRange = ColourPreFilter.extractReferenceColourRange(id, ColourPreFilter.LOOSE);
            ColourRange tightRange = ColourPreFilter.extractReferenceColourRange(id, ColourPreFilter.TIGHT);

            Mat looseMaskRef = ColourPreFilter.apply(refMat,      looseRange);
            Mat tightMaskRef = ColourPreFilter.apply(refMat,      tightRange);
            Mat looseMaskBlk = ColourPreFilter.apply(BLACK_SCENE, looseRange);
            Mat tightMaskBlk = ColourPreFilter.apply(BLACK_SCENE, tightRange);

            double loosePctRef = ColourPreFilter.whitePixelFraction(looseMaskRef) * 100.0;
            double tightPctRef = ColourPreFilter.whitePixelFraction(tightMaskRef) * 100.0;
            double loosePctBlk = ColourPreFilter.whitePixelFraction(looseMaskBlk) * 100.0;
            double tightPctBlk = ColourPreFilter.whitePixelFraction(tightMaskBlk) * 100.0;

            System.out.printf(hdr,
                    id.name(), colourName,
                    String.format("%.1f%%", loosePctRef),
                    String.format("%.1f%%", tightPctRef),
                    String.format("%.1f%%", loosePctBlk),
                    String.format("%.1f%%", tightPctBlk),
                    looseRange.wrapsHue() ? "YES" : "no");

            Mat panel = buildPanel(refMat, looseMaskRef, tightMaskRef, looseMaskBlk,
                    id, colourName, loosePctRef, tightPctRef, loosePctBlk, tightPctBlk);
            Imgcodecs.imwrite(OUT.toAbsolutePath().resolve(id.name() + ".png").toString(), panel);
            panel.release();
            saved++;

            refMat.release();
            looseMaskRef.release(); tightMaskRef.release();
            looseMaskBlk.release(); tightMaskBlk.release();
        }
        System.out.println(sep);
        System.out.printf("%nSaved %d panel images to %s%n%n", saved, OUT.toAbsolutePath());
    }

    @Test @Order(2)
    @DisplayName("Spot-check: red wrap-around and white low-saturation")
    void spotCheckSpecialCases() {
        for (int slot : new int[]{0, 1, 7}) {
            ReferenceId id = findByFgSlot(slot);
            if (id == null) continue;
            String label = slot == 0 ? "white (low-sat)" : slot == 1 ? "red (wrap)" : "orange (near-red)";
            ColourRange loose = ColourPreFilter.extractReferenceColourRange(id, ColourPreFilter.LOOSE);
            ColourRange tight = ColourPreFilter.extractReferenceColourRange(id, ColourPreFilter.TIGHT);
            System.out.printf("%s — %s%n  LOOSE: %s%n  TIGHT: %s%n%n", label, id.name(), loose, tight);
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private static Mat buildPanel(Mat refMat,
                                   Mat looseMaskRef, Mat tightMaskRef, Mat looseMaskBlk,
                                   ReferenceId id, String colour,
                                   double loosePctRef, double tightPctRef,
                                   double loosePctBlk, double tightPctBlk) {
        int W = ReferenceImageFactory.SIZE;
        int H = ReferenceImageFactory.SIZE;
        int headerH = 20;
        int totalW = W * 4;
        int totalH = H + headerH;

        Mat out = new Mat(totalH, totalW, CvType.CV_8UC3, new Scalar(30, 30, 30));

        Mat lRef3 = toColor(looseMaskRef);
        Mat tRef3 = toColor(tightMaskRef);
        Mat lBlk3 = toColor(looseMaskBlk);

        copyInto(out, refMat,  0, headerH);
        copyInto(out, lRef3,   W, headerH);
        copyInto(out, tRef3, 2*W, headerH);
        copyInto(out, lBlk3, 3*W, headerH);

        lRef3.release(); tRef3.release(); lBlk3.release();

        Scalar wh = new Scalar(220, 220, 220);
        Scalar yl = new Scalar(0, 220, 220);
        Scalar rd = new Scalar(0, 0, 200);
        double fnt = 0.32;

        Imgproc.putText(out, id.name() + " (" + colour + ")",    new Point(2,     13), Imgproc.FONT_HERSHEY_SIMPLEX, fnt, wh, 1);
        Imgproc.putText(out, String.format("LOOSE %.1f%%", loosePctRef), new Point(W+2,   13), Imgproc.FONT_HERSHEY_SIMPLEX, fnt, yl, 1);
        Imgproc.putText(out, String.format("TIGHT %.1f%%", tightPctRef), new Point(2*W+2, 13), Imgproc.FONT_HERSHEY_SIMPLEX, fnt, yl, 1);
        Imgproc.putText(out, String.format("BLACK %.1f%%", loosePctBlk), new Point(3*W+2, 13), Imgproc.FONT_HERSHEY_SIMPLEX, fnt,
                loosePctBlk > 0.01 ? rd : yl, 1);

        for (int p = 1; p <= 3; p++) {
            Imgproc.line(out, new Point(p*W, 0), new Point(p*W, totalH), new Scalar(80,80,80), 1);
        }
        return out;
    }

    private static Mat toColor(Mat mask) {
        Mat out = new Mat();
        Imgproc.cvtColor(mask, out, Imgproc.COLOR_GRAY2BGR);
        return out;
    }

    private static void copyInto(Mat dst, Mat src, int x, int y) {
        Mat roi = dst.submat(new Rect(x, y, src.cols(), src.rows()));
        src.copyTo(roi);
        roi.release();
    }

    private static ReferenceId findByFgSlot(int slot) {
        for (ReferenceId id : ReferenceId.values()) {
            if (id.ordinal() % 8 == slot) return id;
        }
        return null;
    }
}

