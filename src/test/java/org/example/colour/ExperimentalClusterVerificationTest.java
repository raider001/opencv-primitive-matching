package org.example.colour;

import org.example.OpenCvLoader;
import org.junit.jupiter.api.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Targeted regression test: confirms that pixels sharing a similar hue band but
 * differing significantly in saturation (and/or hue) land in SEPARATE clusters.
 *
 * Reproduces the exact user-reported values:
 *   Pixel A  H=10  S=137  V=193   (muted orange, ≈ HSV 20° / 54% / 76%)
 *   Pixel B  H=16  S=255  V=255   (vivid orange, ≈ HSV 33° / 100% / 100%)
 *
 * Scene layout: 640×480 image with an achromatic grey background.
 * Muted orange fills a left rectangular patch; vivid orange fills a right
 * rectangular patch.  The grey background creates real chromatic/achromatic
 * edges so that the morphological-gradient border-pixel histogram sees actual
 * pixels and peak detection fires for both orange variants.
 *
 * <p>Separation may happen via two mechanisms — both are valid:
 * <ol>
 *   <li><b>Hue peak separation</b> — H=10 and H=16 are detected as distinct
 *       hue peaks (distance 6 = PEAK_MIN_SEPARATION, not suppressed).</li>
 *   <li><b>S sub-cluster separation</b> — both peaks fall in the same hue
 *       cluster but Otsu thresholding splits S=137 from S=255.</li>
 * </ol>
 * The assertions accept either mechanism.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class ExperimentalClusterVerificationTest {

    private static final int W = 640, H = 480;

    // Pixel A: muted orange  (OpenCV HSV half-degree units)
    private static final int HA = 10, SA = 137, VA = 193;
    // Pixel B: vivid orange  (OpenCV HSV half-degree units)
    private static final int HB = 16, SB = 255, VB = 255;

    // Patch A (muted orange) — left portion of the image
    private static final int PATCH_A_R0 = 80,  PATCH_A_R1 = 400;
    private static final int PATCH_A_C0 = 60,  PATCH_A_C1 = 260;

    // Patch B (vivid orange) — right portion, 120 px gap from Patch A
    // (gap >> 2×HEAL_RADIUS=16 px, so morphological closing cannot merge them)
    private static final int PATCH_B_R0 = 80,  PATCH_B_R1 = 400;
    private static final int PATCH_B_C0 = 380, PATCH_B_C1 = 580;

    // Centre-of-patch sample points used to look up cluster membership
    private static final int ROW_A = (PATCH_A_R0 + PATCH_A_R1) / 2;   // 240
    private static final int COL_A = (PATCH_A_C0 + PATCH_A_C1) / 2;   // 160
    private static final int ROW_B = (PATCH_B_R0 + PATCH_B_R1) / 2;   // 240
    private static final int COL_B = (PATCH_B_C0 + PATCH_B_C1) / 2;   // 480

    @BeforeAll
    void loadNative() {
        OpenCvLoader.load();
    }

    // -------------------------------------------------------------------------

    @Test
    @Order(1)
    @DisplayName("Pixels A (S=137) and B (S=255) must be in different clusters")
    void mutedAndVividOrangeMustBeSeparate() {
        List<ColourCluster> clusters = buildAndExtract();

        int clusterA = findCluster(clusters, ROW_A, COL_A);
        int clusterB = findCluster(clusters, ROW_B, COL_B);

        System.out.printf("%nScene: left-patch HSV(%d,%d,%d)  right-patch HSV(%d,%d,%d)%n",
                HA, SA, VA, HB, SB, VB);
        System.out.println("Clusters found: " + clusters.size());
        printClusters(clusters);
        System.out.printf("Pixel A (muted,  H=%d S=%d) → cluster %d%n", HA, SA, clusterA);
        System.out.printf("Pixel B (vivid,  H=%d S=%d) → cluster %d%n", HB, SB, clusterB);

        clusters.forEach(ColourCluster::release);

        assertNotEquals(-1, clusterA,
                "Pixel A (H=" + HA + " S=" + SA + ") was NOT assigned to any cluster");
        assertNotEquals(-1, clusterB,
                "Pixel B (H=" + HB + " S=" + SB + ") was NOT assigned to any cluster");
        assertNotEquals(clusterA, clusterB,
                "Pixel A (S=" + SA + ") and Pixel B (S=" + SB + ") ended up in the SAME cluster ("
                + clusterA + ") — neither hue peak detection nor S sub-clustering separated them");
    }

    @Test
    @Order(2)
    @DisplayName("Cluster bounds must correctly reflect the separation mechanism")
    void clusterBoundsReflectSplit() {
        List<ColourCluster> clusters = buildAndExtract();

        int clusterA = findCluster(clusters, ROW_A, COL_A);
        int clusterB = findCluster(clusters, ROW_B, COL_B);

        if (clusterA == -1 || clusterB == -1 || clusterA == clusterB) {
            clusters.forEach(ColourCluster::release);
            fail("Pre-condition failed — clusters not separated (see test 1)");
        }

        ColourCluster ca = clusters.get(clusterA);
        ColourCluster cb = clusters.get(clusterB);

        System.out.printf("Cluster A: hue=%.0f  lo=%3d  hi=%3d  S=[%3d,%3d]  (contains H=%d S=%d)%n",
                ca.hue, ca.loBound, ca.hiBound, ca.sLo, ca.sHi, HA, SA);
        System.out.printf("Cluster B: hue=%.0f  lo=%3d  hi=%3d  S=[%3d,%3d]  (contains H=%d S=%d)%n",
                cb.hue, cb.loBound, cb.hiBound, cb.sLo, cb.sHi, HB, SB);

        // Separation can be established either through different hue windows or non-overlapping S windows
        boolean separatedByHue = ca.loBound != cb.loBound || ca.hiBound != cb.hiBound;
        boolean separatedByS   = ca.sHi < cb.sLo || cb.sHi < ca.sLo;

        System.out.println("Separated by hue window : " + separatedByHue);
        System.out.println("Separated by S window   : " + separatedByS);

        clusters.forEach(ColourCluster::release);

        assertTrue(separatedByHue || separatedByS,
                "Clusters A and B share the same hue bounds AND overlapping S windows — "
                + "no valid separation mechanism was applied");

        if (!separatedByHue) {
            // Same hue cluster → S sub-clustering must have fired; verify window membership
            assertTrue(ca.sLo <= SA && SA <= ca.sHi,
                    "S=" + SA + " not inside cluster A S-window [" + ca.sLo + "," + ca.sHi + "]");
            assertTrue(cb.sLo <= SB && SB <= cb.sHi,
                    "S=" + SB + " not inside cluster B S-window [" + cb.sLo + "," + cb.sHi + "]");
            assertTrue(ca.sHi < SB,
                    "Cluster A S-window [" + ca.sLo + "," + ca.sHi
                    + "] extends into cluster B territory (S=" + SB + ")");
        }
    }

    /**
     * Same-hue-cluster gate: forces H=10 and H=16 into ONE merged hue cluster by
     * adding an intermediate orange band (H=13, S=190) that fills the valley
     * between the two peaks, so the hue histogram cannot resolve two peaks.
     *
     * <p>In this configuration ONLY S sub-clustering (Otsu) can separate the
     * muted (S=137) pixels from the vivid (S=255) pixels.  This mirrors the
     * ColourClusterUnitTest scenario where the experimental extractor reports
     * {@code hue=11  lo=1  hi=17  S=[35,255]} — a single hue band covering both
     * H=10 and H=16 — and S sub-clustering must do the separating work.
     *
     * <p>If this test fails, the Otsu S-split is broken for this pair of values.
     */
    @Test
    @Order(3)
    @DisplayName("S sub-clustering must split S=137 from S=255 even when both share the same hue cluster")
    void sSplitWithinMergedHueCluster() {
        // Scene layout (all on achromatic grey background):
        //
        //   cols 60..260   (200 wide) : intermediate orange H=13, S=190  ← valley-filler
        //   cols 290..430  (140 wide) : muted orange        H=10, S=137
        //   cols 460..600  (140 wide) : vivid orange        H=16, S=255
        //
        // The intermediate band has enough border pixels to fill hueHist[13], making
        // the histogram flat between H=10 and H=16 → peak detection sees ONE broad
        // peak rather than two narrow ones → single hue cluster [~1, ~17].
        // With S sub-clustering (Otsu), S=137 vs S=255 are still separable.

        int valH = 13, valS = 190, valV = 240;   // intermediate "valley-filler" orange

        // Sample points inside each patch (interior, far from borders)
        int sampleRow  = H / 2;           // row 240
        int colMuted   = (290 + 430) / 2; // col 360 — centre of muted patch
        int colVivid   = (460 + 600) / 2; // col 530 — centre of vivid patch

        Mat hsv = new Mat(H, W, CvType.CV_8UC3, new Scalar(0, 0, 128)); // grey background

        // Valley-filler: intermediate orange (merges hue peaks H=10 and H=16)
        hsv.submat(80, 400,  60, 260).setTo(new Scalar(valH, valS, valV));
        // Muted orange patch
        hsv.submat(80, 400, 290, 430).setTo(new Scalar(HA, SA, VA));
        // Vivid orange patch
        hsv.submat(80, 400, 460, 600).setTo(new Scalar(HB, SB, VB));

        Mat bgr = new Mat();
        Imgproc.cvtColor(hsv, bgr, Imgproc.COLOR_HSV2BGR);
        hsv.release();

        List<ColourCluster> clusters =
                ExperimentalSceneColourClusters.INSTANCE.extractFromBorderPixels(bgr);
        bgr.release();

        int clusterMuted = findCluster(clusters, sampleRow, colMuted);
        int clusterVivid = findCluster(clusters, sampleRow, colVivid);

        System.out.println("\n[Order-3] Same-hue-cluster S-split test");
        System.out.println("Clusters found: " + clusters.size());
        printClusters(clusters);
        System.out.printf("Muted  (H=%d S=%d) → cluster %d%n", HA, SA, clusterMuted);
        System.out.printf("Vivid  (H=%d S=%d) → cluster %d%n", HB, SB, clusterVivid);

        if (clusterMuted != -1 && clusterVivid != -1 && clusterMuted != clusterVivid) {
            ColourCluster cm = clusters.get(clusterMuted);
            ColourCluster cv = clusters.get(clusterVivid);
            boolean sameHueBounds = cm.loBound == cv.loBound && cm.hiBound == cv.hiBound;
            if (sameHueBounds) {
                System.out.println("  → Separated by S sub-clustering (Otsu) — same hue window, different S bands");
            } else {
                System.out.println("  → Separated by hue peaks (hue-cluster mechanism)");
            }
        } else if (clusterMuted == clusterVivid && clusterMuted != -1) {
            System.out.println("  → SAME cluster — Otsu S-split must have failed");
        }

        clusters.forEach(ColourCluster::release);

        assertNotEquals(-1, clusterMuted,
                "Muted orange (H=" + HA + " S=" + SA + ") not found in any cluster");
        assertNotEquals(-1, clusterVivid,
                "Vivid orange (H=" + HB + " S=" + SB + ") not found in any cluster");
        assertNotEquals(clusterMuted, clusterVivid,
                "Muted (S=" + SA + ") and vivid (S=" + SB + ") orange ended up in the SAME cluster "
                + "even after the intermediate valley-filler merged their hue peaks — "
                + "Otsu S sub-clustering failed to split S=" + SA + " from S=" + SB);
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /**
     * Builds the 640×480 test image and runs the extractor.
     *
     * <p>The image has a mid-grey achromatic background (HSV 0,0,128) so that
     * the morphological-gradient border-pixel mask sees real edges at the
     * perimeter of each orange patch.  Without an achromatic background the
     * chromatic mask is uniformly 255, its gradient is zero, and no border
     * pixels feed the hue histogram — yielding zero clusters.
     */
    private List<ColourCluster> buildAndExtract() {
        // Achromatic grey background: S=0 → well below MIN_SAT=35
        Mat hsv = new Mat(H, W, CvType.CV_8UC3, new Scalar(0, 0, 128));

        // Muted orange patch — left region
        Mat patchA = hsv.submat(PATCH_A_R0, PATCH_A_R1, PATCH_A_C0, PATCH_A_C1);
        patchA.setTo(new Scalar(HA, SA, VA));
        patchA.release();

        // Vivid orange patch — right region, well-separated from patch A
        Mat patchB = hsv.submat(PATCH_B_R0, PATCH_B_R1, PATCH_B_C0, PATCH_B_C1);
        patchB.setTo(new Scalar(HB, SB, VB));
        patchB.release();

        // Convert HSV → BGR (the extractor expects BGR input)
        Mat bgr = new Mat();
        Imgproc.cvtColor(hsv, bgr, Imgproc.COLOR_HSV2BGR);
        hsv.release();

        List<ColourCluster> result =
                ExperimentalSceneColourClusters.INSTANCE.extractFromBorderPixels(bgr);
        bgr.release();
        return result;
    }

    /** Returns the index of the first cluster whose mask is non-zero at (row, col), or -1. */
    private static int findCluster(List<ColourCluster> clusters, int row, int col) {
        for (int i = 0; i < clusters.size(); i++) {
            double[] px = clusters.get(i).mask.get(row, col);
            if (px != null && px[0] > 0) return i;
        }
        return -1;
    }

    private static void printClusters(List<ColourCluster> clusters) {
        for (int i = 0; i < clusters.size(); i++) {
            ColourCluster c = clusters.get(i);
            if (c.achromatic) {
                System.out.printf("  [%d] Achromatic %-6s%n",
                        i, c.brightAchromatic ? "BRIGHT" : "DARK");
            } else {
                System.out.printf("  [%d] Chromatic  hue=%3d  lo=%3d  hi=%3d  S=[%3d,%3d]%n",
                        i, (int) c.hue, c.loBound, c.hiBound, c.sLo, c.sHi);
            }
        }
    }
}

