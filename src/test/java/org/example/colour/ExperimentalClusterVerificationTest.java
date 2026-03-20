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

