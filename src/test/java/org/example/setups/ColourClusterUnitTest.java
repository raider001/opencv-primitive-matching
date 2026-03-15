package org.example.setups;

import org.example.OpenCvLoader;
import org.example.colour.ColourCluster;
import org.example.colour.ExperimentalSceneColourClusters;
import org.example.colour.SceneColourClusters;
import org.example.factories.BackgroundFactory;
import org.example.factories.BackgroundId;
import org.example.factories.ReferenceId;
import org.example.factories.ReferenceImageFactory;
import org.junit.jupiter.api.*;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Observational unit test — places 20 reference images onto two noisy
 * backgrounds and prints the colour-cluster breakdown extracted by
 * {@link SceneColourClusters#extractFromBorderPixels(Mat)}.
 *
 * <p>No detection is attempted and scores are not checked.  The goal is to
 * record how many clusters each (reference × background) combination produces
 * so we can validate the cluster-extraction heuristics by eye.
 *
 * <p>Output:
 * <ul>
 *   <li>Disk overlay PNGs → {@code test_output/cluster_unit_test/<bg_name>/}</li>
 *   <li>HTML report       → {@code test_output/cluster_unit_test/report.html}</li>
 * </ul>
 * The HTML report renders the raw scene image with an SVG overlay for each
 * cluster drawn on top.  Each cluster's layer can be toggled on/off independently.
 * Clicking the scene opens a 2× lightbox (600 px wide) with the same SVG overlay.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class ColourClusterUnitTest {

    // -------------------------------------------------------------------------
    // 20 reference images — deliberately varied: lines, circles, polygons,
    // compound shapes, text, and multi-colour shapes.
    // -------------------------------------------------------------------------
    private static final List<ReferenceId> TEST_REFS = List.of(
        ReferenceId.CIRCLE_FILLED,
        ReferenceId.CIRCLE_OUTLINE,
        ReferenceId.RECT_FILLED,
        ReferenceId.TRIANGLE_FILLED,
        ReferenceId.STAR_5_OUTLINE,
        ReferenceId.HEXAGON_FILLED,
        ReferenceId.LINE_H,
        ReferenceId.PENTAGON_OUTLINE,
        ReferenceId.ELLIPSE_H,
        ReferenceId.OCTAGON_FILLED,
        ReferenceId.POLYLINE_ZIGZAG_H,
        ReferenceId.ARC_QUARTER,
        ReferenceId.CONCAVE_PAC_MAN,
        ReferenceId.IRREGULAR_STAR,
        ReferenceId.COMPOUND_BULLSEYE,
        ReferenceId.GRID_4X4,
        ReferenceId.TEXT_A,
        ReferenceId.BICOLOUR_CIRCLE_RING,
        ReferenceId.TRICOLOUR_TRIANGLE,
        ReferenceId.CHECKERBOARD_4X4
    );

    /** The two backgrounds under test. */
    private static final BackgroundId[] BACKGROUNDS = {
        BackgroundId.BG_RANDOM_CIRCLES,
        BackgroundId.BG_RANDOM_LINES
    };

    /**
     * 12 perceptually-distinct RGB colours spread around the hue wheel.
     * Stored as [R, G, B] — convert to BGR when building OpenCV Mats,
     * use directly as CSS rgb() values in the SVG.
     */
    private static final int[][] CLUSTER_RGB = {
        {255,  50,  50},   //  0  red
        {  0, 200,  80},   //  1  green
        { 80, 120, 255},   //  2  blue
        {255, 170,   0},   //  3  amber/orange
        {200,  50, 255},   //  4  violet
        {  0, 220, 220},   //  5  cyan
        {255, 220,   0},   //  6  yellow
        {255,  50, 180},   //  7  hot-pink
        {  0, 190, 150},   //  8  teal
        {220, 130, 255},   //  9  lavender
        {255, 110,  30},   // 10  deep-orange
        { 30, 160, 255},   // 11  sky-blue
    };

    private static final int SCENE_W = 640;
    private static final int SCENE_H = 480;

    // -------------------------------------------------------------------------
    // Report data
    // -------------------------------------------------------------------------

    private record ClusterDetail(
            boolean achromatic, boolean brightAchromatic,
            int hue, int loBound, int hiBound, int sLo, int sHi, int pixels,
            String svgPaths
    ) {
        String label() {
            if (achromatic)
                return (brightAchromatic ? "Achromatic BRIGHT" : "Achromatic DARK ")
                        + "  pixels=" + pixels;
            String base = String.format("Chromatic  hue=%3d  lo=%3d  hi=%3d",
                    hue, loBound, hiBound);
            // Show S range whenever either bound is non-default:
            //  • sHi < 255  → lower sub-cluster band  (e.g. S=[35,163])
            //  • sLo > 35   → upper sub-cluster band  (e.g. S=[164,255])
            // Without this, the upper band label was identical to a non-split cluster,
            // hiding that S sub-clustering had separated the two orange types.
            if (sLo > 35 || sHi < 255)
                base += String.format("  S=[%d,%d]", sLo, sHi);
            return base + String.format("  pixels=%5d", pixels);
        }
    }

    /** One row in the HTML report — one (ref, bg) pair. */
    private record ReportRow(
            ReferenceId refId, BackgroundId bgId,
            int clusterCount,
            List<ClusterDetail> details,
            String scenePng,      // base64 PNG of the raw scene (no overlay)
            long extractNanos     // wall-clock time for the extraction call
    ) {}

    private final List<ReportRow> reportRows = new ArrayList<>();
    private final List<ReportRow> experimentalReportRows = new ArrayList<>();
    private Path outputDir;
    private Path experimentalOutputDir;

    // -------------------------------------------------------------------------
    // Colour helpers
    // -------------------------------------------------------------------------

    /** BGR Scalar for OpenCV Mat operations. */
    private static Scalar bgrScalar(int ci) {
        int[] c = CLUSTER_RGB[ci % CLUSTER_RGB.length];
        return new Scalar(c[2], c[1], c[0]);   // BGR order
    }

    /** CSS {@code rgb(...)} string for SVG stroke. */
    private static String svgRgb(int ci) {
        int[] c = CLUSTER_RGB[ci % CLUSTER_RGB.length];
        return "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
    }

    /** CSS {@code rgba(...,0.40)} string for SVG fill. */
    private static String svgFill(int ci) {
        int[] c = CLUSTER_RGB[ci % CLUSTER_RGB.length];
        return "rgba(" + c[0] + "," + c[1] + "," + c[2] + ",0.40)";
    }

    // -------------------------------------------------------------------------
    // Setup / teardown
    // -------------------------------------------------------------------------

    @BeforeAll
    void setup() throws IOException {
        OpenCvLoader.load();
        outputDir = Paths.get("test_output", "cluster_unit_test").toAbsolutePath().normalize();
        experimentalOutputDir = Paths.get("test_output", "cluster_unit_test_experimental").toAbsolutePath().normalize();
        Files.createDirectories(outputDir);
        Files.createDirectories(experimentalOutputDir);
        System.out.printf("%n=== ColourClusterUnitTest  output: %s ===%n%n", outputDir);
    }

    @AfterAll
    void writeHtmlReport() throws IOException {
        if (!reportRows.isEmpty()) {
            Path report = outputDir.resolve("report.html");
            Files.deleteIfExists(report);
            Files.writeString(report, buildHtml(reportRows, "Colour Cluster Unit Test (Production)"), StandardCharsets.UTF_8);
            System.out.printf("[ColourClusterUnitTest] HTML report: %s%n", report);
        }

        if (experimentalReportRows.isEmpty()) return;
        Path report = experimentalOutputDir.resolve("exoreport.html");
        Files.deleteIfExists(report);
        Files.writeString(report, buildHtml(experimentalReportRows,
                "Colour Cluster Unit Test (Experimental — spatial split, heal=" + ExperimentalSceneColourClusters.HEAL_RADIUS + "px)"),
                StandardCharsets.UTF_8);
        System.out.printf("[ColourClusterUnitTest] Experimental HTML report: %s%n", report);
    }

    // -------------------------------------------------------------------------
    // Test
    // -------------------------------------------------------------------------

    @Test
    @Order(1)
    @DisplayName("Print colour clusters for 20 refs × 2 backgrounds")
    void printColourClusters() throws IOException {

        int[][] summary      = new int[TEST_REFS.size()][BACKGROUNDS.length];
        int    totalClusters = 0;

        for (int bi = 0; bi < BACKGROUNDS.length; bi++) {
            BackgroundId bgId = BACKGROUNDS[bi];
            Path bgDir = outputDir.resolve(bgId.name().toLowerCase());
            Files.createDirectories(bgDir);

            System.out.printf("══════════════════════════════════════════════════════════%n");
            System.out.printf("  Background: %s%n", bgId.name());
            System.out.printf("══════════════════════════════════════════════════════════%n%n");

            for (int ri = 0; ri < TEST_REFS.size(); ri++) {
                ReferenceId refId = TEST_REFS.get(ri);

                Mat refMat = ReferenceImageFactory.build(refId);
                Mat scene  = BackgroundFactory.build(bgId, SCENE_W, SCENE_H);
                placeRefCentred(scene, refMat);

                long t0 = System.nanoTime();
                List<ColourCluster> clusters =
                        SceneColourClusters.extractFromBorderPixels(scene);
                long extractNanos = System.nanoTime() - t0;

                summary[ri][bi]  = clusters.size();
                totalClusters   += clusters.size();

                // ── Console output ────────────────────────────────────────
                System.out.printf("  %-35s  →  %2d cluster(s)%n", refId.name(), clusters.size());

                List<ClusterDetail> details = new ArrayList<>();
                for (int ci = 0; ci < clusters.size(); ci++) {
                    ColourCluster c = clusters.get(ci);
                    int pixels   = Core.countNonZero(c.mask);
                    String paths = extractSvgPaths(c.mask);

                    details.add(new ClusterDetail(
                            c.achromatic, c.brightAchromatic,
                            (int) c.hue, c.loBound, c.hiBound, c.sLo, c.sHi, pixels,
                            paths));

                    if (c.achromatic) {
                        System.out.printf("    [%2d] Achromatic %-7s  pixels=%5d%n",
                                ci, c.brightAchromatic ? "BRIGHT" : "DARK", pixels);
                    } else {
                        System.out.printf(
                                "    [%2d] Chromatic   hue=%3d  lo=%3d  hi=%3d  pixels=%5d%n",
                                ci, (int) c.hue, c.loBound, c.hiBound, pixels);
                    }
                }

                String scenePng = matToBase64Png(scene);
                reportRows.add(new ReportRow(refId, bgId, clusters.size(), details, scenePng, extractNanos));

                // ── Save colour-painted overlay to disk (for local inspection) ─
                Mat diskOverlay = buildDiskOverlay(scene, clusters);
                Imgcodecs.imwrite(
                        bgDir.resolve(refId.name().toLowerCase() + "_clusters.png").toString(),
                        diskOverlay);
                diskOverlay.release();

                // ── Release natives ───────────────────────────────────────
                clusters.forEach(ColourCluster::release);
                refMat.release();
                scene.release();

                System.out.println();
            }
        }

        // ── Summary table ─────────────────────────────────────────────────────
        System.out.printf("══════════════════════════════════════════════════════════%n");
        System.out.printf("  SUMMARY — cluster count per reference × background%n");
        System.out.printf("══════════════════════════════════════════════════════════%n%n");

        String bg0 = BACKGROUNDS[0].name().replace("BG_RANDOM_", "");
        String bg1 = BACKGROUNDS[1].name().replace("BG_RANDOM_", "");
        System.out.printf("  %-35s  %-10s  %s%n", "Reference", bg0, bg1);
        System.out.printf("  %s%n", "-".repeat(60));
        for (int ri = 0; ri < TEST_REFS.size(); ri++) {
            System.out.printf("  %-35s  %-10d  %d%n",
                    TEST_REFS.get(ri).name(), summary[ri][0], summary[ri][1]);
        }
        System.out.printf("%n  Total clusters across all %d scenes: %d%n%n",
                TEST_REFS.size() * BACKGROUNDS.length, totalClusters);

        assertTrue(totalClusters > 0,
                "Expected at least some clusters to be extracted across all scenes");
    }

    // -------------------------------------------------------------------------
    // Experimental test — spatial splitting + auto-heal
    // -------------------------------------------------------------------------

    @Test
    @Order(2)
    @DisplayName("Experimental: spatially-split clusters for 20 refs × 2 backgrounds")
    void printExperimentalColourClusters() throws IOException {

        int[][] summary      = new int[TEST_REFS.size()][BACKGROUNDS.length];
        int    totalClusters = 0;

        for (int bi = 0; bi < BACKGROUNDS.length; bi++) {
            BackgroundId bgId = BACKGROUNDS[bi];
            Path bgDir = experimentalOutputDir.resolve(bgId.name().toLowerCase());
            Files.createDirectories(bgDir);

            System.out.printf("══════════════════════════════════════════════════════════%n");
            System.out.printf("  [EXPERIMENTAL] Background: %s%n", bgId.name());
            System.out.printf("══════════════════════════════════════════════════════════%n%n");

            for (int ri = 0; ri < TEST_REFS.size(); ri++) {
                ReferenceId refId = TEST_REFS.get(ri);

                Mat refMat = ReferenceImageFactory.build(refId);
                Mat scene  = BackgroundFactory.build(bgId, SCENE_W, SCENE_H);
                placeRefCentred(scene, refMat);

                long t0 = System.nanoTime();
                List<ColourCluster> clusters =
                        ExperimentalSceneColourClusters.INSTANCE.extractFromBorderPixels(scene);
                long extractNanos = System.nanoTime() - t0;

                summary[ri][bi]  = clusters.size();
                totalClusters   += clusters.size();

                System.out.printf("  %-35s  →  %2d cluster(s)%n", refId.name(), clusters.size());

                List<ClusterDetail> details = new ArrayList<>();
                for (int ci = 0; ci < clusters.size(); ci++) {
                    ColourCluster c = clusters.get(ci);
                    int pixels   = Core.countNonZero(c.mask);
                    // dilate=false: full-pixel masks don't need gap-bridging, and
                    // dilation causes SVG paths to visually overflow cluster boundaries.
                    String paths = extractSvgPaths(c.mask, false);
                    details.add(new ClusterDetail(
                            c.achromatic, c.brightAchromatic,
                            (int) c.hue, c.loBound, c.hiBound, c.sLo, c.sHi, pixels, paths));

                    if (c.achromatic) {
                        System.out.printf("    [%2d] Achromatic %-7s  pixels=%5d%n",
                                ci, c.brightAchromatic ? "BRIGHT" : "DARK", pixels);
                    } else {
                        System.out.printf(
                                "    [%2d] Chromatic   hue=%3d  lo=%3d  hi=%3d  pixels=%5d%n",
                                ci, (int) c.hue, c.loBound, c.hiBound, pixels);
                    }
                }

                String scenePng = matToBase64Png(scene);
                experimentalReportRows.add(new ReportRow(refId, bgId, clusters.size(), details, scenePng, extractNanos));

                // Disk overlay for local inspection
                Mat diskOverlay = buildDiskOverlay(scene, clusters);
                Imgcodecs.imwrite(
                        bgDir.resolve(refId.name().toLowerCase() + "_clusters.png").toString(),
                        diskOverlay);
                diskOverlay.release();

                clusters.forEach(ColourCluster::release);
                refMat.release();
                scene.release();

                System.out.println();
            }
        }

        // Summary
        System.out.printf("══════════════════════════════════════════════════════════%n");
        System.out.printf("  [EXPERIMENTAL] SUMMARY%n");
        System.out.printf("══════════════════════════════════════════════════════════%n%n");
        String bg0 = BACKGROUNDS[0].name().replace("BG_RANDOM_", "");
        String bg1 = BACKGROUNDS[1].name().replace("BG_RANDOM_", "");
        System.out.printf("  %-35s  %-10s  %s%n", "Reference", bg0, bg1);
        System.out.printf("  %s%n", "-".repeat(60));
        for (int ri = 0; ri < TEST_REFS.size(); ri++) {
            System.out.printf("  %-35s  %-10d  %d%n",
                    TEST_REFS.get(ri).name(), summary[ri][0], summary[ri][1]);
        }
        System.out.printf("%n  Total clusters: %d  (heal radius=%dpx)%n%n",
                totalClusters, ExperimentalSceneColourClusters.HEAL_RADIUS);

        assertTrue(totalClusters > 0, "Experimental extractor produced no clusters");
    }

    // -------------------------------------------------------------------------
    // Regression assertion — muted vs vivid orange
    // -------------------------------------------------------------------------

    /**
     * Regression guard: the two orange HSV values reported as problematic,
     * H=10 S=137 V=193 (muted) and H=16 S=255 V=255 (vivid), must end up in
     * <em>different</em> clusters when both are present in the same scene.
     *
     * <p>Three sub-scenarios are exercised:
     * <ol>
     *   <li><b>Spatially separated</b> — each patch sits on a grey background so
     *       the morphological gradient produces separate border pixels.</li>
     *   <li><b>Merged-hue-cluster</b> — an intermediate orange band fills the
     *       valley between H=10 and H=16 so only Otsu S sub-clustering can
     *       separate them (S=137 → [35,254]; S=255 → [255,255]).</li>
     *   <li><b>Exact real scene</b> — {@code BG_RANDOM_LINES + CONCAVE_PAC_MAN}:
     *       the background has a random line with BGR≈(89,124,193) = H=10 S=137
     *       while the placed shape is orange BGR(0,140,255) = H=16 S=255.
     *       Both fall in the merged hue cluster [1,18]; S sub-clustering must
     *       place the PAC_MAN pixels in a different S band.</li>
     * </ol>
     */
    @Test
    @Order(3)
    @DisplayName("Regression: H=10 S=137 and H=16 S=255 must be in separate clusters")
    void mutedAndVividOrangeMustBeSeparate() {
        final int HA = 10, SA = 137, VA = 193;   // muted orange
        final int HB = 16, SB = 255, VB = 255;   // vivid orange

        // ====== Sub-scenario 1: spatially separated on grey background ======
        {
            Mat hsv = new Mat(SCENE_H, SCENE_W, CvType.CV_8UC3, new Scalar(0, 0, 128));
            hsv.submat(80, 400,  60, 260).setTo(new Scalar(HA, SA, VA));
            hsv.submat(80, 400, 380, 580).setTo(new Scalar(HB, SB, VB));
            Mat bgr = new Mat();
            Imgproc.cvtColor(hsv, bgr, Imgproc.COLOR_HSV2BGR);
            hsv.release();

            List<ColourCluster> clusters =
                    ExperimentalSceneColourClusters.INSTANCE.extractFromBorderPixels(bgr);
            bgr.release();

            int rowMid = (80 + 400) / 2;
            int cA = findClusterAt(clusters, rowMid, (60  + 260) / 2);
            int cB = findClusterAt(clusters, rowMid, (380 + 580) / 2);

            System.out.printf("%n[Order-3 sub-1] Spatially-separated: muted→cluster %d  vivid→cluster %d%n", cA, cB);
            printClusters(clusters);
            clusters.forEach(ColourCluster::release);

            assertNotEquals(-1, cA, "Sub-1: muted orange not assigned to any cluster");
            assertNotEquals(-1, cB, "Sub-1: vivid orange not assigned to any cluster");
            assertNotEquals(cA, cB, "Sub-1: muted and vivid orange are in the SAME cluster");
        }

        // ====== Sub-scenario 2: merged hue — Otsu S sub-clustering only ======
        {
            Mat hsv = new Mat(SCENE_H, SCENE_W, CvType.CV_8UC3, new Scalar(0, 0, 128));
            hsv.submat(80, 400,  60, 260).setTo(new Scalar(13, 190, 240)); // valley-filler
            hsv.submat(80, 400, 290, 430).setTo(new Scalar(HA, SA, VA));
            hsv.submat(80, 400, 460, 600).setTo(new Scalar(HB, SB, VB));
            Mat bgr = new Mat();
            Imgproc.cvtColor(hsv, bgr, Imgproc.COLOR_HSV2BGR);
            hsv.release();

            List<ColourCluster> clusters =
                    ExperimentalSceneColourClusters.INSTANCE.extractFromBorderPixels(bgr);
            bgr.release();

            int rowMid = (80 + 400) / 2;
            int cA = findClusterAt(clusters, rowMid, (290 + 430) / 2);
            int cB = findClusterAt(clusters, rowMid, (460 + 600) / 2);

            System.out.printf("[Order-3 sub-2] Merged-hue-cluster:    muted→cluster %d  vivid→cluster %d%n", cA, cB);
            printClusters(clusters);
            clusters.forEach(ColourCluster::release);

            assertNotEquals(-1, cA, "Sub-2: muted orange not assigned to any cluster");
            assertNotEquals(-1, cB, "Sub-2: vivid orange not assigned to any cluster");
            assertNotEquals(cA, cB,
                    "Sub-2: muted and vivid orange in SAME cluster — Otsu S split failed for S="
                    + SA + " vs S=" + SB);
        }

        // ====== Sub-scenario 3: exact scene BG_RANDOM_LINES + CONCAVE_PAC_MAN ======
        // CONCAVE_PAC_MAN ordinal=55 → colour palette index 55%8=7 → BGR(0,140,255) = H≈16 S=255 V=255
        // BG_RANDOM_LINES seed=44 → one random line has BGR≈(89,124,193) = H≈10 S≈137 V≈193
        // Both land in hue cluster [1,18] (hue peak ≈11); Otsu must split by S.
        {
            Mat scene  = BackgroundFactory.build(BackgroundId.BG_RANDOM_LINES, SCENE_W, SCENE_H);
            Mat refMat = ReferenceImageFactory.build(ReferenceId.CONCAVE_PAC_MAN);
            placeRefCentred(scene, refMat);
            refMat.release();

            List<ColourCluster> clusters =
                    ExperimentalSceneColourClusters.INSTANCE.extractFromBorderPixels(scene);

            // PAC_MAN orange body pixel — placement: centred in 640×480 → rows 176..303, cols 256..383.
            // Centre (64,64) in ref coords = scene (240,320) is the mouth VERTEX (painted black).
            // Use col=286 in scene = ref col 30, which is left of mouth and inside the circle body.
            //   ref (64, 30): distance from circle centre (64,64) = 34 < radius 54 ✓  not in mouth ✓
            int pacManRow = 240;
            int pacManCol = 286;   // left-of-centre inside orange body, NOT in mouth

            int pacManCluster = findClusterAt(clusters, pacManRow, pacManCol);

            // Print full cluster list with S windows for diagnosis
            System.out.printf("[Order-3 sub-3] BG_RANDOM_LINES + CONCAVE_PAC_MAN  (%d clusters)%n",
                    clusters.size());
            printClusters(clusters);
            System.out.printf("PAC_MAN body pixel (%d,%d) → cluster %d%n",
                    pacManRow, pacManCol, pacManCluster);

            // Find any cluster in the SAME hue window as the PAC_MAN cluster whose S band overlaps.
            // Such overlap means the background orange (S≈137) and PAC_MAN (S=255) were NOT split.
            String assertion = null;
            if (pacManCluster != -1) {
                ColourCluster pacC = clusters.get(pacManCluster);
                System.out.printf("PAC_MAN cluster: hue=%d  lo=%d  hi=%d  S=[%d,%d]%n",
                        (int) pacC.hue, pacC.loBound, pacC.hiBound, pacC.sLo, pacC.sHi);
                for (int i = 0; i < clusters.size(); i++) {
                    if (i == pacManCluster) continue;
                    ColourCluster bg = clusters.get(i);
                    if (bg.achromatic) continue;
                    if (bg.loBound == pacC.loBound && bg.hiBound == pacC.hiBound) {
                        boolean overlap = !(pacC.sHi < bg.sLo || bg.sHi < pacC.sLo);
                        System.out.printf(
                                "  Same-hue peer [%d]: sLo=%d sHi=%d  PAC_MAN S=[%d,%d]  overlap=%b%n",
                                i, bg.sLo, bg.sHi, pacC.sLo, pacC.sHi, overlap);
                        if (overlap) {
                            assertion = "Sub-3 BG_RANDOM_LINES+CONCAVE_PAC_MAN: "
                                    + "background orange cluster [" + i + "] S=[" + bg.sLo + "," + bg.sHi
                                    + "] overlaps with PAC_MAN cluster [" + pacManCluster + "] S=["
                                    + pacC.sLo + "," + pacC.sHi + "] — Otsu S sub-clustering did NOT"
                                    + " separate muted orange (S≈137) from vivid orange (S=255) in the real scene";
                            break;
                        }
                    }
                }
            }

            scene.release();
            clusters.forEach(ColourCluster::release);

            assertNotEquals(-1, pacManCluster,
                    "Sub-3: PAC_MAN body pixel (" + pacManRow + "," + pacManCol + ") not in any cluster");
            if (assertion != null) {
                org.junit.jupiter.api.Assertions.fail(assertion);
            }
        }
    }

    /** Returns the index of the first cluster whose mask is 255 at (row, col), or -1. */
    private static int findClusterAt(List<ColourCluster> clusters, int row, int col) {
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
                System.out.printf("  [%d] Achromatic %-6s%n", i,
                        c.brightAchromatic ? "BRIGHT" : "DARK");
            } else {
                System.out.printf("  [%d] Chromatic  hue=%3d  lo=%3d  hi=%3d  S=[%3d,%3d]%n",
                        i, (int) c.hue, c.loBound, c.hiBound, c.sLo, c.sHi);
            }
        }
    }

    // -------------------------------------------------------------------------
    // SVG path extraction
    // -------------------------------------------------------------------------

    /**
     * Converts a cluster mask into SVG path strings.
     * When {@code dilate} is {@code true} the mask is expanded by a 5×5 kernel
     * first, which helps close gaps in thin border-pixel masks (production).
     * Pass {@code false} for full-pixel masks (experimental) to avoid path
     * boundaries spilling outside the actual cluster area.
     */
    private static String extractSvgPaths(Mat mask, boolean dilate) {
        Mat src;
        Mat kernel = null;
        if (dilate) {
            kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));
            src = new Mat();
            Imgproc.dilate(mask, src, kernel);
            kernel.release();
        } else {
            src = mask.clone();
        }

        List<MatOfPoint> contours  = new ArrayList<>();
        Mat              hierarchy = new Mat();
        Imgproc.findContours(src, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        src.release();
        hierarchy.release();

        StringBuilder sb = new StringBuilder();
        for (MatOfPoint cnt : contours) {
            if (Imgproc.contourArea(cnt) < 16.0) { cnt.release(); continue; }

            MatOfPoint2f cnt2f  = new MatOfPoint2f(cnt.toArray());
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(cnt2f, approx, 2.0, true);
            cnt.release();
            cnt2f.release();

            Point[] pts = approx.toArray();
            approx.release();
            if (pts.length < 2) continue;

            sb.append('M').append((int) pts[0].x).append(',').append((int) pts[0].y);
            for (int i = 1; i < pts.length; i++) {
                sb.append('L').append((int) pts[i].x).append(',').append((int) pts[i].y);
            }
            sb.append('Z');
        }
        return sb.toString();
    }

    /** Convenience overload — dilates by default (preserves production behaviour). */
    private static String extractSvgPaths(Mat mask) {
        return extractSvgPaths(mask, true);
    }

    // -------------------------------------------------------------------------
    // HTML report builder
    // -------------------------------------------------------------------------

    private String buildHtml(List<ReportRow> rows, String title) {
        String ts    = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));

        Map<BackgroundId, List<ReportRow>> byBg = new LinkedHashMap<>();
        for (ReportRow r : rows)
            byBg.computeIfAbsent(r.bgId(), _ -> new ArrayList<>()).add(r);

        Map<BackgroundId, Integer> bgTotals = new LinkedHashMap<>();
        byBg.forEach((bg, list) ->
                bgTotals.put(bg, list.stream().mapToInt(ReportRow::clusterCount).sum()));

        StringBuilder sb = new StringBuilder();
        sb.append("<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>")
          .append("<title>").append(esc(title)).append("</title>")
          .append("<style>").append(CSS).append("</style></head><body>")
          // ── Header ──────────────────────────────────────────────────────────
          .append("<div class='header'>")
          .append("<h1>").append(esc(title)).append("</h1>")
          .append("<div class='ts-line'>Generated: <span class='ts-val'>").append(ts).append("</span></div>")
          .append("<p class='subtitle'>")
          .append(rows.size()).append(" scenes &nbsp; ")
          .append(TEST_REFS.size()).append(" references &nbsp; ")
          .append(BACKGROUNDS.length).append(" backgrounds")
          .append("</p>")
          .append("<div class='legend-block'>")
          .append("<div class='legend-title'>How to read each row</div>")
          .append("<div class='legend-row'>")
          .append("<span class='pl-step'>Hover image → overlays appear</span>")
          .append("<span class='pl-arrow'>→</span>")
          .append("<span class='pl-step'>Click button: pin on → pin off → hover-only</span>")
          .append("<span class='pl-arrow'>→</span>")
          .append("<span class='pl-step'>Click image to open 2× lightbox</span>")
          .append("</div></div>")
          .append("<div class='legend-block'><div class='legend-title'>Total clusters by background</div>")
          .append("<div class='legend-row'>");
        bgTotals.forEach((bg, total) ->
                sb.append("<span class='pl-step'>").append(esc(bg.name()))
                  .append("&nbsp;:&nbsp;<b>").append(total).append("</b></span>"));
        sb.append("</div></div></div>");

        // ── Per-background sections ─────────────────────────────────────────
        int rowIdx = 0;
        for (Map.Entry<BackgroundId, List<ReportRow>> entry : byBg.entrySet()) {
            sb.append("<section>")
              .append("<h2>").append(esc(entry.getKey().name())).append("</h2>");

            for (ReportRow r : entry.getValue()) {
                sb.append("<div class='row'>");

                // ── meta line ────────────────────────────────────────────
                double ms = r.extractNanos() / 1_000_000.0;
                String timeStr = ms < 1.0
                        ? String.format("%.2f ms", ms)
                        : String.format("%.1f ms", ms);
                sb.append("<div class='row-meta'>")
                  .append("<span class='row-id'>").append(esc(r.refId().name())).append("</span>")
                  .append("<span class='cluster-badge'>").append(r.clusterCount())
                  .append(r.clusterCount() == 1 ? " cluster" : " clusters").append("</span>")
                  .append("<span class='time-badge'>⏱ ").append(timeStr).append("</span>")
                  .append("</div>");

                // ── cluster toggle buttons ────────────────────────────────
                sb.append("<div class='cluster-controls'>");
                for (int ci = 0; ci < r.details().size(); ci++) {
                    ClusterDetail d  = r.details().get(ci);
                    String gId       = "cl-" + rowIdx + "-" + ci;
                    String btnId     = "cbt-" + rowIdx + "-" + ci;
                    String colorCss  = svgRgb(ci);
                    sb.append("<button id='").append(btnId)
                      .append("' class='cbt cbt-off' style='--cc:").append(colorCss).append("'")
                      .append(" title='Click to pin on'")
                      .append(" onclick=\"toggleCluster('").append(gId)
                      .append("','").append(btnId).append("');event.stopPropagation()\">")
                      .append("[").append(ci).append("] ").append(esc(d.label()))
                      .append("</button>");
                }
                sb.append("</div>");

                // ── scene image + SVG overlay ─────────────────────────────
                String imgId    = "si-" + rowIdx;
                String svgId    = "sv-" + rowIdx;
                String caption  = r.refId().name() + " on " + r.bgId().name();

                sb.append("<div class='scene-block' onclick=\"openSceneLb(")
                  .append(rowIdx).append(",'").append(imgId).append("','")
                  .append(svgId).append("','").append(escJs(caption)).append("')\">")
                  // scene image
                  .append("<img id='").append(imgId).append("' src='data:image/png;base64,")
                  .append(r.scenePng()).append("' alt='scene'/>")
                  // SVG overlay — viewBox matches the 640×480 scene coordinate system
                  .append("<svg id='").append(svgId)
                  .append("' viewBox='0 0 ").append(SCENE_W).append(" ").append(SCENE_H).append("'>");

                for (int ci = 0; ci < r.details().size(); ci++) {
                    ClusterDetail d = r.details().get(ci);
                    String gId = "cl-" + rowIdx + "-" + ci;
                    sb.append("<g id='").append(gId).append("' data-idx='").append(ci).append("'>");
                    if (!d.svgPaths().isEmpty()) {
                        sb.append("<path d='").append(d.svgPaths())
                          .append("' fill='").append(svgFill(ci))
                          .append("' stroke='").append(svgRgb(ci))
                          .append("' stroke-width='2' vector-effect='non-scaling-stroke'/>");
                    }
                    sb.append("</g>");
                }
                sb.append("</svg></div>"); // close scene-block

                sb.append("</div>"); // close row
                rowIdx++;
            }
            sb.append("</section>");
        }

        // ── Summary table ─────────────────────────────────────────────────────
        sb.append("<section><h2>Summary — cluster count per reference × background</h2>")
          .append("<table class='sum-table'><thead><tr><th>Reference</th>");
        for (BackgroundId bg : BACKGROUNDS)
            sb.append("<th>").append(esc(bg.name().replace("BG_RANDOM_", ""))).append("</th>");
        sb.append("</tr></thead><tbody>");

        Map<ReferenceId, int[]> countMap = new LinkedHashMap<>();
        for (ReferenceId ref : TEST_REFS) countMap.put(ref, new int[BACKGROUNDS.length]);
        for (ReportRow r : rows) {
            int[] counts = countMap.get(r.refId());
            for (int bi = 0; bi < BACKGROUNDS.length; bi++) {
                if (r.bgId() == BACKGROUNDS[bi]) counts[bi] = r.clusterCount();
            }
        }
        for (Map.Entry<ReferenceId, int[]> e : countMap.entrySet()) {
            sb.append("<tr><td class='ref-cell'>").append(esc(e.getKey().name())).append("</td>");
            for (int c : e.getValue())
                sb.append("<td class='cnt-cell'>").append(c).append("</td>");
            sb.append("</tr>");
        }
        sb.append("</tbody></table></section>");

        // ── Lightbox ─────────────────────────────────────────────────────────
        sb.append("<div id='lb' class='lb-overlay' onclick='closeLb()'>")
          .append("<div class='lb-box' onclick='event.stopPropagation()'>")
          .append("<button class='lb-close' onclick='closeLb()'>✕</button>")
          .append("<div id='lb-scene'></div>")
          .append("<div id='lb-caption' class='lb-caption'></div>")
          .append("</div></div>")
          .append("<script>").append(JS).append("</script>")
          .append("</body></html>");

        return sb.toString();
    }

    // -------------------------------------------------------------------------
    // HTML / JS helpers
    // -------------------------------------------------------------------------

    private static String esc(String s) {
        return s == null ? "" : s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;");
    }

    /** Escapes a string for embedding inside a JS single-quoted string literal. */
    private static String escJs(String s) {
        return s == null ? "" : s.replace("\\", "\\\\").replace("'", "\\'");
    }

    /** Encodes {@code m} as a base64 PNG string. */
    private static String matToBase64Png(Mat m) {
        try {
            MatOfByte buf = new MatOfByte();
            Imgcodecs.imencode(".png", m, buf);
            return Base64.getEncoder().encodeToString(buf.toArray());
        } catch (Exception e) { return ""; }
    }

    // -------------------------------------------------------------------------
    // Placement helper
    // -------------------------------------------------------------------------

    private static void placeRefCentred(Mat scene, Mat refMat) {
        int dstX = (scene.cols() - refMat.cols()) / 2;
        int dstY = (scene.rows() - refMat.rows()) / 2;

        Mat grey = new Mat();
        Imgproc.cvtColor(refMat, grey, Imgproc.COLOR_BGR2GRAY);
        Mat mask = new Mat();
        Imgproc.threshold(grey, mask, 5, 255, Imgproc.THRESH_BINARY);
        grey.release();

        Mat roi = scene.submat(new Rect(dstX, dstY, refMat.cols(), refMat.rows()));
        refMat.copyTo(roi, mask);
        mask.release();
        roi.release();
    }

    // -------------------------------------------------------------------------
    // Disk overlay (saved as PNG — not embedded in HTML)
    // -------------------------------------------------------------------------

    /**
     * Builds a colour-painted overlay image for disk inspection.
     * The scene is dimmed to 30 % and each cluster mask is painted in its
     * distinct colour.  Cluster centroids are labelled.
     * This image is <em>not</em> embedded in the HTML report — the HTML uses
     * the SVG overlay instead.
     */
    private static Mat buildDiskOverlay(Mat scene,
                                        List<ColourCluster> clusters) {
        Mat backdrop = new Mat();
        scene.convertTo(backdrop, -1, 0.3, 0);
        Mat overlay = backdrop.clone();
        backdrop.release();

        for (int i = 0; i < clusters.size(); i++) {
            Mat colorLayer = new Mat(scene.size(), scene.type(), bgrScalar(i));
            colorLayer.copyTo(overlay, clusters.get(i).mask);
            colorLayer.release();
        }

        Imgproc.putText(overlay, "clusters: " + clusters.size(),
                new Point(8, 24), Imgproc.FONT_HERSHEY_SIMPLEX, 0.7,
                new Scalar(255, 255, 255), 2);

        for (int i = 0; i < clusters.size(); i++) {
            ColourCluster c = clusters.get(i);
            Moments m = Imgproc.moments(c.mask, true);
            if (m.get_m00() > 0) {
                int cx = (int) (m.get_m10() / m.get_m00());
                int cy = (int) (m.get_m01() / m.get_m00());
                String lbl = c.achromatic ? (c.brightAchromatic ? "B" : "D") : String.valueOf(i);
                Imgproc.putText(overlay, lbl, new Point(cx - 5, cy + 6),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.55, bgrScalar(i), 2);
            }
        }
        return overlay;
    }

    // -------------------------------------------------------------------------
    // CSS
    // -------------------------------------------------------------------------
    private static final String CSS = """
        *{box-sizing:border-box;margin:0;padding:0}
        body{font-family:system-ui,sans-serif;background:#0d1117;color:#c9d1d9;padding:0 0 48px}
        .header{background:#161b22;padding:20px 32px 16px;border-bottom:1px solid #30363d;margin-bottom:20px}
        .header h1{color:#58a6ff;font-size:1.4rem;margin-bottom:4px}
        .ts-line{font-size:.75rem;color:#8b949e;margin-bottom:6px}
        .ts-val{color:#79c0ff;font-weight:600}
        .subtitle{color:#8b949e;font-size:.88rem;margin-bottom:8px}
        .pl-step{background:#21262d;border:1px solid #30363d;border-radius:4px;padding:2px 7px;font-size:.76rem}
        .pl-arrow{color:#484f58;font-size:.9rem;margin:0 2px}
        .legend-block{background:#21262d;border:1px solid #30363d;border-radius:6px;padding:8px 12px;margin-top:8px}
        .legend-title{font-size:.72rem;font-weight:700;color:#79c0ff;margin-bottom:5px;text-transform:uppercase}
        .legend-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
        section{padding:0 24px 16px}
        h2{color:#79c0ff;font-size:1rem;margin:14px 0 10px;padding-bottom:4px;border-bottom:1px solid #21262d}
        .row{background:#161b22;border:1px solid #30363d;border-radius:8px;margin-bottom:8px;padding:10px 12px;display:flex;flex-direction:column;gap:6px}
        .row-meta{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
        .row-id{font-size:.72rem;font-weight:700;background:#21262d;border-radius:3px;padding:1px 6px;color:#79c0ff;white-space:nowrap}
        .cluster-badge{font-size:.75rem;font-weight:600;background:#0d2619;border:1px solid #238636;border-radius:4px;padding:1px 7px;color:#56d364;white-space:nowrap}
        .time-badge{font-size:.75rem;font-weight:600;background:#0d1b2a;border:1px solid #1f6feb;border-radius:4px;padding:1px 7px;color:#79c0ff;white-space:nowrap}
        /* Cluster toggle buttons — 2 states: off (hover-reveals) / on (pinned) */
        .cluster-controls{display:flex;flex-wrap:wrap;gap:4px;margin:2px 0 4px}
        .cbt{font-family:monospace;font-size:.65rem;padding:2px 8px;border-radius:4px;cursor:pointer;transition:opacity .15s,color .15s,border-color .15s}
        .cbt-off{background:#0d1117;border:1.5px solid #484f58;border-left:4px solid var(--cc);color:#8b949e}
        .cbt-off:hover{opacity:.9;color:#c9d1d9}
        .cbt-on{background:#0d1117;border:1.5px solid var(--cc);border-left:4px solid var(--cc);color:#c9d1d9}
        .cbt-on:hover{opacity:.75}
        .cbt-disabled{background:#0d1117;border:1.5px solid #3a2020;border-left:4px solid #6e3535;color:#5a3a3a;text-decoration:line-through}
        .cbt-disabled:hover{border-color:#7a3a3a;color:#8b5555}
        /* Scene + SVG overlay block */
        .scene-block{position:relative;display:inline-block;width:300px;cursor:zoom-in;flex-shrink:0;user-select:none}
        .scene-block img{width:300px;height:auto;display:block;border:1px solid #30363d;border-radius:4px}
        .scene-block svg{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;border-radius:4px}
        /* Overlay groups: invisible by default, revealed on hover; cluster-on stays visible; cluster-off always hidden */
        .scene-block svg g{opacity:0;transition:opacity .18s}
        .scene-block:hover svg g:not(.cluster-full-scene):not(.cluster-off){opacity:1}
        .scene-block svg g.cluster-on{opacity:1!important}
        .scene-block svg g.cluster-off{opacity:0!important}
        /* Full-scene clusters: excluded from hover, semi-transparent when pinned on, paths non-interactive */
        .cluster-full-scene path{pointer-events:none!important}
        .scene-block svg g.cluster-full-scene.cluster-on{opacity:.65!important}
        /* Summary table */
        .sum-table{border-collapse:collapse;width:auto;margin:8px 0;font-size:.8rem}
        .sum-table th{background:#21262d;color:#79c0ff;padding:4px 14px;border:1px solid #30363d;font-weight:600}
        .ref-cell{color:#c9d1d9;padding:3px 12px;border:1px solid #30363d;font-family:monospace;white-space:nowrap}
        .cnt-cell{color:#56d364;padding:3px 14px;border:1px solid #30363d;text-align:center;font-weight:600}
        /* Lightbox */
        .lb-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.88);z-index:1000;align-items:center;justify-content:center;cursor:zoom-out}
        .lb-visible{display:flex}
        .lb-box{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;max-width:92vw;max-height:92vh;overflow:auto;position:relative;cursor:default}
        .lb-close{position:absolute;top:8px;right:10px;background:none;border:none;color:#8b949e;font-size:1.2rem;cursor:pointer;line-height:1}
        .lb-close:hover{color:#c9d1d9}
        #lb-scene{display:flex;flex-direction:column;align-items:center;gap:8px}
        .lb-controls{display:flex;flex-wrap:wrap;gap:4px;max-width:1200px;width:100%}
        #lb-scene .lb-wrap{position:relative;display:inline-block;width:1200px}
        #lb-scene .lb-wrap img{width:1200px;height:auto;display:block;border:1px solid #30363d;border-radius:4px}
        #lb-scene .lb-wrap svg{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:auto;border-radius:4px}
        /* In lightbox: normal groups visible by default; full-scene groups hidden by default */
        #lb-scene .lb-wrap svg g{opacity:1;transition:opacity .18s}
        #lb-scene .lb-wrap svg g.cluster-lb-off{opacity:0!important}
        #lb-scene .lb-wrap svg g.cluster-full-scene{opacity:0}
        #lb-scene .lb-wrap svg g.cluster-full-scene.cluster-lb-on{opacity:.65!important}
        #lb-scene .lb-wrap svg path{pointer-events:all;cursor:pointer}
        .lb-caption{font-size:.78rem;color:#8b949e;margin-top:8px;text-align:center}
        """;

    // -------------------------------------------------------------------------
    // JavaScript
    // -------------------------------------------------------------------------
    private static final String JS = """
        // On load: mark full-scene overlay groups so they are excluded from hover-reveal
        document.addEventListener('DOMContentLoaded', () => {
            document.querySelectorAll('.scene-block svg g[data-idx]').forEach(g => {
                if (isFullSceneGroup(g)) {
                    g.classList.add('cluster-full-scene');
                    const btn = document.getElementById(g.id.replace(/^cl-/, 'cbt-'));
                    if (btn) btn.title = 'Click to show (covers entire scene)';
                }
            });
        });
        // True when the group's path bounding box covers >85% of both SVG dimensions
        function isFullSceneGroup(g) {
            const path = g.querySelector('path');
            if (!path) return false;
            try {
                const bb = path.getBBox();
                const vb = g.closest('svg').viewBox.baseVal;
                return bb.width > 0.85 * vb.width && bb.height > 0.85 * vb.height;
            } catch { return false; }
        }
        // 3-state cycle for thumbnail:
        //  • cbt-off  (no g-class)   → hover-only (default)
        //  • cbt-on   (cluster-on)   → pinned visible
        //  • cbt-disabled (cluster-off) → explicitly disabled — never shows, not even on hover
        function toggleCluster(gId, btnId) {
            const g   = document.getElementById(gId);
            const btn = document.getElementById(btnId);
            if (!g) return;
            const isFull = g.classList.contains('cluster-full-scene');
            const isOn   = g.classList.contains('cluster-on');
            const isOff  = g.classList.contains('cluster-off');
            // Remove current state classes before applying next
            g.classList.remove('cluster-on', 'cluster-off');
            if (btn) btn.classList.remove('cbt-on', 'cbt-off', 'cbt-disabled');
            if (!isOn && !isOff) {
                // hover-only → pinned on
                g.classList.add('cluster-on');
                if (btn) {
                    btn.classList.add('cbt-on');
                    btn.title = isFull ? 'Click to disable (covers entire scene)' : 'Click to disable (hide even on hover)';
                }
            } else if (isOn) {
                // pinned on → disabled (never shows)
                g.classList.add('cluster-off');
                if (btn) {
                    btn.classList.add('cbt-disabled');
                    btn.title = 'Click to restore hover-only mode';
                }
            } else {
                // disabled → hover-only (back to default, no classes)
                if (btn) {
                    btn.classList.add('cbt-off');
                    btn.title = isFull ? 'Click to show (covers entire scene)' : 'Click to pin on';
                }
            }
        }
        function openSceneLb(rowIdx, imgId, svgId, caption) {
            const img = document.getElementById(imgId);
            const svg = document.getElementById(svgId);
            const lbs = document.getElementById('lb-scene');
            lbs.innerHTML = '';

            const controls = document.createElement('div');
            controls.className = 'lb-controls';

            const wrap = document.createElement('div');
            wrap.className = 'lb-wrap';
            const i2 = img.cloneNode(true);
            i2.removeAttribute('id');
            wrap.appendChild(i2);

            if (svg) {
                const s2 = svg.cloneNode(true);
                s2.removeAttribute('id');
                const groups = Array.from(s2.querySelectorAll('g'));
                groups.forEach((g, idx) => {
                    const lbGid = 'lb-g-' + idx;
                    const lbBid = 'lb-b-' + idx;
                    g.id = lbGid;

                    const origBtn = document.getElementById('cbt-' + rowIdx + '-' + idx);
                    if (origBtn) {
                        const btn = origBtn.cloneNode(true);
                        btn.id = lbBid;
                        // Normal clusters start visible (cbt-on); full-scene start hidden (cbt-off)
                        btn.classList.remove('cbt-off', 'cbt-on');
                        btn.classList.add('cbt-on');
                        btn.disabled = false;
                        btn.title = 'Click to hide';
                        btn.removeAttribute('onclick');
                        btn.addEventListener('click', e => {
                            toggleLbCluster(lbGid, lbBid);
                            e.stopPropagation();
                        });
                        controls.appendChild(btn);
                    }
                });
                // full-scene paths have pointer-events:none so clicks fall through naturally
                s2.addEventListener('click', e => {
                    const g = e.target.closest('g[data-idx]');
                    if (!g) return;
                    toggleLbCluster(g.id, 'lb-b-' + g.dataset.idx);
                    e.stopPropagation();
                });
                wrap.appendChild(s2);
            }

            lbs.appendChild(controls);
            lbs.appendChild(wrap);
            document.getElementById('lb-caption').textContent = caption;
            document.getElementById('lb').classList.add('lb-visible');

            // Lightbox is live — mark full-scene groups and flip their buttons to off (hidden)
            wrap.querySelectorAll('svg g[data-idx]').forEach(g => {
                if (isFullSceneGroup(g)) {
                    g.classList.add('cluster-full-scene');
                    const btn = document.getElementById('lb-b-' + g.dataset.idx);
                    if (btn) {
                        btn.classList.replace('cbt-on', 'cbt-off');
                        btn.title = 'Click to show (covers entire scene)';
                    }
                }
            });
        }
        // Thumbnail: off (hover-reveals, or hidden for full-scene) ↔ on (pinned visible)
        function toggleCluster(gId, btnId) {
            const g   = document.getElementById(gId);
            const btn = document.getElementById(btnId);
            if (!g) return;
            const pinOn = g.classList.toggle('cluster-on');
            if (btn) {
                btn.classList.toggle('cbt-on',  pinOn);
                btn.classList.toggle('cbt-off', !pinOn);
                btn.title = g.classList.contains('cluster-full-scene')
                    ? (pinOn ? 'Click to hide (covers entire scene)' : 'Click to show (covers entire scene)')
                    : (pinOn ? 'Click to unpin' : 'Click to pin on');
            }
        }
        // Lightbox: normal = default visible, click hides; full-scene = default hidden, click shows
        function toggleLbCluster(gId, btnId) {
            const g   = document.getElementById(gId);
            const btn = document.getElementById(btnId);
            if (!g) return;
            if (g.classList.contains('cluster-full-scene')) {
                const shown = g.classList.toggle('cluster-lb-on');
                if (btn) {
                    btn.classList.toggle('cbt-on',   shown);
                    btn.classList.toggle('cbt-off',  !shown);
                    btn.title = shown ? 'Click to hide (covers entire scene)' : 'Click to show (covers entire scene)';
                }
            } else {
                const hidden = g.classList.toggle('cluster-lb-off');
                if (btn) {
                    btn.classList.toggle('cbt-on',  !hidden);
                    btn.classList.toggle('cbt-off',  hidden);
                    btn.title = hidden ? 'Click to show' : 'Click to hide';
                }
            }
        }
        function closeLb() { document.getElementById('lb').classList.remove('lb-visible'); }
        document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLb(); });
        """;
}
