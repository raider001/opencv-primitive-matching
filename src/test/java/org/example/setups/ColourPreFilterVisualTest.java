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
import java.util.*;

/**
 * TEMPORARY visual sanity-check for ColourPreFilter.
 *
 * <p>For each reference, picks one representative scene per category
 * (A_CLEAN, B_TRANSFORMED, C_DEGRADED, D_NEGATIVE) and saves a panel PNG:
 * <pre>
 *   Row 0 — Reference:  [ Original ] [ CF_LOOSE masked ] [ CF_TIGHT masked ]
 *   Row 1 — Scene:      [ Original ] [ CF_LOOSE masked ] [ CF_TIGHT masked ]
 * </pre>
 * A/B/C scenes contain the reference shape; D scenes do not — useful for
 * confirming that the filter isolates the shape colour rather than background noise.
 *
 * <p>Outputs go to test_output/colour_prefilter_visual/.
 * No assertions — inspect the images visually.
 */
@DisplayName("TEMP — Colour Pre-Filter Visual Check")
class ColourPreFilterVisualTest {

    private static final Path OUT = Paths.get("test_output", "colour_prefilter_visual");

    /** References to inspect. */
    private static final ReferenceId[] REFS = {
        ReferenceId.CIRCLE_OUTLINE,
        ReferenceId.RECT_FILLED,
        ReferenceId.HEXAGON_OUTLINE,
        ReferenceId.TEXT_A,
    };

    /**
     * One preferred variant per category.  The first match found in the catalogue
     * for that category is used if the preferred variant isn't present.
     */
    private static final Map<SceneCategory, String> PREFERRED = Map.of(
        SceneCategory.A_CLEAN,       "clean_bg_noise_light",
        SceneCategory.B_TRANSFORMED, "rot_45",
        SceneCategory.C_DEGRADED,    "blur_5x5",
        SceneCategory.D_NEGATIVE,    ""   // any D scene will do
    );

    /** Tile size for each image in the panel. */
    private static final int TILE = 256;

    @BeforeAll
    static void setup() throws IOException {
        OpenCvLoader.load();
        Files.createDirectories(OUT.toAbsolutePath());
    }

    @Test
    @DisplayName("Save CF_LOOSE / CF_TIGHT panels — one per (reference × scene category)")
    void generatePanels() throws IOException {

        List<SceneEntry> catalogue = SceneCatalogueLoader.load(REFS);

        System.out.println("\n=== ColourPreFilter Visual Check (per category) ===");
        System.out.printf("%-26s  %-10s  %-24s  %8s  %8s  %8s  %8s%n",
                "Reference", "Category", "Scene variant",
                "LRef%", "TRef%", "LScene%", "TScene%");
        System.out.println("-".repeat(105));

        for (ReferenceId refId : REFS) {
            Mat refMat    = ReferenceImageFactory.build(refId);
            Mat refLoose  = ColourPreFilter.applyMaskedBgrToReference(refMat, refId, ColourPreFilter.LOOSE);
            Mat refTight  = ColourPreFilter.applyMaskedBgrToReference(refMat, refId, ColourPreFilter.TIGHT);
            double lRefPct = pct(ColourPreFilter.applyToReference(refMat, refId, ColourPreFilter.LOOSE));
            double tRefPct = pct(ColourPreFilter.applyToReference(refMat, refId, ColourPreFilter.TIGHT));

            for (SceneCategory cat : SceneCategory.values()) {
                String preferred = PREFERRED.get(cat);

                // For A/B/C: only scenes that contain this reference shape.
                // For D: any negative scene.
                SceneEntry scene = catalogue.stream()
                        .filter(s -> s.category() == cat)
                        .filter(s -> cat == SceneCategory.D_NEGATIVE
                                  || s.primaryReferenceId() == refId)
                        .filter(s -> preferred == null || preferred.isEmpty()
                                  || preferred.equals(s.variantLabel()))
                        .findFirst()
                        // fallback: any scene in this category for this ref
                        .or(() -> catalogue.stream()
                                .filter(s -> s.category() == cat)
                                .filter(s -> cat == SceneCategory.D_NEGATIVE
                                          || s.primaryReferenceId() == refId)
                                .findFirst())
                        .orElse(null);

                if (scene == null) {
                    System.out.printf("  [SKIP] %s / %s — no scene found%n", refId.name(), cat);
                    continue;
                }

                Mat sceneMat   = scene.sceneMat();
                Mat sceneLoose = ColourPreFilter.applyMaskedBgrToScene(sceneMat, refId, ColourPreFilter.LOOSE);
                Mat sceneTight = ColourPreFilter.applyMaskedBgrToScene(sceneMat, refId, ColourPreFilter.TIGHT);
                double lScenePct = pct(ColourPreFilter.applyToScene(sceneMat, refId, ColourPreFilter.LOOSE));
                double tScenePct = pct(ColourPreFilter.applyToScene(sceneMat, refId, ColourPreFilter.TIGHT));

                System.out.printf("%-26s  %-10s  %-24s  %6.1f%%  %6.1f%%  %7.1f%%  %7.1f%%%n",
                        refId.name(), cat.name(), scene.variantLabel(),
                        lRefPct, tRefPct, lScenePct, tScenePct);

                Mat panel = buildPanel(
                        refMat, refLoose, refTight,
                        sceneMat, sceneLoose, sceneTight,
                        refId.name(), cat.name(), scene.variantLabel(),
                        lRefPct, tRefPct, lScenePct, tScenePct);

                String filename = refId.name() + "__" + cat.name() + "__" + scene.variantLabel() + ".png";
                Imgcodecs.imwrite(OUT.toAbsolutePath().resolve(filename).toString(), panel);

                sceneLoose.release();
                sceneTight.release();
                panel.release();
            }

            refMat.release();
            refLoose.release();
            refTight.release();
        }

        System.out.println("\nAll panels written to: " + OUT.toAbsolutePath());
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /** Returns white-pixel fraction as a percentage, then releases the mask. */
    private static double pct(Mat mask) {
        double v = ColourPreFilter.whitePixelFraction(mask) * 100.0;
        mask.release();
        return v;
    }

    /**
     * Builds a 2-row × 3-column panel:
     * <pre>
     *   Row 0 (reference):  [ original ] [ CF LOOSE ] [ CF TIGHT ]
     *   Row 1 (scene):      [ original ] [ CF LOOSE ] [ CF TIGHT ]
     * </pre>
     */
    private static Mat buildPanel(Mat refOrig,   Mat refLoose,   Mat refTight,
                                   Mat sceneOrig, Mat sceneLoose, Mat sceneTight,
                                   String refName, String catName, String sceneVariant,
                                   double lRefPct, double tRefPct,
                                   double lScenePct, double tScenePct) {

        final int LABEL_H = 32;
        final int COLS    = 3;
        final int ROWS    = 2;
        final int W       = COLS * TILE;
        final int H       = ROWS * (TILE + LABEL_H);

        Mat out = new Mat(H, W, CvType.CV_8UC3, new Scalar(20, 20, 20));

        String[][] labels = {
            { "Ref: " + refName,
              String.format("CF LOOSE  (%.1f%% kept)", lRefPct),
              String.format("CF TIGHT  (%.1f%% kept)", tRefPct) },
            { catName + ": " + sceneVariant,
              String.format("CF LOOSE  (%.1f%% kept)", lScenePct),
              String.format("CF TIGHT  (%.1f%% kept)", tScenePct) }
        };

        Mat[][] tiles = {
            { refOrig,   refLoose,   refTight   },
            { sceneOrig, sceneLoose, sceneTight }
        };

        for (int row = 0; row < ROWS; row++) {
            int yBase = row * (TILE + LABEL_H);
            for (int col = 0; col < COLS; col++) {
                int xBase = col * TILE;

                Imgproc.putText(out, labels[row][col],
                        new Point(xBase + 4, yBase + 20),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.42,
                        new Scalar(180, 200, 255), 1, Imgproc.LINE_AA, false);

                Mat tile = new Mat();
                Imgproc.resize(tiles[row][col], tile, new Size(TILE, TILE));
                tile.copyTo(out.submat(new Rect(xBase, yBase + LABEL_H, TILE, TILE)));
                tile.release();
            }
        }

        // Grid lines
        Scalar grid = new Scalar(60, 60, 60);
        for (int col = 1; col < COLS; col++)
            Imgproc.line(out, new Point(col * TILE, 0), new Point(col * TILE, H), grid, 1);
        for (int row = 1; row < ROWS; row++)
            Imgproc.line(out, new Point(0, row * (TILE + LABEL_H)),
                    new Point(W, row * (TILE + LABEL_H)), grid, 1);

        return out;
    }
}

