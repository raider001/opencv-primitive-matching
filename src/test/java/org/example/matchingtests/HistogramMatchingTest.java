package org.example.matchingtests;

import org.example.*;
import org.example.matchers.HistogramMatcher;
import org.example.matchers.HistVariant;
import org.example.utilities.AnalyticalTestBase;
import org.junit.jupiter.api.DisplayName;
import org.opencv.core.Mat;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Milestone 12 — Histogram Comparison analytical test.
 *
 * <p>Runs all 12 variants (CORREL / CHISQR / INTERSECT / BHATTACHARYYA
 * × base / CF_LOOSE / CF_TIGHT) across every reference × scene pair.
 * All shared infrastructure lives in {@link AnalyticalTestBase}.
 */
@DisplayName("Milestone 12 — Histogram Comparison")
class HistogramMatchingTest extends AnalyticalTestBase {

    // CIRCLE_FILLED = ordinal 9 → foreground palette slot 9%8=1 = RED (BGR 0,0,220)
    // A vivid saturated colour gives a strong H-S histogram signal, making this the
    // best debug reference for demonstrating histogram comparison's real behaviour.
    // (CIRCLE_OUTLINE is ordinal 0 = white → zero HSV saturation → degenerate histogram)
    private static final boolean     DEBUG     = false;
    private static final ReferenceId DEBUG_REF = ReferenceId.CIRCLE_FILLED;
    private static final Path        OUT       = Paths.get("test_output", "histogram_matching");

    private static final ReferenceId[] REF_FILTER =
            {
                    ReferenceId.CIRCLE_OUTLINE,
                    ReferenceId.RECT_FILLED,
                    ReferenceId.HEXAGON_OUTLINE,
                    ReferenceId.GRID_4X4,
                    ReferenceId.CROSSHAIR,
                    ReferenceId.POLYLINE_CHEVRON,
                    ReferenceId.POLYLINE_PARALLELOGRAM,
                    ReferenceId.TEXT_HELLO,
                    ReferenceId.STAR_5_FILLED,
                    ReferenceId.COMPOUND_TRIANGLE_IN_CIRCLE
            };

    private static final Set<SceneVariant> SCENE_VARIANTS = Set.of(
            SceneVariant.SCALE_0_50,
            SceneVariant.SCALE_1_50,
            SceneVariant.SCALE0_75_ROT30,
            SceneVariant.ROT_90,
            SceneVariant.ROT_180,
            SceneVariant.NOISE_S25,
            SceneVariant.OFFSET_TOPLEFT,
            SceneVariant.CLEAN_BG_GRADIENT_H_COLOUR
    );

    private static final Set<String> SAVE = MatcherVariant.allNamesOf(HistVariant.class);

    private static final EnumSet<CfMode> CF_TIERS = EnumSet.allOf(CfMode.class);

    @Override protected String           tag()             { return "HM"; }
    @Override protected String           techniqueName()   { return "Histogram Comparison"; }
    @Override protected Path             outputDir()       { return OUT; }
    @Override protected boolean          debugMode()       { return DEBUG; }
    @Override protected ReferenceId      debugRef()        { return DEBUG_REF; }
    @Override protected Set<String>      saveVariants()    { return SAVE; }
    @Override protected ReferenceId[]    referenceFilter() { return REF_FILTER; }
    @Override protected EnumSet<CfMode>  cfTierFilter()    { return CF_TIERS; }

    @Override
    protected boolean sceneFilter(SceneEntry scene) {
        if (scene.category() == SceneCategory.D_NEGATIVE) return true;
        return SCENE_VARIANTS.stream().anyMatch(v -> v.matches(scene));
    }

    @Override
    protected List<AnalysisResult> runMatcher(ReferenceId refId, Mat refMat,
                                              SceneEntry scene, Set<String> saveVariants,
                                              Path outputDir) {
        return HistogramMatcher.match(refId, refMat, scene, saveVariants, outputDir);
    }
}


