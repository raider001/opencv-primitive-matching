package org.example.matchingtests;

import org.example.*;
import org.example.matchers.MorphologyAnalyzer;
import org.example.utilities.AnalyticalTestBase;
import org.junit.jupiter.api.DisplayName;
import org.opencv.core.Mat;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Set;

/**
 * Milestone 14 — Morphology Analysis analytical test.
 *
 * <p>Runs all 9 morphology variants (3 base × base/CF_LOOSE/CF_TIGHT) across every
 * reference × scene pair.  All shared infrastructure lives in {@link AnalyticalTestBase}.
 */
@DisplayName("Milestone 14 — Morphology Analysis")
class MorphologyAnalysisTest extends AnalyticalTestBase {

    private static final boolean     DEBUG     = false;
    private static final ReferenceId DEBUG_REF = ReferenceId.CIRCLE_OUTLINE;
    private static final Path        OUT       = Paths.get("test_output", "morphology_analysis");

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

    private static final Set<String> SAVE = Set.of(
            MorphologyAnalyzer.VAR_POLY,     MorphologyAnalyzer.VAR_POLY     + "_CF_LOOSE", MorphologyAnalyzer.VAR_POLY     + "_CF_TIGHT",
            MorphologyAnalyzer.VAR_CIRC,     MorphologyAnalyzer.VAR_CIRC     + "_CF_LOOSE", MorphologyAnalyzer.VAR_CIRC     + "_CF_TIGHT",
            MorphologyAnalyzer.VAR_COMBINED, MorphologyAnalyzer.VAR_COMBINED + "_CF_LOOSE", MorphologyAnalyzer.VAR_COMBINED + "_CF_TIGHT"
    );

    @Override protected String        tag()             { return "MA"; }
    @Override protected String        techniqueName()   { return "Morphology Analysis"; }
    @Override protected Path          outputDir()       { return OUT; }
    @Override protected boolean       debugMode()       { return DEBUG; }
    @Override protected ReferenceId   debugRef()        { return DEBUG_REF; }
    @Override protected Set<String>   saveVariants()    { return SAVE; }
    @Override protected ReferenceId[] referenceFilter() { return REF_FILTER; }

    @Override
    protected boolean sceneFilter(SceneEntry scene) {
        if (scene.category() == SceneCategory.D_NEGATIVE) return true;
        return SCENE_VARIANTS.stream().anyMatch(v -> v.matches(scene));
    }

    @Override
    protected List<AnalysisResult> runMatcher(ReferenceId refId, Mat refMat,
                                              SceneEntry scene, Set<String> saveVariants,
                                              Path outputDir) {
        return MorphologyAnalyzer.match(refId, refMat, scene, saveVariants, outputDir);
    }
}

