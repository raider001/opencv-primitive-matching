package org.example.matchingtests;

import org.example.*;
import org.example.matchers.TemplateMatcher;
import org.example.matchers.TmVariant;
import org.example.utilities.AnalyticalTestBase;
import org.junit.jupiter.api.DisplayName;
import org.opencv.core.Mat;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Milestone 7 — Template Matching analytical test.
 *
 * <p>Runs all 18 TM variants (6 base + 6 CF_LOOSE + 6 CF_TIGHT) across a curated
 * subset of scenes.  All shared infrastructure (catalogue build, parallel loop,
 * output writing, HTML report) lives in {@link AnalyticalTestBase}.
 *
 * <h2>Filters</h2>
 * <ul>
 *   <li><b>Reference filter</b> — 5 references covering distinct shape families.</li>
 *   <li><b>Scene filter</b> — one representative variant per difficulty tier:
 *     <ul>
 *       <li>A_CLEAN:       solid-black bg, noise bg, gradient bg</li>
 *       <li>B_TRANSFORMED: rot 45°, rot 90°, scale ×0.5, scale ×1.5,
 *                          scale+rot combo, offset (top-left, bot-right)</li>
 *       <li>C_DEGRADED:    noise s10, noise s25, blur, contrast, occlusion 25%,
 *                          occlusion 50%, hue shift</li>
 *       <li>D_NEGATIVE:    all (always included)</li>
 *     </ul>
 *   </li>
 * </ul>
 */
@DisplayName("Milestone 7 — Template Matching")
class TemplateMatchingTest extends AnalyticalTestBase {

    private static final boolean     DEBUG      = false;
    private static final ReferenceId DEBUG_REF  = ReferenceId.CIRCLE_OUTLINE;
    private static final Path        OUT        = Paths.get("test_output", "template_matching");

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

    private static final Set<String> SAVE = MatcherVariant.allNamesOf(TmVariant.class);

    private static final EnumSet<CfMode> CF_TIERS = EnumSet.allOf(CfMode.class);

    @Override protected String           tag()             { return "TM"; }
    @Override protected String           techniqueName()   { return "Template Matching"; }
    @Override protected Path             outputDir()       { return OUT; }
    @Override protected boolean          debugMode()       { return DEBUG; }
    @Override protected ReferenceId      debugRef()        { return DEBUG_REF; }
    @Override protected Set<String>      saveVariants()    { return SAVE; }
    @Override protected ReferenceId[]    referenceFilter() { return REF_FILTER; }
    @Override protected EnumSet<CfMode>  cfTierFilter()    { return CF_TIERS; }

    /**
     * Accept only scenes whose variant label is in {@link #SCENE_VARIANTS},
     * plus all Category D negatives.
     *
     * <p>Note: A/B/C scenes whose {@code primaryReferenceId} is not in
     * {@link #REF_FILTER} are already excluded by {@link AnalyticalTestBase}
     * before this hook fires — so only scenes that actually contain one of
     * our 5 queried shapes reach this filter.
     */
    @Override
    protected boolean sceneFilter(SceneEntry scene) {
        if (scene.category() == SceneCategory.D_NEGATIVE) return true;
        return SCENE_VARIANTS.stream().anyMatch(v -> v.matches(scene));
    }

    @Override
    protected List<AnalysisResult> runMatcher(ReferenceId refId, Mat refMat,
                                              SceneEntry scene, Set<String> saveVariants,
                                              Path outputDir) {
        return TemplateMatcher.match(refId, refMat, scene, saveVariants, outputDir);
    }
}
