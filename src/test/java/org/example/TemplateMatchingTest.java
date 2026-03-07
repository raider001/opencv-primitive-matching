package org.example;

import org.example.matchers.TemplateMatcher;
import org.junit.jupiter.api.DisplayName;
import org.opencv.core.Mat;

import java.nio.file.Path;
import java.nio.file.Paths;
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

    private static final ReferenceId[] REF_FILTER = {
        ReferenceId.CIRCLE_OUTLINE,   // curved outline
        ReferenceId.RECT_FILLED,      // filled solid block
        ReferenceId.HEXAGON_OUTLINE,  // regular polygon
        ReferenceId.GRID_4X4,         // repeating pattern
        ReferenceId.TEXT_A,           // text glyph
    };

    /**
     * Scene variant labels that are included in the test run.
     * Category D negatives are always included regardless of this set.
     * Non-REF_FILTER primary references are filtered by {@link AnalyticalTestBase}
     * before this hook is invoked.
     */
    private static final Set<String> SCENE_VARIANTS = Set.of(
        // A_CLEAN — one per background type
//        "clean_bg_solid_black",
        "clean_bg_noise_light",
        "clean_bg_gradient_h_colour",
        "clean_bg_random_mixed",

        // B_TRANSFORMED — rotations, scales, offsets, combos
        "rot_45",
        "rot_90",
        "rot_180",
        "scale_0.50"   // JSON stores "scale_0.50" (dot), not underscore
//        "scale_1.50",
//        "scale0_75_rot30",
//        "scale1_5_rot45",
//        "offset_topleft",
//        "offset_botright"

        // C_DEGRADED — one per degradation type
//        "noise_s10",
//        "noise_s25",
//        "blur_5x5",
//        "contrast_40pct",
//        "occ_25pct",
//        "occ_50pct",
//        "hue_shift_40"
    );

    private static final Set<String> SAVE = Set.of(
            "TM_SQDIFF",          "TM_SQDIFF_CF_LOOSE",          "TM_SQDIFF_CF_TIGHT",
            "TM_SQDIFF_NORMED",   "TM_SQDIFF_NORMED_CF_LOOSE",   "TM_SQDIFF_NORMED_CF_TIGHT",
            "TM_CCORR",           "TM_CCORR_CF_LOOSE",           "TM_CCORR_CF_TIGHT",
            "TM_CCORR_NORMED",    "TM_CCORR_NORMED_CF_LOOSE",    "TM_CCORR_NORMED_CF_TIGHT",
            "TM_CCOEFF",          "TM_CCOEFF_CF_LOOSE",          "TM_CCOEFF_CF_TIGHT",
            "TM_CCOEFF_NORMED",   "TM_CCOEFF_NORMED_CF_LOOSE",   "TM_CCOEFF_NORMED_CF_TIGHT"
    );

    @Override protected String        tag()             { return "TM"; }
    @Override protected String        techniqueName()   { return "Template Matching"; }
    @Override protected Path          outputDir()       { return OUT; }
    @Override protected boolean       debugMode()       { return DEBUG; }
    @Override protected ReferenceId   debugRef()        { return DEBUG_REF; }
    @Override protected Set<String>   saveVariants()    { return SAVE; }
    @Override protected ReferenceId[] referenceFilter() { return REF_FILTER; }

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
        return SCENE_VARIANTS.contains(scene.variantLabel());
    }

    @Override
    protected List<AnalysisResult> runMatcher(ReferenceId refId, Mat refMat,
                                               SceneEntry scene, Set<String> saveVariants,
                                               Path outputDir) {
        return TemplateMatcher.match(refId, refMat, scene, saveVariants, outputDir);
    }
}
