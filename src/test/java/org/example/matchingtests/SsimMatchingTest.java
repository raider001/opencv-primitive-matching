package org.example.matchingtests;

import org.example.*;
import org.example.matchers.SsimMatcher;
import org.example.matchers.SsimVariant;
import org.example.utilities.AnalyticalTestBase;
import org.junit.jupiter.api.DisplayName;
import org.opencv.core.Mat;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Milestone 17 — Structural Similarity (SSIM) analytical test.
 *
 * <p>Runs all 3 SSIM variants (base / CF_LOOSE / CF_TIGHT) across a curated subset
 * of scenes.  All shared infrastructure lives in {@link AnalyticalTestBase}.
 *
 * <h2>What to look for</h2>
 * <ul>
 *   <li>A_CLEAN — near-100% (SSIM ≈ 1 on exact structural match)</li>
 *   <li>B_TRANSFORMED — drops sharply under scale/rotation (not invariant) but
 *       degrades more gracefully than raw pixel diff on small noise/blur</li>
 *   <li>C_DEGRADED — SSIM's luminance/contrast/structure decomposition makes it
 *       more tolerant of global brightness shifts than TM or Pixel Diff</li>
 *   <li>D_NEGATIVE — lower false-positive rate than Pixel Diff; CF variants push
 *       it further toward 0%</li>
 * </ul>
 */
@DisplayName("Milestone 17 — SSIM (Structural Similarity)")
class SsimMatchingTest extends AnalyticalTestBase {

    private static final boolean     DEBUG     = false;
    private static final ReferenceId DEBUG_REF = ReferenceId.CIRCLE_OUTLINE;
    private static final Path        OUT       = Paths.get("test_output", "ssim_matching");

    private static final ReferenceId[] REF_FILTER = {
        ReferenceId.CIRCLE_OUTLINE,
        ReferenceId.RECT_FILLED,
        ReferenceId.HEXAGON_OUTLINE,
        ReferenceId.GRID_4X4,
        ReferenceId.CROSSHAIR,
        ReferenceId.POLYLINE_CHEVRON,
        ReferenceId.POLYLINE_PARALLELOGRAM,
        ReferenceId.TEXT_HELLO,
    };

    private static final Set<SceneVariant> SCENE_VARIANTS = Set.of(
        SceneVariant.SCALE_0_50,
        SceneVariant.SCALE_1_50,
        SceneVariant.SCALE0_75_ROT30,
        SceneVariant.NOISE_S25,
        SceneVariant.CLEAN_BG_GRADIENT_H_COLOUR,
        SceneVariant.CLEAN_BG_NOISE_LIGHT
    );

    private static final Set<String>     SAVE     = MatcherVariant.allNamesOf(SsimVariant.class);
    private static final EnumSet<CfMode> CF_TIERS = EnumSet.allOf(CfMode.class);

    @Override protected String           tag()             { return "SSIM"; }
    @Override protected String           techniqueName()   { return "SSIM (Structural Similarity)"; }
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
        return SsimMatcher.match(refId, refMat, scene, saveVariants, outputDir);
    }
}

