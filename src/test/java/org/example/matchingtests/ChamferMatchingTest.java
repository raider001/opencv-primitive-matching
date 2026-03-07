package org.example.matchingtests;

import org.example.*;
import org.example.matchers.ChamferMatcher;
import org.example.matchers.ChamferVariant;
import org.example.utilities.AnalyticalTestBase;
import org.junit.jupiter.api.DisplayName;
import org.opencv.core.Mat;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Milestone 18 — Chamfer Distance Matching analytical test.
 *
 * <p>Runs all 6 Chamfer variants (L1 / L2 distance transform x base / CF_LOOSE / CF_TIGHT)
 * across a curated subset of scenes.  All shared infrastructure lives in
 * {@link AnalyticalTestBase}.
 *
 * <h2>What to look for</h2>
 * <ul>
 *   <li>A_CLEAN — high score; the reference edge map aligns well onto the scene's
 *       distance-transform field</li>
 *   <li>B_TRANSFORMED/occlusion — Chamfer tolerates partial matches much better than
 *       pixel diff because missing edge points contribute a bounded distance rather
 *       than a hard mismatch</li>
 *   <li>D_NEGATIVE — CF variants greatly reduce false positives from background edges
 *       that happen to share the colour of the reference</li>
 *   <li>L1 vs L2 — L2 penalises outlier distances more strongly; expect L1 to be
 *       more robust on heavily degraded or occluded scenes</li>
 * </ul>
 */
@DisplayName("Milestone 18 — Chamfer Distance Matching")
class ChamferMatchingTest extends AnalyticalTestBase {

    private static final boolean     DEBUG     = false;
    private static final ReferenceId DEBUG_REF = ReferenceId.CIRCLE_OUTLINE;
    private static final Path        OUT       = Paths.get("test_output", "chamfer_matching");

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
        SceneVariant.ROT_90,
        SceneVariant.ROT_180,
        SceneVariant.NOISE_S25,
        SceneVariant.CLEAN_BG_GRADIENT_H_COLOUR,
        SceneVariant.CLEAN_BG_RANDOM_MIXED
    );

    private static final Set<String>     SAVE     = MatcherVariant.allNamesOf(ChamferVariant.class);
    private static final EnumSet<CfMode> CF_TIERS = EnumSet.allOf(CfMode.class);

    @Override protected String           tag()             { return "CH"; }
    @Override protected String           techniqueName()   { return "Chamfer Distance Matching"; }
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
        return ChamferMatcher.match(refId, refMat, scene, saveVariants, outputDir);
    }
}

