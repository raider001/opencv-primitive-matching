package org.example.matchingtests;

import org.example.*;
import org.example.matchers.FourierShapeMatcher;
import org.example.matchers.FourierShapeVariant;
import org.example.utilities.AnalyticalTestBase;
import org.junit.jupiter.api.DisplayName;
import org.opencv.core.Mat;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Milestone 19 — Fourier Shape Descriptor matching analytical test.
 *
 * <p>Runs all 3 Fourier Shape variants (base / CF_LOOSE / CF_TIGHT) across a curated
 * subset of scenes.  All shared infrastructure lives in {@link AnalyticalTestBase}.
 *
 * <h2>What to look for</h2>
 * <ul>
 *   <li>Rotation variants (ROT_45 / ROT_90 / ROT_180) — score should be near-identical
 *       to the clean scene score, since the magnitude spectrum is rotation-invariant</li>
 *   <li>Scale variants (SCALE_0_50 / SCALE_1_50) — normalisation by the k=1 coefficient
 *       should absorb most of the scale change; expect moderate scores</li>
 *   <li>Occlusion (NOISE_S25) — partial contours shift the lower Fourier coefficients
 *       less than higher ones; lower-frequency descriptors are most robust</li>
 *   <li>D_NEGATIVE — shapes with very different Fourier profiles should score near 0</li>
 *   <li>CF variants — by restricting contour extraction to colour-matching pixels,
 *       false contour candidates from background clutter are suppressed</li>
 * </ul>
 */
@DisplayName("Milestone 19 — Fourier Shape Descriptors")
class FourierShapeMatchingTest extends AnalyticalTestBase {

    private static final boolean     DEBUG     = false;
    private static final ReferenceId DEBUG_REF = ReferenceId.CIRCLE_OUTLINE;
    private static final Path        OUT       = Paths.get("test_output", "fourier_shape_matching");

    private static final ReferenceId[] REF_FILTER = {
        ReferenceId.CIRCLE_OUTLINE,
        ReferenceId.RECT_FILLED,
        ReferenceId.HEXAGON_OUTLINE,
        ReferenceId.GRID_4X4,
        ReferenceId.CROSSHAIR,
        ReferenceId.POLYLINE_CHEVRON,
        ReferenceId.POLYLINE_PARALLELOGRAM,
        ReferenceId.TEXT_HELLO,
        ReferenceId.STAR_5_FILLED,
        ReferenceId.COMPOUND_TRIANGLE_IN_CIRCLE,
    };

    private static final Set<SceneVariant> SCENE_VARIANTS = Set.of(
        SceneVariant.SCALE_0_50,
        SceneVariant.SCALE_1_50,
        SceneVariant.SCALE0_75_ROT30,
        SceneVariant.ROT_45,
        SceneVariant.ROT_90,
        SceneVariant.ROT_180,
        SceneVariant.NOISE_S25,
        SceneVariant.CLEAN_BG_GRADIENT_H_COLOUR,
        SceneVariant.CLEAN_BG_RANDOM_MIXED
    );

    private static final Set<String>     SAVE     = MatcherVariant.allNamesOf(FourierShapeVariant.class);
    private static final EnumSet<CfMode> CF_TIERS = EnumSet.allOf(CfMode.class);

    @Override protected String           tag()             { return "FS"; }
    @Override protected String           techniqueName()   { return "Fourier Shape Descriptors"; }
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
        return FourierShapeMatcher.match(refId, refMat, scene, saveVariants, outputDir);
    }
}

