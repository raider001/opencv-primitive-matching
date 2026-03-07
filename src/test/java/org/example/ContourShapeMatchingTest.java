package org.example;

import org.example.matchers.ContourShapeMatcher;
import org.example.utilities.AnalyticalTestBase;
import org.junit.jupiter.api.DisplayName;
import org.opencv.core.Mat;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Set;

/**
 * Milestone 9 — Contour Shape Matching (Hu Moments) analytical test.
 *
 * <p>Runs all 9 contour-shape-matching variants (3 base CONTOURS_MATCH_I1/I2/I3 ×
 * base/CF_LOOSE/CF_TIGHT) across every reference × scene pair.
 * All shared infrastructure lives in {@link AnalyticalTestBase}.
 */
@DisplayName("Milestone 9 — Contour Shape Matching (Hu Moments)")
class ContourShapeMatchingTest extends AnalyticalTestBase {

    private static final boolean     DEBUG     = false;
    private static final ReferenceId DEBUG_REF = ReferenceId.CIRCLE_OUTLINE;
    private static final Path        OUT       = Paths.get("test_output", "contour_shape_matching");

    private static final ReferenceId[] REF_FILTER = {
        ReferenceId.CIRCLE_OUTLINE,
        ReferenceId.RECT_FILLED,
        ReferenceId.HEXAGON_OUTLINE,
        ReferenceId.GRID_4X4,
        ReferenceId.TEXT_A,
    };

    private static final Set<String> SCENE_VARIANTS = Set.of(
        "clean_bg_noise_light",
        "clean_bg_gradient_h_colour",
        "clean_bg_random_mixed",
        "rot_45", "rot_90", "rot_180",
        "scale_0.50"
    );

    private static final Set<String> SAVE = Set.of(
            "CONTOURS_MATCH_I1", "CONTOURS_MATCH_I1_CF_LOOSE", "CONTOURS_MATCH_I1_CF_TIGHT",
            "CONTOURS_MATCH_I2", "CONTOURS_MATCH_I2_CF_LOOSE", "CONTOURS_MATCH_I2_CF_TIGHT",
            "CONTOURS_MATCH_I3", "CONTOURS_MATCH_I3_CF_LOOSE", "CONTOURS_MATCH_I3_CF_TIGHT",
            ContourShapeMatcher.VAR_CF1_LOOSE, ContourShapeMatcher.VAR_CF1_TIGHT
    );

    @Override protected String        tag()             { return "CSM"; }
    @Override protected String        techniqueName()   { return "Contour Shape Matching (Hu Moments)"; }
    @Override protected Path          outputDir()       { return OUT; }
    @Override protected boolean       debugMode()       { return DEBUG; }
    @Override protected ReferenceId   debugRef()        { return DEBUG_REF; }
    @Override protected Set<String>   saveVariants()    { return SAVE; }
    @Override protected ReferenceId[] referenceFilter() { return REF_FILTER; }

    @Override
    protected boolean sceneFilter(SceneEntry scene) {
        if (scene.category() == SceneCategory.D_NEGATIVE) return true;
        return SCENE_VARIANTS.contains(scene.variantLabel());
    }

    @Override
    protected List<AnalysisResult> runMatcher(ReferenceId refId, Mat refMat,
                                               SceneEntry scene, Set<String> saveVariants,
                                               Path outputDir) {
        return ContourShapeMatcher.match(refId, refMat, scene, saveVariants, outputDir);
    }
}

