package org.example.matchingtests;

import org.example.*;
import org.example.matchers.HoughDetector;
import org.example.matchers.HoughVariant;
import org.example.utilities.AnalyticalTestBase;
import org.junit.jupiter.api.DisplayName;
import org.opencv.core.Mat;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Set;

/**
 * Milestone 10 — Hough Transforms analytical test.
 *
 * <p>Runs all 6 Hough variants (HoughLinesP + HoughCircles × base/CF_LOOSE/CF_TIGHT)
 * across every reference × scene pair.
 * All shared infrastructure lives in {@link AnalyticalTestBase}.
 */
@DisplayName("Milestone 10 — Hough Transforms")
class HoughDetectorTest extends AnalyticalTestBase {

    private static final boolean     DEBUG     = false;
    private static final ReferenceId DEBUG_REF = ReferenceId.CIRCLE_OUTLINE;
    private static final Path        OUT       = Paths.get("test_output", "hough_transforms");

    private static final ReferenceId[] REF_FILTER = {
        ReferenceId.CIRCLE_OUTLINE,
        ReferenceId.RECT_FILLED,
        ReferenceId.HEXAGON_OUTLINE,
        ReferenceId.GRID_4X4,
        ReferenceId.TEXT_A,
    };

    private static final Set<SceneVariant> SCENE_VARIANTS = Set.of(
        SceneVariant.CLEAN_BG_NOISE_LIGHT,
        SceneVariant.CLEAN_BG_GRADIENT_H_COLOUR,
        SceneVariant.CLEAN_BG_RANDOM_MIXED,
        SceneVariant.ROT_45, SceneVariant.ROT_90, SceneVariant.ROT_180,
        SceneVariant.SCALE_0_50
    );

    private static final Set<String> SAVE = MatcherVariant.allNamesOf(HoughVariant.class);

    @Override protected String        tag()             { return "HT"; }
    @Override protected String        techniqueName()   { return "Hough Transforms"; }
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
        return HoughDetector.match(refId, refMat, scene, saveVariants, outputDir);
    }
}
