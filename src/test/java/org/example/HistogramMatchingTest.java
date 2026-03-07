package org.example;

import org.example.matchers.HistogramMatcher;
import org.junit.jupiter.api.DisplayName;
import org.opencv.core.Mat;

import java.nio.file.Path;
import java.nio.file.Paths;
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
            HistogramMatcher.VAR_CORREL,    HistogramMatcher.VAR_CORREL    + "_CF_LOOSE", HistogramMatcher.VAR_CORREL    + "_CF_TIGHT",
            HistogramMatcher.VAR_CHISQR,    HistogramMatcher.VAR_CHISQR    + "_CF_LOOSE", HistogramMatcher.VAR_CHISQR    + "_CF_TIGHT",
            HistogramMatcher.VAR_INTERSECT, HistogramMatcher.VAR_INTERSECT + "_CF_LOOSE", HistogramMatcher.VAR_INTERSECT + "_CF_TIGHT",
            HistogramMatcher.VAR_BHATTA,    HistogramMatcher.VAR_BHATTA    + "_CF_LOOSE", HistogramMatcher.VAR_BHATTA    + "_CF_TIGHT"
    );

    @Override protected String        tag()             { return "HM"; }
    @Override protected String        techniqueName()   { return "Histogram Comparison"; }
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
        return HistogramMatcher.match(refId, refMat, scene, saveVariants, outputDir);
    }
}


