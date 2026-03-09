package org.example.slowmatchingtests;

import org.example.*;
import org.example.analytics.AnalysisResult;
import org.example.factories.ReferenceId;
import org.example.matchers.VectorMatcher;
import org.example.matchers.VectorVariant;
import org.example.scene.SceneCategory;
import org.example.scene.SceneEntry;
import org.example.scene.SceneVariant;
import org.example.utilities.AnalyticalTestBase;
import org.junit.jupiter.api.DisplayName;
import org.opencv.core.Mat;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Set;

/**
 * Vector Matching — full analytical test (feeds the benchmark report).
 *
 * <p>Runs all 9 variants (3 epsilon levels × 3 CF modes) across a curated
 * set of references and scene variants.  Results are written to
 * {@code test_output/vector_matching/report.html} which the unified
 * benchmark report collates.
 *
 * <p>All shared infrastructure (catalogue loading, parallel dispatch,
 * HTML report writing, verdict scoring) lives in {@link AnalyticalTestBase}.
 *
 * <p>To keep runtime reasonable the reference and scene filters below are
 * intentionally tighter than the full catalogue.  Widen them once the
 * matcher is tuned to handle more variants.
 */
@DisplayName("Vector Matching — Analytical")
class VectorMatchingAnalyticalTest extends AnalyticalTestBase {

    private static final boolean     DEBUG     = false;
    private static final ReferenceId DEBUG_REF = ReferenceId.CIRCLE_OUTLINE;
    private static final Path        OUT       = Paths.get("test_output", "vector_matching");

    /**
     * Initial small reference set — shapes the vector matcher is designed to
     * identify.  Expand as tuning improves results.
     */
    private static final ReferenceId[] REF_FILTER = {
            ReferenceId.CIRCLE_FILLED,
            ReferenceId.CIRCLE_OUTLINE,
            ReferenceId.RECT_FILLED,
            ReferenceId.RECT_OUTLINE,
            ReferenceId.TRIANGLE_FILLED,
            ReferenceId.HEXAGON_FILLED,
            ReferenceId.HEXAGON_OUTLINE,
            ReferenceId.STAR_5_FILLED,
            ReferenceId.POLYLINE_CHEVRON,
            ReferenceId.COMPOUND_TRIANGLE_IN_CIRCLE,
    };

    /**
     * Scene variants to run.  Starts conservative (clean + key transforms)
     * to keep the first run fast.  Expand alongside ref filter as needed.
     */
    private static final Set<SceneVariant> SCENE_VARIANTS = Set.of(
            SceneVariant.CLEAN,
            SceneVariant.SCALE_0_50,
            SceneVariant.SCALE_0_75,
            SceneVariant.SCALE_1_50,
            SceneVariant.ROT_45,
            SceneVariant.ROT_90,
            SceneVariant.OFFSET_TOPLEFT,
            SceneVariant.OFFSET_BOTRIGHT
    );

    /** Save annotated images for every variant so they appear in the report. */
    private static final Set<String> SAVE = MatcherVariant.allNamesOf(VectorVariant.class);

    @Override protected String        tag()           { return "VM"; }
    @Override protected String        techniqueName() { return "Vector Matching"; }
    @Override protected Path          outputDir()     { return OUT; }
    @Override protected boolean       debugMode()     { return DEBUG; }
    @Override protected ReferenceId   debugRef()      { return DEBUG_REF; }
    @Override protected Set<String>   saveVariants()  { return SAVE; }
    @Override protected ReferenceId[] referenceFilter() { return REF_FILTER; }

    @Override
    protected boolean sceneFilter(SceneEntry scene) {
        if (scene.category() == SceneCategory.D_NEGATIVE) return true;
        return SCENE_VARIANTS.stream().anyMatch(v -> v.matches(scene));
    }

    @Override
    protected List<AnalysisResult> runMatcher(ReferenceId refId, Mat refMat,
                                              SceneEntry scene,
                                              Set<String> saveVariants,
                                              Path outputDir) {
        return VectorMatcher.match(refId, refMat, scene, saveVariants, outputDir);
    }
}




