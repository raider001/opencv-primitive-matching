package org.example;

import org.example.matchers.MorphologyAnalyzer;
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

    private static final Set<String> SAVE = Set.of(
            MorphologyAnalyzer.VAR_POLY,     MorphologyAnalyzer.VAR_POLY     + "_CF_LOOSE", MorphologyAnalyzer.VAR_POLY     + "_CF_TIGHT",
            MorphologyAnalyzer.VAR_CIRC,     MorphologyAnalyzer.VAR_CIRC     + "_CF_LOOSE", MorphologyAnalyzer.VAR_CIRC     + "_CF_TIGHT",
            MorphologyAnalyzer.VAR_COMBINED, MorphologyAnalyzer.VAR_COMBINED + "_CF_LOOSE", MorphologyAnalyzer.VAR_COMBINED + "_CF_TIGHT"
    );

    @Override protected String      tag()           { return "MA"; }
    @Override protected String      techniqueName() { return "Morphology Analysis"; }
    @Override protected Path        outputDir()     { return OUT; }
    @Override protected boolean     debugMode()     { return DEBUG; }
    @Override protected ReferenceId debugRef()      { return DEBUG_REF; }
    @Override protected Set<String> saveVariants()  { return SAVE; }

    @Override
    protected List<AnalysisResult> runMatcher(ReferenceId refId, Mat refMat,
                                               SceneEntry scene, Set<String> saveVariants,
                                               Path outputDir) {
        return MorphologyAnalyzer.match(refId, refMat, scene, saveVariants, outputDir);
    }
}

