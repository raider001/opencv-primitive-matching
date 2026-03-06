package org.example;

import org.example.matchers.GeneralizedHoughDetector;
import org.junit.jupiter.api.DisplayName;
import org.opencv.core.Mat;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Set;

/**
 * Milestone 11 — Generalized Hough Transform analytical test.
 *
 * <p>Runs all 6 variants (GeneralizedHoughBallard + GeneralizedHoughGuil
 * × base / CF_LOOSE / CF_TIGHT) across every reference × scene pair.
 * All shared infrastructure lives in {@link AnalyticalTestBase}.
 */
@DisplayName("Milestone 11 — Generalized Hough Transform")
class GeneralizedHoughTest extends AnalyticalTestBase {

    private static final boolean     DEBUG     = true;
    private static final ReferenceId DEBUG_REF = ReferenceId.CIRCLE_OUTLINE;
    private static final Path        OUT       = Paths.get("test_output", "generalized_hough");

    private static final Set<String> SAVE = Set.of(
            GeneralizedHoughDetector.VAR_BALLARD,
            GeneralizedHoughDetector.VAR_BALLARD + "_CF_LOOSE",
            GeneralizedHoughDetector.VAR_BALLARD + "_CF_TIGHT",
            GeneralizedHoughDetector.VAR_GUIL,
            GeneralizedHoughDetector.VAR_GUIL    + "_CF_LOOSE",
            GeneralizedHoughDetector.VAR_GUIL    + "_CF_TIGHT"
    );

    @Override protected String      tag()           { return "GH"; }
    @Override protected String      techniqueName() { return "Generalized Hough Transform"; }
    @Override protected Path        outputDir()     { return OUT; }
    @Override protected boolean     debugMode()     { return DEBUG; }
    @Override protected ReferenceId debugRef()      { return DEBUG_REF; }
    @Override protected Set<String> saveVariants()  { return SAVE; }

    @Override
    protected List<AnalysisResult> runMatcher(ReferenceId refId, Mat refMat,
                                               SceneEntry scene, Set<String> saveVariants,
                                               Path outputDir) {
        return GeneralizedHoughDetector.match(refId, refMat, scene, saveVariants, outputDir);
    }
}

