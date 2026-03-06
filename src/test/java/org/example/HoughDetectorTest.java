package org.example;

import org.example.matchers.HoughDetector;
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

    private static final boolean     DEBUG     = true;
    private static final ReferenceId DEBUG_REF = ReferenceId.CIRCLE_OUTLINE;
    private static final Path        OUT       = Paths.get("test_output", "hough_transforms");

    private static final Set<String> SAVE = Set.of(
            HoughDetector.VAR_LINES,
            HoughDetector.VAR_LINES   + "_CF_LOOSE",
            HoughDetector.VAR_LINES   + "_CF_TIGHT",
            HoughDetector.VAR_CIRCLES,
            HoughDetector.VAR_CIRCLES + "_CF_LOOSE",
            HoughDetector.VAR_CIRCLES + "_CF_TIGHT"
    );

    @Override protected String      tag()           { return "HT"; }
    @Override protected String      techniqueName() { return "Hough Transforms"; }
    @Override protected Path        outputDir()     { return OUT; }
    @Override protected boolean     debugMode()     { return DEBUG; }
    @Override protected ReferenceId debugRef()      { return DEBUG_REF; }
    @Override protected Set<String> saveVariants()  { return SAVE; }

    @Override
    protected List<AnalysisResult> runMatcher(ReferenceId refId, Mat refMat,
                                               SceneEntry scene, Set<String> saveVariants,
                                               Path outputDir) {
        return HoughDetector.match(refId, refMat, scene, saveVariants, outputDir);
    }
}

