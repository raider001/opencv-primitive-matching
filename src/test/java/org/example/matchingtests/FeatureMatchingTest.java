package org.example.matchingtests;

import org.example.*;
import org.example.matchers.FeatureMatcher;
import org.example.matchers.FeatureVariant;
import org.example.utilities.AnalyticalTestBase;
import org.junit.jupiter.api.DisplayName;
import org.opencv.core.Mat;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Set;

/**
 * Milestone 8 — Feature Matching analytical test.
 *
 * <p>Runs all available feature detector variants (SIFT, ORB, AKAZE, BRISK, KAZE)
 * each in base, CF_LOOSE, and CF_TIGHT modes.  All shared infrastructure lives in
 * {@link AnalyticalTestBase}.
 */
@DisplayName("Milestone 8 — Feature Matching")
class FeatureMatchingTest extends AnalyticalTestBase {

    private static final boolean     DEBUG     = false;
    private static final ReferenceId DEBUG_REF = ReferenceId.CIRCLE_OUTLINE;
    private static final Path        OUT       = Paths.get("test_output", "feature_matching");

    private static final Set<String> SAVE = MatcherVariant.allNamesOf(FeatureVariant.class);

    @Override protected String      tag()           { return "FM"; }
    @Override protected String      techniqueName() { return "Feature Matching"; }
    @Override protected Path        outputDir()     { return OUT; }
    @Override protected boolean     debugMode()     { return DEBUG; }
    @Override protected ReferenceId debugRef()      { return DEBUG_REF; }
    @Override protected Set<String> saveVariants()  { return SAVE; }

    @Override
    protected List<AnalysisResult> runMatcher(ReferenceId refId, Mat refMat,
                                              SceneEntry scene, Set<String> saveVariants,
                                              Path outputDir) {
        return FeatureMatcher.match(refId, refMat, scene, saveVariants, outputDir);
    }
}
