package org.example;

import org.example.matchers.TemplateMatcher;
import org.junit.jupiter.api.DisplayName;
import org.opencv.core.Mat;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Set;

/**
 * Milestone 7 — Template Matching analytical test.
 *
 * <p>Runs all 18 TM variants (6 base + 6 CF_LOOSE + 6 CF_TIGHT) across every
 * reference × scene pair.  All shared infrastructure (catalogue build, parallel
 * loop, output writing, HTML report) lives in {@link AnalyticalTestBase}.
 */
@DisplayName("Milestone 7 — Template Matching")
class TemplateMatchingTest extends AnalyticalTestBase {

    private static final boolean     DEBUG      = true;
    private static final ReferenceId DEBUG_REF  = ReferenceId.CIRCLE_OUTLINE;
    private static final Path        OUT        = Paths.get("test_output", "template_matching");

    private static final Set<String> SAVE = Set.of(
            "TM_SQDIFF",          "TM_SQDIFF_CF_LOOSE",          "TM_SQDIFF_CF_TIGHT",
            "TM_SQDIFF_NORMED",   "TM_SQDIFF_NORMED_CF_LOOSE",   "TM_SQDIFF_NORMED_CF_TIGHT",
            "TM_CCORR",           "TM_CCORR_CF_LOOSE",           "TM_CCORR_CF_TIGHT",
            "TM_CCORR_NORMED",    "TM_CCORR_NORMED_CF_LOOSE",    "TM_CCORR_NORMED_CF_TIGHT",
            "TM_CCOEFF",          "TM_CCOEFF_CF_LOOSE",          "TM_CCOEFF_CF_TIGHT",
            "TM_CCOEFF_NORMED",   "TM_CCOEFF_NORMED_CF_LOOSE",   "TM_CCOEFF_NORMED_CF_TIGHT"
    );

    @Override protected String      tag()           { return "TM"; }
    @Override protected String      techniqueName() { return "Template Matching"; }
    @Override protected Path        outputDir()     { return OUT; }
    @Override protected boolean     debugMode()     { return DEBUG; }
    @Override protected ReferenceId debugRef()      { return DEBUG_REF; }
    @Override protected Set<String> saveVariants()  { return SAVE; }

    @Override
    protected List<AnalysisResult> runMatcher(ReferenceId refId, Mat refMat,
                                               SceneEntry scene, Set<String> saveVariants,
                                               Path outputDir) {
        return TemplateMatcher.match(refId, refMat, scene, saveVariants, outputDir);
    }
}
