package org.example;

import org.example.matchers.PixelDiffMatcher;
import org.example.utilities.AnalyticalTestBase;
import org.junit.jupiter.api.DisplayName;
import org.opencv.core.Mat;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Set;

/**
 * Milestone 16 — Pixel Diff (baseline) analytical test.
 *
 * <p>Runs all 3 pixel-diff variants (base / CF_LOOSE / CF_TIGHT) across every
 * reference × scene pair.  All shared infrastructure lives in {@link AnalyticalTestBase}.
 */
@DisplayName("Milestone 16 — Pixel Diff (Baseline)")
class PixelDiffTest extends AnalyticalTestBase {

    private static final boolean     DEBUG     = false;
    private static final ReferenceId DEBUG_REF = ReferenceId.CIRCLE_OUTLINE;
    private static final Path        OUT       = Paths.get("test_output", "pixel_diff");

    private static final Set<String> SAVE = Set.of(
            PixelDiffMatcher.VAR_BASE,
            PixelDiffMatcher.VAR_LOOSE,
            PixelDiffMatcher.VAR_TIGHT
    );

    @Override protected String      tag()           { return "PD"; }
    @Override protected String      techniqueName() { return "Pixel Diff (Baseline)"; }
    @Override protected Path        outputDir()     { return OUT; }
    @Override protected boolean     debugMode()     { return DEBUG; }
    @Override protected ReferenceId debugRef()      { return DEBUG_REF; }
    @Override protected Set<String> saveVariants()  { return SAVE; }

    @Override
    protected List<AnalysisResult> runMatcher(ReferenceId refId, Mat refMat,
                                               SceneEntry scene, Set<String> saveVariants,
                                               Path outputDir) {
        return PixelDiffMatcher.match(refId, refMat, scene, saveVariants, outputDir);
    }
}

