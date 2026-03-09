package org.example.slowmatchingtests;

import org.example.analytics.AnalysisResult;
import org.example.colour.CfMode;
import org.example.factories.ReferenceId;
import org.example.matchers.*;
import org.example.scene.SceneCategory;
import org.example.scene.SceneEntry;
import org.example.scene.SceneVariant;
import org.example.utilities.AnalyticalTestBase;
import org.junit.jupiter.api.DisplayName;
import org.opencv.core.Mat;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Stream;

/**
 * Milestone 15 — Colour-First Region Proposal analytical test.
 *
 * <p>Exercises the CF1 (Colour-First window search) variants across all 9 base
 * matching techniques in a single focused run.  For each technique the CF1 variants
 * are run alongside the best base variant and its CF_LOOSE counterpart so the
 * speedup and accuracy trade-off is immediately visible in the report.
 *
 * <h2>Variants exercised (18 CF1 + 9 base + 9 CF_LOOSE = 36 total)</h2>
 * <ul>
 *   <li>Template Matching — {@code TM_CCOEFF_NORMED} / {@code TM_CCOEFF_NORMED_CF_LOOSE} /
 *       {@code TM_CCOEFF_NORMED_CF1_LOOSE} / {@code TM_CCOEFF_NORMED_CF1_TIGHT}</li>
 *   <li>Feature Matching  — {@code SIFT} / {@code SIFT_CF_LOOSE} /
 *       {@code SIFT_CF1_LOOSE} / {@code SIFT_CF1_TIGHT}</li>
 *   <li>Contour/Hu Moments — {@code CONTOURS_MATCH_I1} / {@code CONTOURS_MATCH_I1_CF_LOOSE} /
 *       {@code CONTOURS_MATCH_I1_CF1_LOOSE} / {@code CONTOURS_MATCH_I1_CF1_TIGHT}</li>
 *   <li>Hough             — {@code HoughLinesP} / {@code HoughLinesP_CF_LOOSE} /
 *       {@code HoughLinesP_CF1_LOOSE} / {@code HoughLinesP_CF1_TIGHT}</li>
 *   <li>Gen. Hough        — {@code GeneralizedHoughBallard} / {@code GeneralizedHoughBallard_CF_LOOSE} /
 *       {@code GeneralizedHoughBallard_CF1_LOOSE} / {@code GeneralizedHoughBallard_CF1_TIGHT}</li>
 *   <li>Histogram         — {@code HISTCMP_CORREL} / {@code HISTCMP_CORREL_CF_LOOSE} /
 *       {@code HISTCMP_CORREL_CF1_LOOSE} / {@code HISTCMP_CORREL_CF1_TIGHT}</li>
 *   <li>Phase Correlation — {@code PHASE_CORRELATE} / {@code PHASE_CORRELATE_CF_LOOSE} /
 *       {@code PHASE_CORRELATE_CF1_LOOSE} / {@code PHASE_CORRELATE_CF1_TIGHT}</li>
 *   <li>Morphology        — {@code MORPH_COMBINED} / {@code MORPH_COMBINED_CF_LOOSE} /
 *       {@code MORPH_COMBINED_CF1_LOOSE} / {@code MORPH_COMBINED_CF1_TIGHT}</li>
 *   <li>Pixel Diff        — {@code PIXEL_DIFF} / {@code PIXEL_DIFF_CF_LOOSE} /
 *       {@code PIXEL_DIFF_CF1_LOOSE} / {@code PIXEL_DIFF_CF1_TIGHT}</li>
 * </ul>
 *
 * <h2>What to look for</h2>
 * <ul>
 *   <li><b>Speed:</b> {@code preFilterElapsedMs} for CF1 should be tiny vs base elapsed ms.
 *       The CF1 pipeline runs the expensive matcher only inside colour blobs, not the full frame.</li>
 *   <li><b>Accuracy:</b> CF1 scores should be ≥ CF_LOOSE on A_CLEAN and most B_TRANSFORMED
 *       scenes because the colour-localised window is already centred on the target.</li>
 *   <li><b>Failure modes:</b> C_DEGRADED/colour-shift scenes — the CF1 blob proposal may
 *       miss the target entirely (score = 0) if the hue has drifted outside the filter window.</li>
 *   <li><b>D_NEGATIVE:</b> CF1 dramatically reduces false alarm rate vs base — if no
 *       colour blob is found, only the full-scene fallback is searched.</li>
 * </ul>
 */
@DisplayName("Milestone 15 — Colour-First Region Proposal")
class ColourFirstRegionProposalTest extends AnalyticalTestBase {

    private static final boolean     DEBUG     = false;
    private static final ReferenceId DEBUG_REF = ReferenceId.CIRCLE_OUTLINE;
    private static final Path        OUT       = Paths.get("test_output", "colour_first");

    private static final ReferenceId[] REF_FILTER = {
//        ReferenceId.CIRCLE_OUTLINE,
        ReferenceId.RECT_FILLED,
//        ReferenceId.HEXAGON_OUTLINE,
//        ReferenceId.CROSSHAIR,
        ReferenceId.STAR_5_FILLED,
        ReferenceId.TEXT_HELLO,
    };

    private static final Set<SceneVariant> SCENE_VARIANTS = Set.of(
        SceneVariant.CLEAN,
        SceneVariant.SCALE_0_50,
        SceneVariant.SCALE_1_50,
        SceneVariant.ROT_90
//        SceneVariant.NOISE_S25,
//        SceneVariant.OFFSET_TOPLEFT
    );

    // Save the CF1 and their paired base/CF_LOOSE names for annotation
    private static final Set<String> SAVE = Set.of(
        // TM
        TmVariant.TM_CCOEFF_NORMED.variantName(),
        TmVariant.TM_CCOEFF_NORMED_CF_LOOSE.variantName(),
        TmVariant.TM_CCOEFF_NORMED_CF1_LOOSE.variantName(),
        TmVariant.TM_CCOEFF_NORMED_CF1_TIGHT.variantName(),
        // Feature
        FeatureVariant.SIFT.variantName(),
        FeatureVariant.SIFT_CF_LOOSE.variantName(),
        FeatureVariant.SIFT_CF1_LOOSE.variantName(),
        FeatureVariant.SIFT_CF1_TIGHT.variantName(),
        // Contour
        ContourVariant.CONTOURS_MATCH_I1.variantName(),
        ContourVariant.CONTOURS_MATCH_I1_CF_LOOSE.variantName(),
        ContourVariant.CONTOURS_MATCH_I1_CF1_LOOSE.variantName(),
        ContourVariant.CONTOURS_MATCH_I1_CF1_TIGHT.variantName(),
        // Hough
        HoughVariant.HOUGH_LINES_P.variantName(),
        HoughVariant.HOUGH_LINES_P_CF_LOOSE.variantName(),
        HoughVariant.HOUGH_LINES_P_CF1_LOOSE.variantName(),
        HoughVariant.HOUGH_LINES_P_CF1_TIGHT.variantName(),
        // Gen Hough
        GenHoughVariant.BALLARD.variantName(),
        GenHoughVariant.BALLARD_CF_LOOSE.variantName(),
        GenHoughVariant.BALLARD_CF1_LOOSE.variantName(),
        GenHoughVariant.BALLARD_CF1_TIGHT.variantName(),
        // Histogram
        HistVariant.HISTCMP_CORREL.variantName(),
        HistVariant.HISTCMP_CORREL_CF_LOOSE.variantName(),
        HistVariant.HISTCMP_CORREL_CF1_LOOSE.variantName(),
        HistVariant.HISTCMP_CORREL_CF1_TIGHT.variantName(),
        // Phase
        PhaseVariant.PHASE_CORRELATE.variantName(),
        PhaseVariant.PHASE_CORRELATE_CF_LOOSE.variantName(),
        PhaseVariant.PHASE_CORRELATE_CF1_LOOSE.variantName(),
        PhaseVariant.PHASE_CORRELATE_CF1_TIGHT.variantName(),
        // Morphology
        "MORPH_COMBINED",
        "MORPH_COMBINED_CF_LOOSE",
        "MORPH_COMBINED_CF1_LOOSE",
        "MORPH_COMBINED_CF1_TIGHT",
        // Pixel Diff
        PixelDiffVariant.PIXEL_DIFF.variantName(),
        PixelDiffVariant.PIXEL_DIFF_CF_LOOSE.variantName(),
        PixelDiffVariant.PIXEL_DIFF_CF1_LOOSE.variantName(),
        PixelDiffVariant.PIXEL_DIFF_CF1_TIGHT.variantName()
    );

    // Enable all CF tiers so CF_LOOSE, CF_TIGHT, and CF1 variants all pass the tier filter
    private static final EnumSet<CfMode> CF_TIERS = EnumSet.allOf(CfMode.class);

    @Override protected String           tag()             { return "CF1"; }
    @Override protected String           techniqueName()   { return "Colour-First Region Proposal"; }
    @Override protected Path             outputDir()       { return OUT; }
    @Override protected boolean          debugMode()       { return DEBUG; }
    @Override protected ReferenceId      debugRef()        { return DEBUG_REF; }
    @Override protected Set<String>      saveVariants()    { return SAVE; }
    @Override protected ReferenceId[]    referenceFilter() { return REF_FILTER; }
    @Override protected EnumSet<CfMode>  cfTierFilter()    { return CF_TIERS; }

    @Override
    protected boolean sceneFilter(SceneEntry scene) {
        if (scene.category() == SceneCategory.D_NEGATIVE) return true;
        return SCENE_VARIANTS.stream().anyMatch(v -> v.matches(scene));
    }

    /**
     * Runs all 9 matchers for each (refId, scene) pair and merges the results.
     * Each matcher emits its full variant set; the {@link #cfTierFilter()} and
     * {@link #saveVariants()} filtering in {@link AnalyticalTestBase} then
     * restrict which results appear in the report and which produce annotated PNGs.
     */
    @Override
    protected List<AnalysisResult> runMatcher(ReferenceId refId, Mat refMat,
                                              SceneEntry scene, Set<String> saveVariants,
                                              Path outputDir) {
        return Stream.of(
            TemplateMatcher.match(refId, refMat, scene, saveVariants, outputDir),
            FeatureMatcher.match(refId, refMat, scene, saveVariants, outputDir),
            ContourShapeMatcher.match(refId, refMat, scene, saveVariants, outputDir),
            HoughDetector.match(refId, refMat, scene, saveVariants, outputDir),
            GeneralizedHoughDetector.match(refId, refMat, scene, saveVariants, outputDir),
            HistogramMatcher.match(refId, refMat, scene, saveVariants, outputDir),
            PhaseCorrelationMatcher.match(refId, refMat, scene, saveVariants, outputDir),
            MorphologyAnalyzer.match(refId, refMat, scene, saveVariants, outputDir),
            PixelDiffMatcher.match(refId, refMat, scene, saveVariants, outputDir)
        ).flatMap(List::stream).toList();
    }
}

