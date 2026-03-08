package org.example.matchingtests;

import org.example.*;
import org.example.matchers.*;
import org.example.utilities.AnalyticalTestBase;
import org.junit.jupiter.api.DisplayName;
import org.opencv.core.Mat;
import org.opencv.core.Rect;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.BiFunction;

/**
 * Milestone 21 — Multi-Colour-First Region Proposal analytical test.
 *
 * <h2>Purpose</h2>
 * Validates {@link ColourFirstLocator#proposeMulti} across the 5 new multi-colour
 * reference shapes plus 3 single-colour control shapes.  All 9 base matchers are
 * exercised in MCF1_LOOSE and MCF1_TIGHT modes by:
 * <ol>
 *   <li>Proposing candidate windows via {@link ColourFirstLocator#proposeMulti}.</li>
 *   <li>Cropping the scene to each window and calling the matcher's existing
 *       {@code match()} on the cropped sub-Mat.</li>
 *   <li>Selecting the highest-scoring window and re-mapping its bbox to scene coords.</li>
 * </ol>
 *
 * <h2>What to look for</h2>
 * <ul>
 *   <li>MCF1 score ≥ single-colour CF1 on bi/tri-colour references</li>
 *   <li>MCF1 score ≈ CF1 on single-colour control references (graceful degradation)</li>
 *   <li>D_NEGATIVE: score ≈ 0% (no real windows → full-scene fallback, near-zero)</li>
 *   <li>C_DEGRADED hue-shift: MCF1 degrades gracefully (≥1 channel may still fire)</li>
 * </ul>
 */
@DisplayName("Milestone 21 — Multi-Colour-First Region Proposal")
class MultiColourFirstTest extends AnalyticalTestBase {

    private static final boolean     DEBUG     = false;
    private static final ReferenceId DEBUG_REF = ReferenceId.BICOLOUR_CIRCLE_RING;
    private static final Path        OUT       = Paths.get("test_output", "multi_colour_first");

    private static final ReferenceId[] REF_FILTER = {
        ReferenceId.BICOLOUR_CIRCLE_RING,
        ReferenceId.BICOLOUR_RECT_HALVES,
        ReferenceId.TRICOLOUR_TRIANGLE,
        ReferenceId.BICOLOUR_CROSSHAIR_RING,
        ReferenceId.BICOLOUR_CHEVRON_FILLED,
        // Single-colour controls
        ReferenceId.CIRCLE_OUTLINE,
        ReferenceId.HEXAGON_OUTLINE,
        ReferenceId.CROSSHAIR
    };

    private static final Set<SceneVariant> SCENE_VARIANTS = Set.of(
        SceneVariant.CLEAN_BG_SOLID_BLACK,
        SceneVariant.CLEAN_BG_NOISE_LIGHT,
        SceneVariant.SCALE_0_50,
        SceneVariant.SCALE_1_50,
        SceneVariant.ROT_45,
        SceneVariant.ROT_90,
        SceneVariant.NOISE_S25,
        SceneVariant.OCC_25PCT,
        SceneVariant.HUE_SHIFT_40
    );

    private static final Set<String> SAVE = new HashSet<>(List.of(
        "MCF1_LOOSE_TM",    "MCF1_TIGHT_TM",
        "MCF1_LOOSE_SIFT",  "MCF1_TIGHT_SIFT",
        "MCF1_LOOSE_CSM",   "MCF1_TIGHT_CSM",
        "MCF1_LOOSE_HOUGH", "MCF1_TIGHT_HOUGH",
        "MCF1_LOOSE_GHT",   "MCF1_TIGHT_GHT",
        "MCF1_LOOSE_HIST",  "MCF1_TIGHT_HIST",
        "MCF1_LOOSE_PC",    "MCF1_TIGHT_PC",
        "MCF1_LOOSE_MORPH", "MCF1_TIGHT_MORPH",
        "MCF1_LOOSE_PDIFF", "MCF1_TIGHT_PDIFF"
    ));

    private static final EnumSet<CfMode> CF_TIERS = EnumSet.allOf(CfMode.class);

    @Override protected String           tag()             { return "MCF1"; }
    @Override protected String           techniqueName()   { return "Multi-Colour-First Region Proposal"; }
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

    @Override
    protected List<AnalysisResult> runMatcher(ReferenceId refId, Mat refMat,
                                              SceneEntry scene, Set<String> sv,
                                              Path outputDir) {
        List<AnalysisResult> out = new ArrayList<>();

        for (double tol : new double[]{ ColourPreFilter.LOOSE, ColourPreFilter.TIGHT }) {
            String tag = (tol == ColourPreFilter.LOOSE) ? "MCF1_LOOSE" : "MCF1_TIGHT";

            long t0 = System.currentTimeMillis();
            List<Rect> windows = ColourFirstLocator.proposeMulti(
                    scene.sceneMat(), refId, tol,
                    ColourFirstLocator.DEFAULT_MIN_AREA,
                    ColourFirstLocator.DEFAULT_MIN_HUE_SEP,
                    ColourFirstLocator.DEFAULT_MIN_SATURATION);
            long proposeMs = System.currentTimeMillis() - t0;

            out.add(bestWindow(refId, scene, windows, tag + "_TM",    proposeMs,
                    (r, s) -> TemplateMatcher.match(r, refMat, s, Set.of(), outputDir)));
            out.add(bestWindow(refId, scene, windows, tag + "_SIFT",  proposeMs,
                    (r, s) -> FeatureMatcher.match(r, refMat, s, Set.of(), outputDir)));
            out.add(bestWindow(refId, scene, windows, tag + "_CSM",   proposeMs,
                    (r, s) -> ContourShapeMatcher.match(r, refMat, s, Set.of(), outputDir)));
            out.add(bestWindow(refId, scene, windows, tag + "_HOUGH", proposeMs,
                    (r, s) -> HoughDetector.match(r, refMat, s, Set.of(), outputDir)));
            out.add(bestWindow(refId, scene, windows, tag + "_GHT",   proposeMs,
                    (r, s) -> GeneralizedHoughDetector.match(r, refMat, s, Set.of(), outputDir)));
            out.add(bestWindow(refId, scene, windows, tag + "_HIST",  proposeMs,
                    (r, s) -> HistogramMatcher.match(r, refMat, s, Set.of(), outputDir)));
            out.add(bestWindow(refId, scene, windows, tag + "_PC",    proposeMs,
                    (r, s) -> PhaseCorrelationMatcher.match(r, refMat, s, Set.of(), outputDir)));
            out.add(bestWindow(refId, scene, windows, tag + "_MORPH", proposeMs,
                    (r, s) -> MorphologyAnalyzer.match(r, refMat, s, Set.of(), outputDir)));
            out.add(bestWindow(refId, scene, windows, tag + "_PDIFF", proposeMs,
                    (r, s) -> PixelDiffMatcher.match(r, refMat, s, Set.of(), outputDir)));
        }
        return out;
    }

    // -------------------------------------------------------------------------
    // Generic windowed best-match helper
    // -------------------------------------------------------------------------

    private static AnalysisResult bestWindow(ReferenceId refId,
                                              SceneEntry scene,
                                              List<Rect> windows,
                                              String variantName,
                                              long proposeMs,
                                              BiFunction<ReferenceId, SceneEntry, List<AnalysisResult>> matchFn) {
        long t0 = System.currentTimeMillis();
        double bestScore = -1;
        Rect   bestBbox  = null;

        for (Rect w : windows) {
            Mat crop = scene.sceneMat().submat(w);
            SceneEntry croppedScene = new SceneEntry(
                    scene.primaryReferenceId(), scene.category(),
                    scene.variantLabel(), scene.backgroundId(),
                    scene.placements(), crop);
            try {
                for (AnalysisResult r : matchFn.apply(refId, croppedScene)) {
                    if (!r.isError() && r.matchScorePercent() > bestScore) {
                        bestScore = r.matchScorePercent();
                        Rect lb = r.boundingRect();
                        bestBbox = (lb != null)
                                ? new Rect(w.x + lb.x, w.y + lb.y, lb.width, lb.height)
                                : w;
                    }
                }
            } catch (Exception ignored) { /* skip bad windows */ }
        }

        long elapsed = System.currentTimeMillis() - t0 + proposeMs;
        double score = Math.max(0, bestScore);
        boolean error = (bestScore < 0);

        return new AnalysisResult(variantName, refId,
                scene.variantLabel(), scene.category(), scene.backgroundId(),
                score, bestBbox, elapsed, proposeMs,
                scene.sceneMat().rows() * scene.sceneMat().cols(),
                null, error, error ? "No window matched" : null);
    }
}

