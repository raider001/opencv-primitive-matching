package org.example;

import org.example.analytics.AnalysisResult;
import org.example.factories.ReferenceId;
import org.example.matchers.*;
import org.example.matchers.vectormatcher.VectorMatcher;
import org.example.scene.SceneEntry;
import org.opencv.core.Mat;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Central registry of all {@link MatcherDescriptor} instances.
 *
 * <p>Each matcher is registered once here.  The {@code BenchmarkLauncher}
 * and any other tooling reads from this registry to discover available matchers,
 * their variant names, and how to invoke them.
 *
 * <p>To add a new matcher: implement {@link MatcherDescriptor} inline or as a
 * named class and add it to {@link #ALL}.
 */
public final class MatcherRegistry {

    private MatcherRegistry() {}

    /** Ordered list of all registered matchers, in benchmark presentation order. */
    public static final List<MatcherDescriptor> ALL = List.of(

        descriptor("TM",      "Template Matching",
                Paths.get("test_output", "template_matching"),
                MatcherVariant.allNamesOf(TmVariant.class),
                TemplateMatcher::match),

        descriptor("FM",      "Feature Matching",
                Paths.get("test_output", "feature_matching"),
                MatcherVariant.allNamesOf(FeatureVariant.class),
                FeatureMatcher::match),

        descriptor("CSM",     "Contour Shape Matching",
                Paths.get("test_output", "contour_shape_matching"),
                MatcherVariant.allNamesOf(ContourVariant.class),
                ContourShapeMatcher::match),

        descriptor("HOUGH",   "Hough Transforms",
                Paths.get("test_output", "hough_transforms"),
                MatcherVariant.allNamesOf(HoughVariant.class),
                HoughDetector::match),

        descriptor("GHT",     "Generalized Hough",
                Paths.get("test_output", "generalized_hough"),
                MatcherVariant.allNamesOf(GenHoughVariant.class),
                GeneralizedHoughDetector::match),

        descriptor("HIST",    "Histogram Comparison",
                Paths.get("test_output", "histogram_matching"),
                MatcherVariant.allNamesOf(HistVariant.class),
                HistogramMatcher::match),

        descriptor("PC",      "Phase Correlation",
                Paths.get("test_output", "phase_correlation"),
                MatcherVariant.allNamesOf(PhaseVariant.class),
                PhaseCorrelationMatcher::match),

        descriptor("MORPH",   "Morphology Analysis",
                Paths.get("test_output", "morphology_analysis"),
                morphVariantNames(),
                MorphologyAnalyzer::match),

        descriptor("PDIFF",   "Pixel Diff",
                Paths.get("test_output", "pixel_diff"),
                MatcherVariant.allNamesOf(PixelDiffVariant.class),
                PixelDiffMatcher::match),

        descriptor("SSIM",    "SSIM",
                Paths.get("test_output", "ssim_matching"),
                MatcherVariant.allNamesOf(SsimVariant.class),
                SsimMatcher::match),

        descriptor("CHAMFER", "Chamfer Distance",
                Paths.get("test_output", "chamfer_matching"),
                MatcherVariant.allNamesOf(ChamferVariant.class),
                ChamferMatcher::match),

        descriptor("FSM",     "Fourier Shape Descriptors",
                Paths.get("test_output", "fourier_shape_matching"),
                MatcherVariant.allNamesOf(FourierShapeVariant.class),
                FourierShapeMatcher::match),

        descriptor("VM",      "Vector Matching",
                Paths.get("test_output", "vector_matching"),
                MatcherVariant.allNamesOf(VectorVariant.class),
                VectorMatcher::match)
    );

    /** Returns the descriptor with the given tag, or empty. */
    public static Optional<MatcherDescriptor> byTag(String tag) {
        return ALL.stream().filter(d -> d.tag().equals(tag)).findFirst();
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    @FunctionalInterface
    private interface MatchFn {
        List<AnalysisResult> match(ReferenceId refId, Mat refMat,
                                   SceneEntry scene, Set<String> saveVariants,
                                   Path outputDir);
    }

    private static MatcherDescriptor descriptor(String tag, String displayName,
                                                 Path outputDir,
                                                 Set<String> variantNames,
                                                 MatchFn fn) {
        List<String> variants = List.copyOf(variantNames);
        return new MatcherDescriptor() {
            @Override public String           tag()          { return tag; }
            @Override public String           displayName()  { return displayName; }
            @Override public Path             outputDir()    { return outputDir; }
            @Override public List<String>     variantNames() { return variants; }
            @Override public List<AnalysisResult> run(ReferenceId refId, Mat refMat,
                                                      SceneEntry scene,
                                                      Set<String> saveVariants,
                                                      Path outDir) {
                return fn.match(refId, refMat, scene, saveVariants, outDir);
            }
        };
    }

    /** MorphologyAnalyzer uses string constants rather than a variant enum — build the set manually. */
    private static Set<String> morphVariantNames() {
        Set<String> s = new LinkedHashSet<>();
        for (String base : List.of(MorphologyAnalyzer.VAR_POLY,
                                   MorphologyAnalyzer.VAR_CIRC,
                                   MorphologyAnalyzer.VAR_COMBINED)) {
            s.add(base);
            s.add(base + "_CF_LOOSE");
            s.add(base + "_CF_TIGHT");
        }
        return Collections.unmodifiableSet(s);
    }
}

