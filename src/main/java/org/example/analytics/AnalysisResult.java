package org.example.analytics;

import org.example.factories.BackgroundId;
import org.example.factories.ReferenceId;
import org.example.scene.SceneCategory;
import org.opencv.core.Rect;
import java.nio.file.Path;

/**
 * The result of running one matching technique variant against one scene entry.
 *
 * @param methodName        e.g. "TM_CCOEFF_NORMED", "SIFT", "TM_CCOEFF_NORMED_CF_LOOSE"
 * @param referenceId       the reference image that was searched for
 * @param variantLabel      the scene variant label e.g. "rot_45", "noise_s25"
 * @param category          A_CLEAN / B_TRANSFORMED / C_DEGRADED / D_NEGATIVE
 * @param backgroundId      which background the scene used
 * @param matchScorePercent match score 0–100 (higher = better match)
 * @param boundingRect      bounding box of the best match found in the scene (may be null)
 * @param elapsedMs         wall-clock time for the matcher call in milliseconds
 * @param preFilterMs       time spent in colour pre-filter (0 for base variants)
 * @param scenePx           total pixel count of the scene (width × height)
 * @param annotatedPath     path to the saved annotated PNG on disk, or null if not saved
 * @param isError           true if the matcher threw an exception
 * @param errorMessage      exception message if isError, otherwise null
 * @param scoringLayers     per-layer score breakdown (Layer 1/2/3), or {@link ScoringLayers#ZERO}
 */
public record AnalysisResult(
        String        methodName,
        ReferenceId referenceId,
        String        variantLabel,
        SceneCategory category,
        BackgroundId backgroundId,
        double        matchScorePercent,
        Rect          boundingRect,
        long          elapsedMs,
        long          preFilterMs,
        int           scenePx,
        Path          annotatedPath,
        boolean       isError,
        String        errorMessage,
        ScoringLayers scoringLayers
) {
    /**
     * Per-layer scoring breakdown produced by {@code VectorMatcher.scoreRegion()}.
     *
     * <p>All values are in the same 0–100 % scale as {@link #matchScorePercent}.
     *
     * @param boundaryCount Layer 1 (w=0.15) — boundary-count match: exponential
     *                      penalty when the candidate region has more or fewer
     *                      distinct boundaries than the reference.
     * @param structural    Layer 2 (w=0.25) — structural coherence: spatial
     *                      proximity and relative-size consistency of each matched
     *                      boundary, weighted by structural role.
     * @param geometry      Layer 3 (w=0.60) — primary geometry: {@code VectorSignature}
     *                      similarity on the primary boundary contour.
     */
    public record ScoringLayers(double boundaryCount, double structural, double geometry) {
        /** Sentinel used when no real breakdown is available (zero / error results). */
        public static final ScoringLayers ZERO = new ScoringLayers(0, 0, 0);
    }

    /** Convenience constructor — no pre-filter, no error, no saved image, no layer breakdown. */
    public static AnalysisResult of(String method, ReferenceId ref, String variant,
                                    SceneCategory cat, BackgroundId bg,
                                    double score, Rect rect,
                                    long elapsedMs, int scenePx) {
        return new AnalysisResult(method, ref, variant, cat, bg,
                score, rect, elapsedMs, 0L, scenePx, null, false, null,
                ScoringLayers.ZERO);
    }

    /** Convenience constructor — error case. */
    public static AnalysisResult error(String method, ReferenceId ref, String variant,
                                       SceneCategory cat, BackgroundId bg,
                                       long elapsedMs, int scenePx, String message) {
        return new AnalysisResult(method, ref, variant, cat, bg,
                0.0, null, elapsedMs, 0L, scenePx, null, true, message,
                ScoringLayers.ZERO);
    }

    /** Colour-coded tier for display: GREEN ≥ 70, YELLOW ≥ 40, RED < 40. */
    public String matchScoreEmoji() {
        if (isError) return "💥";
        if (matchScorePercent >= 70) return "🟢";
        if (matchScorePercent >= 40) return "🟡";
        return "🔴";
    }
}
