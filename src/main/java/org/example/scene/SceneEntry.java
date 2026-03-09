package org.example.scene;

import org.example.factories.BackgroundId;
import org.example.analytics.DetectionVerdict;
import org.example.factories.ReferenceId;
import org.opencv.core.Mat;
import org.opencv.core.Rect;

import java.util.Collections;
import java.util.List;

/**
 * A single entry in the {@link SceneCatalogue}.
 *
 * @param primaryReferenceId  the reference placed in this scene, or null for Cat D
 * @param category            A_CLEAN / B_TRANSFORMED / C_DEGRADED / D_NEGATIVE
 * @param variantLabel        short description e.g. "scale_0.5", "noise_s25", "negative_bg_only"
 * @param backgroundId        which background was used
 * @param placements          ground-truth list — exactly 1 for A/B/C, empty for D
 * @param sceneMat            the rendered 640x480 BGR scene Mat
 */
public record SceneEntry(
        ReferenceId primaryReferenceId,
        SceneCategory             category,
        String                    variantLabel,
        BackgroundId backgroundId,
        List<SceneShapePlacement> placements,
        Mat                       sceneMat
) {
    @Override
    public List<SceneShapePlacement> placements() {
        return Collections.unmodifiableList(placements);
    }

    public boolean hasReference() {
        return !placements.isEmpty();
    }

    /**
     * Convenience — returns the ground-truth bounding rect of the first placement,
     * or {@code null} for Category D (negative) scenes.
     */
    public Rect groundTruthRect() {
        return placements.isEmpty() ? null : placements.get(0).placedRect();
    }

    /**
     * Creates a minimal stub SceneEntry carrying only enough ground-truth information
     * to reconstruct a {@link DetectionVerdict}.  The {@code sceneMat} is null.
     * Used when reloading results from JSON sidecars.
     */
    public static SceneEntry stub(ReferenceId refId, SceneCategory category,
                                   BackgroundId bgId, String variantLabel,
                                   Rect groundTruthRect) {
        List<SceneShapePlacement> placements = (refId != null && groundTruthRect != null)
                ? List.of(SceneShapePlacement.clean(refId, groundTruthRect))
                : List.of();
        return new SceneEntry(refId, category, variantLabel, bgId, placements, null);
    }
}

