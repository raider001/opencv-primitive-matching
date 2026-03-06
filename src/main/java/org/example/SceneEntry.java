package org.example;

import org.opencv.core.Mat;

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
        ReferenceId               primaryReferenceId,
        SceneCategory             category,
        String                    variantLabel,
        BackgroundId              backgroundId,
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
}

