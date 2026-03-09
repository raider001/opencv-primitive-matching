package org.example.scene;

import org.example.factories.BackgroundId;
import org.example.analytics.DetectionVerdict;
import org.example.factories.ReferenceId;
import org.example.matchers.SceneDescriptor;
import org.opencv.core.Mat;
import org.opencv.core.Rect;

import java.util.Collections;
import java.util.List;

/**
 * A single entry in the {@link SceneCatalogue}.
 *
 * <p>Owns the rendered scene {@link Mat} and a pre-computed {@link SceneDescriptor}
 * (colour-cluster contours) built once at construction time.  Both are released
 * together by calling {@link #release()}.
 *
 * <p>Callers should call {@link #release()} when the entry is no longer needed
 * to free native OpenCV memory.
 */
public final class SceneEntry {

    private final ReferenceId             primaryReferenceId;
    private final SceneCategory           category;
    private final String                  variantLabel;
    private final BackgroundId            backgroundId;
    private final List<SceneShapePlacement> placements;
    private final Mat                     sceneMat;
    private final SceneDescriptor         descriptor;

    /**
     * Primary constructor — builds and owns a {@link SceneDescriptor} from the scene mat.
     *
     * @param primaryReferenceId  the reference placed in this scene, or null for Cat D
     * @param category            A_CLEAN / B_TRANSFORMED / C_DEGRADED / D_NEGATIVE
     * @param variantLabel        short description e.g. "scale_0.5"
     * @param backgroundId        which background was used
     * @param placements          ground-truth list — 1 for A/B/C, empty for D
     * @param sceneMat            the rendered 640×480 BGR scene Mat (owned by this entry)
     */
    public SceneEntry(ReferenceId primaryReferenceId,
                      SceneCategory category,
                      String variantLabel,
                      BackgroundId backgroundId,
                      List<SceneShapePlacement> placements,
                      Mat sceneMat) {
        this.primaryReferenceId = primaryReferenceId;
        this.category           = category;
        this.variantLabel       = variantLabel;
        this.backgroundId       = backgroundId;
        this.placements         = placements != null ? placements : List.of();
        this.sceneMat           = sceneMat;
        // Build the descriptor once — null-safe for stub entries with no mat
        this.descriptor = (sceneMat != null && !sceneMat.empty())
                ? SceneDescriptor.build(sceneMat)
                : null;
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    public ReferenceId primaryReferenceId()         { return primaryReferenceId; }
    public SceneCategory category()                  { return category; }
    public String variantLabel()                     { return variantLabel; }
    public BackgroundId backgroundId()               { return backgroundId; }
    public Mat sceneMat()                            { return sceneMat; }

    /** Pre-computed colour-cluster contour descriptor. May be null for stub entries. */
    public SceneDescriptor descriptor()              { return descriptor; }

    /** Time taken to build the descriptor in ms. 0 for stub entries. */
    public long descriptorBuildMs() {
        return descriptor != null ? descriptor.buildMs : 0L;
    }

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
     * Releases the scene mat and the pre-computed descriptor.
     * Must be called when this entry is no longer needed.
     */
    public void release() {
        if (descriptor != null) descriptor.release();
        if (sceneMat  != null) sceneMat.release();
    }

    // -------------------------------------------------------------------------
    // Stub factory — no mat, no descriptor
    // -------------------------------------------------------------------------

    /**
     * Creates a minimal stub SceneEntry carrying only enough ground-truth information
     * to reconstruct a {@link DetectionVerdict}.  The {@code sceneMat} is null
     * and no descriptor is built.  Used when reloading results from JSON sidecars.
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
