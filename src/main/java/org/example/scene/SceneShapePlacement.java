package org.example.scene;

import org.example.factories.ReferenceId;
import org.opencv.core.Rect;

/**
 * Ground-truth record describing a single reference shape placed into a scene.
 *
 * Every Category A/B/C {@link SceneEntry} carries exactly one {@code SceneShapePlacement}.
 * Category D (negative) scenes carry an empty list.
 *
 * @param referenceId        which reference shape was placed
 * @param placedRect         bounding box of the placed shape in scene pixel coordinates
 * @param scaleFactor        uniform scale applied before placement (1.0 = none)
 * @param rotationDeg        clockwise rotation in degrees (0.0 = none)
 * @param offsetX            horizontal pixel offset of shape centre from scene centre
 * @param offsetY            vertical pixel offset of shape centre from scene centre
 * @param colourShifted      true if +40 deg HSV hue rotation was applied
 * @param occluded           true if part of the shape was masked out
 * @param occlusionFraction  fraction of the bounding rect that was occluded (0.0-1.0)
 */
public record SceneShapePlacement(
        ReferenceId referenceId,
        Rect        placedRect,
        double      scaleFactor,
        double      rotationDeg,
        int         offsetX,
        int         offsetY,
        boolean     colourShifted,
        boolean     occluded,
        double      occlusionFraction
) {
    public static SceneShapePlacement clean(ReferenceId id, Rect rect) {
        return new SceneShapePlacement(id, rect, 1.0, 0.0, 0, 0, false, false, 0.0);
    }

    public static SceneShapePlacement transformed(ReferenceId id, Rect rect,
                                                   double scale, double rotDeg,
                                                   int offX, int offY) {
        return new SceneShapePlacement(id, rect, scale, rotDeg, offX, offY, false, false, 0.0);
    }

    public String summary() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("[%d,%d %dx%d]",
                placedRect.x, placedRect.y, placedRect.width, placedRect.height));
        if (scaleFactor != 1.0)           sb.append(String.format(" s=%.2f",  scaleFactor));
        if (rotationDeg != 0.0)           sb.append(String.format(" r=%.0f",  rotationDeg));
        if (offsetX != 0 || offsetY != 0) sb.append(String.format(" o=%d,%d", offsetX, offsetY));
        if (colourShifted)                sb.append(" hue+40");
        if (occluded)                     sb.append(String.format(" occ%.0f%%", occlusionFraction * 100));
        return sb.toString();
    }
}

