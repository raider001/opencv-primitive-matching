package org.example.matchers.vectormatcher.components;

import org.example.matchers.SceneContourEntry;
import org.opencv.core.Rect;

import java.util.*;

/**
 * Anchor-based expansion and matching logic.
 *
 * <p>For each candidate anchor contour, this class:
 * <ol>
 *   <li>Assigns the anchor to the best-matching reference cluster (by relative size)</li>
 *   <li>Expands the match set by selecting additional scene clusters via proximity
 *       and relative size (no colour, no geometry)</li>
 *   <li>Deduplicates achromatic pairs that trace the same physical edge</li>
 * </ol>
 */
public final class AnchorMatcher {

    /** Max centre-to-centre distance as a fraction of scene diagonal for expansion. */
    private static final double PROXIMITY_THRESHOLD = 0.35;

    /**
     * Assigns an anchor contour to the reference cluster whose relative contour area
     * (vs ref image) most closely matches the anchor's (vs its own bbox area).
     *
     * @param anchor candidate anchor contour
     * @param anchorBboxArea anchor's bounding box area (width × height)
     * @param refClusters all reference clusters
     * @return the best-matching reference cluster
     */
    public static RefCluster assignAnchorToRef(SceneContourEntry anchor,
                                               double anchorBboxArea,
                                               List<RefCluster> refClusters) {
        double anchorFrac = anchor.area() / Math.max(1.0, anchorBboxArea);
        RefCluster best = refClusters.get(0);
        double bestDiff = Double.MAX_VALUE;

        for (RefCluster rc : refClusters) {
            double refFrac = refFraction(rc);
            double diff = Math.abs(refFrac - anchorFrac);
            if (diff < bestDiff) {
                bestDiff = diff;
                best = rc;
            }
        }
        return best;
    }

    /**
     * Expands the matched set from a single anchor by selecting additional scene
     * clusters via proximity and relative size.
     *
     * <p>No colour or geometry checks — purely spatial and size-based selection.
     *
     * @param anchor the anchor contour (starting point)
     * @param anchorRef the reference cluster assigned to the anchor
     * @param candidates all scene candidates
     * @param refClusters all reference clusters
     * @param sceneDiag scene diagonal (for proximity threshold)
     * @return matched set (anchor + expanded contours) and final region bbox
     */
    public static MatchResult expandFromAnchor(SceneContourEntry anchor,
                                               RefCluster anchorRef,
                                               List<SceneContourEntry> candidates,
                                               List<RefCluster> refClusters,
                                               double sceneDiag) {
        // For multi-cluster refs (>2), use larger proximity threshold to handle
        // patterns like BICOLOUR_CROSSHAIR_RING where components may be far apart.
        double proximityThreshold = (refClusters.size() > 2) ? 0.50 : PROXIMITY_THRESHOLD;

        Rect regionBbox = anchor.bbox();
        List<SceneContourEntry> matched = new ArrayList<>();
        matched.add(anchor);
        Set<Integer> usedIdx = new HashSet<>();
        usedIdx.add(anchor.clusterIdx());

        for (RefCluster rc : refClusters) {
            if (rc == anchorRef) continue;   // anchor already covers this one

            double refFrac = refFraction(rc);
            SceneContourEntry best = null;
            double bestDiff = Double.MAX_VALUE;

            for (SceneContourEntry ce : candidates) {
                if (usedIdx.contains(ce.clusterIdx())) continue;

                // Proximity gate (relaxed for multi-cluster patterns)
                // Use regionBbox (growing bbox) instead of anchor bbox to allow expansion
                // to distant components that are part of the same pattern
                double dist = centreDist(regionBbox, ce.bbox());
                if (dist > sceneDiag * proximityThreshold) continue;

                // Relative size match — compare against candidate bbox so far
                double ceFrac = sceneFraction(ce, regionBbox);
                double diff = Math.abs(refFrac - ceFrac);
                if (diff < bestDiff) {
                    bestDiff = diff;
                    best = ce;
                }
            }

            if (best != null) {
                matched.add(best);
                usedIdx.add(best.clusterIdx());
                // Only expand regionBbox for non-background-scale contours
                Rect bestBb = best.bbox();
                double bestArea = (double) bestBb.width * bestBb.height;
                // Assume sceneArea is available through descriptor or passed separately
                // For now, use a conservative threshold based on bbox size
                if (bestArea < 307200 * 0.60) {  // 640×480 = 307200
                    regionBbox = unionRect(regionBbox, bestBb);
                }
            }
        }

        return new MatchResult(matched, regionBbox);
    }

    /**
     * Result of anchor expansion: matched contours + final region bbox.
     */
    public record MatchResult(List<SceneContourEntry> matched, Rect regionBbox) {}

    // ── Helper methods ────────────────────────────────────────────────────────

    private static double refFraction(RefCluster rc) {
        // Reference image is 128×128 = 16384 px
        return rc.maxContourArea / 16384.0;
    }

    private static double sceneFraction(SceneContourEntry ce, Rect regionBbox) {
        double bboxArea = Math.max(1.0, (double) regionBbox.width * regionBbox.height);
        return ce.area() / bboxArea;
    }

    private static double centreDist(Rect a, Rect b) {
        double ax = a.x + a.width / 2.0;
        double ay = a.y + a.height / 2.0;
        double bx = b.x + b.width / 2.0;
        double by = b.y + b.height / 2.0;
        return Math.hypot(ax - bx, ay - by);
    }

    private static Rect unionRect(Rect a, Rect b) {
        int x1 = Math.min(a.x, b.x);
        int y1 = Math.min(a.y, b.y);
        int x2 = Math.max(a.x + a.width, b.x + b.width);
        int y2 = Math.max(a.y + a.height, b.y + b.height);
        return new Rect(x1, y1, x2 - x1, y2 - y1);
    }

    private AnchorMatcher() {}  // static utility class
}
