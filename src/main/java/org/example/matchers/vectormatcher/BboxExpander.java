package org.example.matchers.vectormatcher;

import org.example.matchers.SceneContourEntry;
import org.opencv.core.Rect;

import java.util.List;

/**
 * Post-score bounding box expansion and refinement.
 *
 * <p>After the anchor with the best score is identified, this class performs
 * multi-step bbox expansion to capture the complete shape extent:
 * <ol>
 *   <li><b>Anchor trimming:</b> If background noise inflated the anchor bbox beyond
 *       the reference footprint, trim it proportionally</li>
 *   <li><b>Matched contour union:</b> Union with co-located matched contours from the
 *       scoring loop (already vetted)</li>
 *   <li><b>Sibling expansion:</b> Union with overlapping/concentric unscored contours
 *       from the same cluster</li>
 * </ol>
 *
 * <p>All expansions are capped by area limits derived from reference geometry and
 * estimated scene scale to prevent runaway growth from background elements.
 */
public final class BboxExpander {

    private static final boolean VM_BBOX_DEBUG = System.getProperty("vm.bbox().debug") != null;

    /**
     * Expands the best-match bounding box using reference geometry and matched contours.
     *
     * @param bestBbox initial bbox from the scoring loop
     * @param bestAnchor the anchor contour with the best score
     * @param matched contours that were matched during anchor expansion
     * @param candidates all scene candidates (for sibling expansion)
     * @param refClusters all reference clusters
     * @param referenceId reference shape ID (for debug logging)
     * @param sceneArea total scene area (for sanity check)
     * @return expanded bbox, or original if expansion would exceed 80% of scene area
     */
    public static Rect expandBbox(Rect bestBbox,
                                   SceneContourEntry bestAnchor,
                                   List<SceneContourEntry> matched,
                                   List<SceneContourEntry> candidates,
                                   List<RefCluster> refClusters,
                                   String referenceId,
                                   double sceneArea) {
        Rect initialBbox = new Rect(bestBbox.x, bestBbox.y, bestBbox.width, bestBbox.height);
        Rect anchorBbox = bestAnchor.bbox();

        // ── Step A: Compute reference full extent and scene scale ─────────────
        Rect refFullBbox = refClusters.get(0).primaryBbox();
        for (RefCluster rc : refClusters) {
            refFullBbox = GeometryUtils.unionRect(refFullBbox, rc.primaryBbox());
        }

        // Estimate scene-to-reference scale by searching ALL contours across
        // ALL ref clusters.  Strategy: for each ref contour with diagonal ≥ 6px,
        // compute candidate scale = anchorDiag / refContourDiag.  Select the
        // candidate whose deviation from the nearest 0.5-step value is smallest
        // (e.g. 2.88 → 3.0, dev=0.12; 1.83 → 2.0, dev=0.17).
        // Half-step quantisation reflects natural scene scales (1×, 1.5×, 2×, 3×).
        double anchorDiag = Math.hypot(anchorBbox.width, anchorBbox.height);
        double estimatedScale = estimateScale(anchorDiag, refClusters);

        // ── Step B: Compute area caps derived from reference geometry ─────────
        AreaCaps caps = computeAreaCaps(refFullBbox, estimatedScale);

        // ── Step C: Union co-located matched contours ─────────────────────────
        Rect expandedBbox = trimAndUnionMatched(anchorBbox, matched, bestAnchor,
                refClusters, caps, referenceId, anchorDiag);

        // ── Step D: Sibling expansion ─────────────────────────────────────────
        expandedBbox = unionConcentricAndOverlappingSiblings(
                expandedBbox, matched, candidates, caps.siblingExpArea);

        // Sanity check: reject if frame-spanning (> 80 % scene area)
        double expandedArea = (double) expandedBbox.width * expandedBbox.height;
        if (expandedArea > sceneArea * 0.80) {
            if (VM_BBOX_DEBUG) {
                System.out.printf("[BBOX-DEBUG] %s: REJECTED (frame-spanning %.1f%%)%n",
                        referenceId, expandedArea / sceneArea * 100);
            }
            return bestBbox;  // keep original
        }

        if (VM_BBOX_DEBUG) {
            System.out.printf("[BBOX-DEBUG] %s: scale=%.2f refExt=%dx%d " +
                    "trim=%.0f union=%.0f sib=%.0f initial=(%d,%d %dx%d) expanded=(%d,%d %dx%d) ACCEPTED%n",
                    referenceId, estimatedScale,
                    refFullBbox.width, refFullBbox.height,
                    caps.anchorTrimArea, caps.matchedUnionArea, caps.siblingExpArea,
                    initialBbox.x, initialBbox.y, initialBbox.width, initialBbox.height,
                    expandedBbox.x, expandedBbox.y, expandedBbox.width, expandedBbox.height);
        }

        return expandedBbox;
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private static double estimateScale(double anchorDiag, List<RefCluster> refClusters) {
        double estimatedScale = 3.0;  // fallback
        double bestRatioDev = Double.MAX_VALUE;

        for (RefCluster rc : refClusters) {
            for (int ri = 0; ri < rc.contours.size(); ri++) {
                Rect rcBb = rc.contourBboxes[ri];
                double rcDiag = Math.hypot(rcBb.width, rcBb.height);
                if (rcDiag < 6.0) continue;

                double candidate = anchorDiag / rcDiag;
                if (candidate < 1.0 || candidate > 8.0) continue;

                double dev = Math.abs(candidate - Math.round(candidate * 2.0) / 2.0);
                if (dev < bestRatioDev) {
                    bestRatioDev = dev;
                    estimatedScale = candidate;
                }
            }
        }

        return Math.min(8.0, Math.max(1.0, estimatedScale));
    }

    private static AreaCaps computeAreaCaps(Rect refFullBbox, double estimatedScale) {
        // Three area limits:
        //  anchorTrimArea  (5%)  — cap for trimming inflated anchor bbox
        //  matchedUnionArea (15%) — generous cap for already-scored matched contours
        //  siblingExpArea   (5%)  — tight cap for unverified sibling expansion
        double refMaxDim = Math.max(refFullBbox.width, refFullBbox.height);
        double anchorTrimSide = refMaxDim * estimatedScale * 1.05;
        double anchorTrimArea = anchorTrimSide * anchorTrimSide;

        double matchedUnionW = refFullBbox.width * estimatedScale * 1.15;
        double matchedUnionH = refFullBbox.height * estimatedScale * 1.15;
        double matchedUnionArea = Math.max(matchedUnionW * matchedUnionH,
                anchorTrimArea * 1.15);

        double siblingExpArea = anchorTrimArea;

        return new AreaCaps(anchorTrimArea, matchedUnionArea, siblingExpArea);
    }

    private static Rect trimAndUnionMatched(Rect anchorBbox,
                                            List<SceneContourEntry> matched,
                                            SceneContourEntry bestAnchor,
                                            List<RefCluster> refClusters,
                                            AreaCaps caps,
                                            String referenceId,
                                            double anchorDiag) {
        // Start from anchor bbox, but trim if already exceeds anchorTrimArea
        Rect expandedBbox;
        double anchorBoxArea = (double) anchorBbox.width * anchorBbox.height;

        if (anchorBoxArea <= caps.anchorTrimArea) {
            expandedBbox = new Rect(anchorBbox.x, anchorBbox.y,
                    anchorBbox.width, anchorBbox.height);
        } else {
            // Trim proportionally, keeping centroid fixed
            double trimRatio = Math.sqrt(caps.anchorTrimArea / anchorBoxArea);
            int tw = (int) (anchorBbox.width * trimRatio);
            int th = (int) (anchorBbox.height * trimRatio);
            int cx = anchorBbox.x + anchorBbox.width / 2;
            int cy = anchorBbox.y + anchorBbox.height / 2;
            expandedBbox = new Rect(cx - tw / 2, cy - th / 2, tw, th);
        }

        // Determine cap based on chromatic diversity
        boolean hasDistinctChromaticHues = hasDistinctChromaticHues(refClusters);
        double stepCCap = hasDistinctChromaticHues ? caps.matchedUnionArea : caps.anchorTrimArea;

        if (VM_BBOX_DEBUG) {
            List<RefCluster> chromatics = refClusters.stream()
                    .filter(rc -> !rc.achromatic).toList();
            System.out.printf("[STEP-C] %s chromaticClusters=%d hues=%s distinctHues=%b stepCCap=%.0f%n",
                    referenceId, chromatics.size(),
                    chromatics.stream().map(rc -> String.format("%.1f", rc.hue)).toList(),
                    hasDistinctChromaticHues, stepCCap);
        }

        // Union with co-located matched contours
        for (SceneContourEntry m : matched) {
            if (m == bestAnchor) continue;
            Rect mBb = m.bbox();
            double dist = GeometryUtils.centreDist(anchorBbox, mBb);
            if (dist <= anchorDiag * 0.50) {
                Rect candidate = GeometryUtils.unionRect(expandedBbox, mBb);
                if ((double) candidate.width * candidate.height <= stepCCap) {
                    expandedBbox = candidate;
                }
            }
        }

        return expandedBbox;
    }

    private static boolean hasDistinctChromaticHues(List<RefCluster> refClusters) {
        List<RefCluster> chromatics = refClusters.stream()
                .filter(rc -> !rc.achromatic).toList();

        for (int ci = 0; ci < chromatics.size(); ci++) {
            for (int cj = ci + 1; cj < chromatics.size(); cj++) {
                double diff = Math.abs(chromatics.get(ci).hue - chromatics.get(cj).hue);
                if (Math.min(diff, 180.0 - diff) > 20.0) {
                    // Also require spatially distinct regions
                    Rect bi = chromatics.get(ci).primaryBbox();
                    Rect bj = chromatics.get(cj).primaryBbox();
                    if (GeometryUtils.bboxIoU(bi, bj) < 0.50) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    /**
     * Unions same-cluster siblings of each matched entry into {@code expandedBbox}
     * when the sibling is spatially related to the current result region:
     *
     * <ul>
     *   <li><b>Concentric:</b> sibling centre is within 35% of the sibling's own
     *       diagonal from the expandedBbox centre (concentric rings, bullseye).</li>
     *   <li><b>Overlapping:</b> sibling bbox intersects the current expandedBbox
     *       (adjacent components, enclosed pieces).</li>
     * </ul>
     *
     * <p>Hard cap: the resulting bbox area must not exceed {@code maxAllowedArea}.
     * A tightly-concentric override (centre distance ≤ 5% of anchor diagonal,
     * area ≤ 3× current) allows compound-shape outer rings (e.g. COMPOUND_BULLSEYE)
     * to expand past the normal cap.
     */
    private static Rect unionConcentricAndOverlappingSiblings(Rect expandedBbox,
                                                               List<SceneContourEntry> matched,
                                                               List<SceneContourEntry> candidates,
                                                               double maxAllowedArea) {

        for (int pass = 0; pass < 4; pass++) {
            Rect before = expandedBbox;
            for (SceneContourEntry m : matched) {
                for (SceneContourEntry ce : candidates) {
                    if (ce.clusterIdx() != m.clusterIdx() || ce == m) continue;

                    Rect   ceBb         = ce.bbox();
                    double ceArea       = ce.area();
                    double expandedArea = (double) expandedBbox.width * expandedBbox.height;

                    // Already at cap — no point continuing
                    if (expandedArea >= maxAllowedArea) break;

                    // Guard: sibling must be substantial relative to current bbox
                    if (ceArea < expandedArea * 0.10) continue;

                    // Criterion 1 — concentric
                    double ceDiag      = Math.hypot(ceBb.width, ceBb.height);
                    boolean concentric = GeometryUtils.centreDist(expandedBbox, ceBb) <= ceDiag * 0.35;

                    // Criterion 2 — overlapping
                    boolean overlapping = GeometryUtils.rectsIntersect(expandedBbox, ceBb);

                    if (concentric || overlapping) {
                        Rect candidate     = GeometryUtils.unionRect(expandedBbox, ceBb);
                        double candidateArea = (double) candidate.width * candidate.height;
                        if (candidateArea <= maxAllowedArea) {
                            expandedBbox = candidate;
                        } else if (concentric) {
                            // Tightly concentric override — allows outer rings of compound
                            // shapes (e.g. COMPOUND_BULLSEYE) to expand past the normal cap.
                            double ancDiag = Math.hypot(expandedBbox.width, expandedBbox.height);
                            double dist    = GeometryUtils.centreDist(expandedBbox, ceBb);
                            if (dist <= ancDiag * 0.05
                                    && candidateArea <= expandedArea * 3.0) {
                                expandedBbox = candidate;
                            }
                        }
                    }
                }
            }
            if (expandedBbox.x == before.x && expandedBbox.y == before.y
                    && expandedBbox.width == before.width
                    && expandedBbox.height == before.height) break;
        }
        return expandedBbox;
    }

    private record AreaCaps(double anchorTrimArea, double matchedUnionArea, double siblingExpArea) {}

    private BboxExpander() {}  // static utility class
}




