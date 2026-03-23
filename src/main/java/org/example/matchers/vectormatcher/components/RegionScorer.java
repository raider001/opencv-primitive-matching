            // Find best scene entry for this ref contour
package org.example.matchers.vectormatcher.components;

import org.example.matchers.SceneContourEntry;
import org.example.matchers.SceneDescriptor;
import org.example.matchers.VectorSignature;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;

import java.util.List;

/**
 * Three-layer scoring for a candidate matched set inside a scene region.
 *
 * <p>Extracted from {@code VectorMatcher.scoreRegion()} to make each scoring
 * layer independently testable.  All constants and formulas are transplanted
 * verbatim — this class produces <b>bit-identical</b> results.
 *
 * <h2>Three-layer scoring</h2>
 * <ol>
 *   <li><b>Layer 1 — Boundary count.</b>  Exponential penalty when the candidate
 *       region has more or fewer distinct boundaries than the reference.</li>
 *   <li><b>Layer 2 — Structural coherence.</b>  Spatial proximity and relative size
 *       consistency of each matched boundary, weighted by structural role.</li>
 *   <li><b>Layer 3 — Geometry.</b>  {@link VectorSignature#similarity} on every
 *       ref contour × every matched scene entry — best score wins.</li>
 * </ol>
 */
public final class RegionScorer {

    // ── Layer weights — must sum to 1.0 ──────────────────────────────────────
    static final double W_COUNT = 0.05;   // Layer 1: boundary count match (minimal — over-segmentation from noisy backgrounds)
    static final double W_MATCH = 0.25;   // Layer 2: structural coherence
    static final double W_GEOM  = 0.70;   // Layer 3: primary geometry (dominant discriminator)

    // ── Layer 1 constants ─────────────────────────────────────────────────────
    /** Exponential decay rate for extra boundaries (scene has more than ref). */
    static final double CLUSTER_PENALTY_K      = 0.10;
    /**
     * Decay rate for missing boundaries (scene has fewer than ref).
     * Kept gentler because border-pixel extraction can split/merge one boundary
     * differently between 128×128 refs and larger scene renders.
     */
    static final double CLUSTER_PENALTY_K_MISS = 0.10;
    /** Minimum Layer-1 score when only missing-boundary mismatch is present. */
    static final double MIN_COUNT_SCORE_MISS   = 0.45;

    // ── Layer 2 constants ─────────────────────────────────────────────────────
    /** Weight of secondary (non-primary) boundaries in Layer 2. */
    static final double SECONDARY_WEIGHT = 0.30;

    private RegionScorer() {}

    // =========================================================================
    // Public entry point
    // =========================================================================

    /**
     * Three-layer score for a candidate matched set.
     *
     * @return a {@link RegionScore} with combined and per-layer scores, all in [0,1]
     */
    public static RegionScore score(List<RefCluster> refClusters,
                                    int refCount,
                                    RefCluster primaryRef,
                                    List<SceneContourEntry> matched,
                                    SceneDescriptor descriptor,
                                    Rect regionBbox,
                                    boolean allAchromatic,
                                    double sceneDiag,
                                    double epsilon) {

        // ── Layer 1: boundary count ───────────────────────────────────────
        double countScore = scoreBoundaryCount(matched.size(), refCount);

        // ── Fix B: chromatic contamination (achromatic refs only) ─────────
        countScore = applyContaminationPenalty(countScore, allAchromatic,
                descriptor, regionBbox);

        // ── Layer 2: structural coherence ────────────────────────────────
        double matchScore = scoreStructuralCoherence(refClusters, primaryRef,
                matched, regionBbox, sceneDiag);

        // ── Layer 3: primary boundary geometry ───────────────────────────
        double geomScore = scoreGeometry(refClusters, matched, epsilon);

        // If geometry and structural coherence are both near-perfect, do not let
        // count-only over-segmentation suppress an otherwise clear true match.
        if (geomScore >= 0.95 && matchScore >= 0.95) {
            countScore = Math.max(countScore, 0.45);
        }

        double combined = countScore * W_COUNT
                         + matchScore * W_MATCH
                         + geomScore  * W_GEOM;

        combined = Math.max(0.0, Math.min(1.0, combined));

        return new RegionScore(combined, countScore, matchScore, geomScore);
    }

    // =========================================================================
    // Layer 1 — Boundary count
    // =========================================================================

    /**
     * Computes the Layer-1 boundary-count score.
     *
     * <p>Exponential penalty for extra boundaries (over-segmentation) or
     * missing boundaries (under-segmentation), with a floor for the
     * missing-boundary case.
     *
     * @param matchedCount number of scene boundaries in the matched set
     * @param refCount     number of reference boundaries expected
     * @return score in [0, 1]
     */
    static double scoreBoundaryCount(int matchedCount, int refCount) {
        if (refCount == 0) {
            return 1.0;
        }
        int diff = matchedCount - refCount;
        if (diff > 0) {
            return Math.exp(-CLUSTER_PENALTY_K * diff);              // extra — over-segmentation
        } else {
            double score = Math.exp(-CLUSTER_PENALTY_K_MISS * Math.abs(diff));
            return Math.max(score, MIN_COUNT_SCORE_MISS);
        }
    }

    // =========================================================================
    // Chromatic contamination penalty (Fix B)
    // =========================================================================

    /**
     * Applies a chromatic-contamination penalty to the count score for
     * achromatic-only references.
     *
     * <p>Achromatic references (white/grey on black) should not score highly on
     * regions dominated by chromatic content.  The check measures the fraction
     * of chromatic pixels inside the candidate bbox.
     *
     * <ul>
     *   <li>Floor (0.70): random-line backgrounds produce 40–80+ % chromatic
     *       coverage inside a bbox even when the detected shape itself is purely
     *       achromatic.  Only extreme contamination (&gt; 70 %) is penalised.</li>
     *   <li>Cap (0.30): the contamination-adjusted countScore is floored at 0.30
     *       so background noise never destroys Layer 1.</li>
     * </ul>
     *
     * @param countScore    current Layer-1 score
     * @param allAchromatic true if every ref cluster is achromatic
     * @param descriptor    scene descriptor (provides combinedChromaticMask)
     * @param regionBbox    bounding box of the candidate region
     * @return adjusted count score (may be unchanged)
     */
    static double applyContaminationPenalty(double countScore,
                                            boolean allAchromatic,
                                            SceneDescriptor descriptor,
                                            Rect regionBbox) {
        if (allAchromatic && descriptor.combinedChromaticMask != null) {
            double bboxArea = (double) regionBbox.width * regionBbox.height;
            if (bboxArea > 0) {
                Rect clamped = clampRect(regionBbox, descriptor.combinedChromaticMask);
                if (clamped.width > 0 && clamped.height > 0) {
                    Mat roi = descriptor.combinedChromaticMask.submat(clamped);
                    double contamination = Core.countNonZero(roi) / bboxArea;
                    double CONTAM_FLOOR = 0.70;
                    if (contamination > CONTAM_FLOOR) {
                        double excess = (contamination - CONTAM_FLOOR) / (1.0 - CONTAM_FLOOR);
                        countScore *= Math.pow(1.0 - excess, 1.5);
                    }
                    // Never let contamination alone crush countScore below 0.30
                    countScore = Math.max(countScore, 0.30);
                }
            }
        }
        return countScore;
    }

    // =========================================================================
    // Layer 2 — Structural coherence
    // =========================================================================

    /**
     * Computes the Layer-2 structural-coherence score.
     *
     * <p>For each reference cluster, finds the best-matching scene entry and
     * evaluates spatial proximity, bbox-fill coverage, and absolute scale
     * agreement.  The primary cluster carries full weight (1.0); secondary
     * clusters carry {@value #SECONDARY_WEIGHT}.
     *
     * @param refClusters all reference clusters
     * @param primaryRef  the primary (largest-area) reference cluster
     * @param matched     scene contour entries in the matched set
     * @param regionBbox  bounding box of the candidate region
     * @param sceneDiag   scene diagonal (for proximity normalisation)
     * @return score in [0, 1]
     */
    static double scoreStructuralCoherence(List<RefCluster> refClusters,
                                           RefCluster primaryRef,
                                           List<SceneContourEntry> matched,
                                           Rect regionBbox,
                                           double sceneDiag) {
        double sumContrib = 0.0;
        double sumWeights = 0.0;

        for (RefCluster rc : refClusters) {
            double weight = (rc == primaryRef) ? 1.0 : SECONDARY_WEIGHT;
            sumWeights += weight;

            // Find best matched entry for this ref cluster by relative area closeness
            SceneContourEntry entry = findMatchedEntryForRef(rc, matched);
            if (entry == null) {
                // Missing cluster — zero contribution, weight still in denominator
                continue;
            }

            Rect   entryBb     = entry.bbox();
            double dist         = GeometryUtils.centreDist(regionBbox, entryBb);
            double proxScore    = Math.max(0.0, 1.0 - dist / (sceneDiag * 0.30));

            // Coverage score: compare shape-fill ratios normalised by each
            // contour's OWN bounding box — scale and position invariant.
            // Both sides converge to π/4 for circles, 1.0 for rectangles, etc.
            // Using refImageArea for the ref denominator causes a systematic
            // mismatch (circle fills 30% of 128×128 image but 78.5% of its bbox).
            //
            // LINE_SEGMENT guard: when either the ref or scene contour has extreme
            // aspect ratio (> 4:1), the axis-aligned bounding box is unreliable for
            // coverage/scale comparison — a thin line rotated 45° has an AABB much
            // larger than its actual extent.  Use contour area directly for scale
            // comparison and skip covScore (which compares fill fractions).
            Rect   refBb        = rc.primaryBbox;
            double refBbArea    = Math.max(1.0, (double) refBb.width * refBb.height);
            double entryBbArea  = Math.max(1.0, (double) entryBb.width * entryBb.height);

            double refBbAR      = Math.max(refBb.width, refBb.height)
                                / Math.max(1.0, Math.min(refBb.width, refBb.height));
            double entryBbAR    = Math.max(entryBb.width, entryBb.height)
                                / Math.max(1.0, Math.min(entryBb.width, entryBb.height));
            boolean lineSegLike = refBbAR > 4.0 || entryBbAR > 4.0;

            double covScore;
            double scaleScore = 1.0;

            if (lineSegLike) {
                // For LINE_SEGMENT-like shapes: use contour area ratio for scale.
                // AABB-based coverage is meaningless for rotated thin shapes.
                covScore = 1.0;   // skip — contour fill ratio varies with rotation
                double refArea2   = Math.max(1.0, rc.maxContourArea);
                double sceneArea2 = Math.max(1.0, entry.area());
                double areaSizeRatio = Math.sqrt(sceneArea2 / refArea2);
                if (areaSizeRatio > 6.0 || areaSizeRatio < (1.0/6.0)) {
                    scaleScore = 0.25;
                } else if (areaSizeRatio > 4.5 || areaSizeRatio < (1.0/4.5)) {
                    scaleScore = 0.70;
                }
            } else {
                double refFrac   = rc.maxContourArea / refBbArea;
                double sceneFrac = entry.area() / entryBbArea;
                covScore  = 1.0 - Math.min(1.0,
                        Math.abs(refFrac - sceneFrac) / Math.max(refFrac, 0.01));

                // Absolute scale check: penalize when bbox sizes differ significantly.
                // Even if shapes are geometrically similar (diamond vs rotated rect),
                // extreme scale differences may indicate wrong match.
                // Uses square-root ratio to account for area growing quadratically.
                // Expected scale: scene is typically 3x scaled from 128x128 ref → ratio ≈ 3.0
                double refBboxDim   = Math.sqrt(refBbArea);
                double sceneBboxDim = Math.sqrt(entryBbArea);
                double sizeRatio    = sceneBboxDim / Math.max(refBboxDim, 1.0);
                if (sizeRatio > 6.0 || sizeRatio < (1.0/6.0)) {
                    scaleScore = 0.25;
                } else if (sizeRatio > 4.5 || sizeRatio < (1.0/4.5)) {
                    scaleScore = 0.70;
                }
            }

            // ── Topological mismatch penalties (cross-ref rejection hardening) ───
            // If the scene entry's VectorSignature differs significantly in
            // vertex count, circularity, or shape type from the best ref sig,
            // apply a multiplicative penalty to this cluster's contribution.
            double topoMultiplier = 1.0;
            
            VectorSignature bestRefSig = rc.bestSig;
            VectorSignature sceneSig   = entry.sig();
            
            if (bestRefSig != null && sceneSig != null) {
                // 1. Vertex count mismatch (for low-vertex shapes ≤ 10)
                if (bestRefSig.vertexCount <= 10 && sceneSig.vertexCount <= 10 
                    && bestRefSig.vertexCount > 0 && sceneSig.vertexCount > 0) {
                    double vtxRatio = Math.min(bestRefSig.vertexCount, sceneSig.vertexCount)
                                    / (double) Math.max(bestRefSig.vertexCount, sceneSig.vertexCount);
                    if (vtxRatio < 0.80) {
                        // e.g. 3 vs 5 → ratio=0.6 → penalty^2.5 = 0.22
                        topoMultiplier *= Math.pow(vtxRatio, 2.5);
                    }
                }
                
                // 2. Circularity mismatch (CIRCLE vs polygon discrimination)
                double circDiff = Math.abs(bestRefSig.circularity - sceneSig.circularity);
                if (circDiff > 0.25) {
                    // Large circularity gap (e.g. circle=0.95 vs rect=0.78) → 0.75 multiplier
                    topoMultiplier *= (1.0 - Math.min(0.35, circDiff * 0.70));
                }
                
                // 3. Shape type hard gates (CIRCLE vs POLY, CONVEX vs CONCAVE)
                boolean typeHardMismatch = false;
                if (bestRefSig.type == VectorSignature.ShapeType.CIRCLE 
                    && sceneSig.type != VectorSignature.ShapeType.CIRCLE) {
                    typeHardMismatch = true;
                } else if (bestRefSig.type != VectorSignature.ShapeType.CIRCLE 
                           && sceneSig.type == VectorSignature.ShapeType.CIRCLE) {
                    typeHardMismatch = true;
                } else if (bestRefSig.type == VectorSignature.ShapeType.CLOSED_CONVEX_POLY 
                           && sceneSig.type == VectorSignature.ShapeType.CLOSED_CONCAVE_POLY) {
                    typeHardMismatch = true;
                } else if (bestRefSig.type == VectorSignature.ShapeType.CLOSED_CONCAVE_POLY 
                           && sceneSig.type == VectorSignature.ShapeType.CLOSED_CONVEX_POLY) {
                    typeHardMismatch = true;
                }
                
                if (typeHardMismatch) {
                    topoMultiplier *= 0.50;  // 50% suppression for hard type mismatch
                }
                
                // 4. Concavity mismatch (star vs circle, etc.)
                double concavityDiff = Math.abs(bestRefSig.concavityRatio - sceneSig.concavityRatio);
                if (concavityDiff > 0.10) {
                    // Significant concavity difference → suppress
                    topoMultiplier *= (1.0 - Math.min(0.30, concavityDiff * 2.0));
                }
            }

            sumContrib += weight * topoMultiplier * (proxScore * 0.40 + covScore * 0.40 + scaleScore * 0.20);
        }

        return (sumWeights > 0) ? sumContrib / sumWeights : 0.0;
    }

    // =========================================================================
    // Layer 3 — Primary boundary geometry
    // =========================================================================

    // ── Debug flag for CAS diagnostic — read once at class load ──────────────
    private static final boolean CAS_DEBUG = System.getProperty("vm.cas.debug") != null;

    /**
     * Computes the Layer-3 geometry score.
     *
     * <p>Tries ALL contour signatures from every ref cluster against every matched
     * scene entry.  Using only bestSig (max solidity) per cluster can miss the
     * correct pairing in compound shapes — e.g. a circle-outline cluster's
     * bestSig might be a co-located triangle (higher solidity), causing the
     * circle-ref to compare against the triangle scene contour instead of the
     * circle scene contour.
     *
     * <p>Using the best geometry across ALL ref contours × ALL scene entries is
     * purely structural — it finds the best structural alignment regardless of
     * cluster origin or solidity ranking.
     *
     * <p><b>BAS boost (boundary alignment):</b> When the best VectorSignature
     * geometry score is degraded (&lt; 0.60) but boundary alignment shows the
     * reference's vertices lie on the scene contour boundary (BAS boundary
     * ≥ 0.75), the geometry score is floored at {@code boundaryMatch × 0.70}.
     * This recovers from background-contaminated contours (e.g. IRREGULAR_QUAD
     * on BG_RANDOM_LINES) where extra vertices from background line merges
     * degrade VectorSignature.similarity() but the physical shape corners are
     * still present on the contour boundary.  Only fires for CLOSED_CONVEX_POLY
     * with 3–12 vertices — BAS is unreliable for circles, lines, and high-vertex
     * shapes.
     *
     * <p><b>CAS diagnostic:</b> When {@code -Dvm.cas.debug} is set, also logs
     * vertex-to-vertex CAS and boundary BAS for the best-scoring pair.
     *
     * @param refClusters all reference clusters
     * @param matched     scene contour entries in the matched set
     * @param epsilon     epsilon factor for VectorSignature building
     * @return score in [0, 1]
     */
    static double scoreGeometry(List<RefCluster> refClusters,
                                List<SceneContourEntry> matched,
                                double epsilon) {
        double bestGeom = -1.0;
        MatOfPoint bestRefContour   = null;
        SceneContourEntry bestScene = null;
        VectorSignature bestRefSig  = null;

        // Also track the best BAS-boosted geometry across ALL pairs
        double bestBasBoost = -1.0;

        for (RefCluster rc : refClusters) {
            for (MatOfPoint refContour : rc.contours) {
                VectorSignature rcSig = VectorSignature.buildFromContour(
                        refContour, epsilon, Double.NaN);
                for (SceneContourEntry e : matched) {
                    double sim = rcSig.similarity(e.sig());
                    if (sim > bestGeom) {
                        bestGeom       = sim;
                        bestRefContour = refContour;
                        bestScene      = e;
                        bestRefSig     = rcSig;
                    }

                    // ── BAS probe: check boundary alignment for degraded pairs ──
                    // Only compute BAS when:
                    // 1. VectorSignature is degraded (sim < 0.60) — contaminated
                    // 2. Ref is a convex polygon with well-defined vertices (3–12)
                    // 3. Scene is also a convex polygon (not circle/line/compound)
                    // 4. Scene has at least as many vertices as ref (contamination
                    //    adds vertices, doesn't remove them) and not too many more
                    //    (≤ 2× ref count — beyond that it's a different shape)
                    VectorSignature eSig = e.sig();
                    if (sim < 0.60
                            && rcSig.type == VectorSignature.ShapeType.CLOSED_CONVEX_POLY
                            && rcSig.vertexCount >= 3 && rcSig.vertexCount <= 12
                            && eSig != null
                            && eSig.type == VectorSignature.ShapeType.CLOSED_CONVEX_POLY
                            && eSig.vertexCount >= rcSig.vertexCount
                            && eSig.vertexCount <= rcSig.vertexCount * 2) {
                        var bas = GeometryUtils.computeBoundaryAlignment(
                                refContour, e.contour(), epsilon);
                        // Require BOTH high boundary match AND strong angle agreement.
                        // Angle agreement ≥ 0.65 separates true contamination cases
                        // (angle ~0.77 for IRREGULAR_QUAD) from regular polygon false
                        // matches where vertices trivially land on a higher-order
                        // polygon's boundary (hexagon→octagon: angle ~0.57).
                        if (bas != null
                                && bas.boundaryMatch() >= 0.75
                                && bas.angleMatch() >= 0.65) {
                            // Floor: combined × 0.70 — uses both position AND angle
                            double boost = bas.combined() * 0.70;
                            if (boost > bestBasBoost) {
                                bestBasBoost = boost;
                            }
                        }
                    }
                }
            }
        }

        // ── Apply BAS boost if it exceeds the best VectorSignature score ──
        double finalGeom = (bestGeom >= 0.0) ? bestGeom : 0.0;
        if (bestBasBoost > finalGeom) {
            if (CAS_DEBUG) {
                System.out.printf("[BAS-BOOST] geom %.3f → %.3f%n", finalGeom, bestBasBoost);
            }
            finalGeom = bestBasBoost;
        }

        // ── CAS/BAS diagnostic probe ────────────────────────────────────
        if (CAS_DEBUG && bestRefContour != null) {
            var cas = GeometryUtils.computeAlignment(
                    bestRefContour, bestScene.contour(), epsilon);
            var bas = GeometryUtils.computeBoundaryAlignment(
                    bestRefContour, bestScene.contour(), epsilon);
            VectorSignature sSig = bestScene.sig();
            String refInfo = bestRefSig.type + "(v" + bestRefSig.vertexCount
                    + ",cv" + String.format("%.3f", bestRefSig.edgeLengthCV) + ")";
            String sceneInfo = (sSig != null ? sSig.type : "?") + "(v"
                    + (sSig != null ? sSig.vertexCount : 0) + ",cv"
                    + String.format("%.3f", sSig != null ? sSig.edgeLengthCV : 0.0) + ")";
            if (cas != null) {
                System.out.printf("[CAS-DEBUG] geom=%.3f | pos=%.3f angle=%.3f cas=%.3f | rot=%.1f° scale=%.2f | ref=%s scene=%s%n",
                        bestGeom, cas.positionMatch(), cas.angleMatch(),
                        cas.combined(), cas.rotationDeg(), cas.scale(),
                        refInfo, sceneInfo);
            }
            if (bas != null) {
                System.out.printf("[BAS-DEBUG] geom=%.3f | bnd=%.3f angle=%.3f bas=%.3f | rot=%.1f° scale=%.2f | ref=%s scene=%s%n",
                        bestGeom, bas.boundaryMatch(), bas.angleMatch(),
                        bas.combined(), bas.rotationDeg(), bas.scale(),
                        refInfo, sceneInfo);
            }
        }

        return finalGeom;
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    /**
     * Finds the matched entry that best represents the given ref cluster,
     * using bbox-normalised fill ratios for both sides so scale differences
     * between ref (128×128) and scene (640×480) do not affect assignment.
     */
    static SceneContourEntry findMatchedEntryForRef(RefCluster rc,
                                                    List<SceneContourEntry> matched) {
        Rect   refBb     = rc.primaryBbox;
        double refBbArea = Math.max(1.0, (double) refBb.width * refBb.height);
        double refFrac   = rc.maxContourArea / refBbArea;

        SceneContourEntry best    = null;
        double            bestDiff = Double.MAX_VALUE;
        for (SceneContourEntry e : matched) {
            Rect   eBb   = e.bbox();
            double eBbA  = Math.max(1.0, (double) eBb.width * eBb.height);
            double eFrac = e.area() / eBbA;
            double diff  = Math.abs(refFrac - eFrac);
            if (diff < bestDiff) { bestDiff = diff; best = e; }
        }
        return best;
    }

    /**
     * Clamps a rectangle to fit within the given image dimensions.
     *
     * @param r   rectangle to clamp
     * @param img image whose dimensions define the clamp bounds
     * @return clamped rectangle (may have zero width/height if entirely outside)
     */
    static Rect clampRect(Rect r, Mat img) {
        int x  = Math.max(0, r.x);
        int y  = Math.max(0, r.y);
        int x2 = Math.min(img.cols(), r.x + r.width);
        int y2 = Math.min(img.rows(), r.y + r.height);
        return new Rect(x, y, Math.max(0, x2 - x), Math.max(0, y2 - y));
    }
}

