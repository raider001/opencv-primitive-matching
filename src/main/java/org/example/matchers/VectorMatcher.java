package org.example.matchers;

import org.example.analytics.AnalysisResult;
import org.example.colour.ColourCluster;
import org.example.colour.ExperimentalSceneColourClusters;
import org.example.colour.SceneColourClusters;
import org.example.factories.ReferenceId;
import org.example.matchers.vectormatcher.*;
import org.example.scene.SceneEntry;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * VectorMatcher — structural pattern matching via colour-edge cluster decomposition.
 *
 * <h2>Design principles</h2>
 * <ol>
 *   <li><b>Structural, not colour, matching.</b>  Hue identifies distinct boundary
 *       clusters during discovery only.  After that, hue plays no role in selection,
 *       matching, or scoring.</li>
 *   <li><b>Identical pipeline for ref and scene.</b>  Both go through
 *       {@link SceneColourClusters#extractFromBorderPixels} so their cluster
 *       representations are comparable.</li>
 *   <li><b>Selection and scoring are independent.</b>  The expansion loop selects
 *       candidate clusters using only spatial proximity and relative contour size.
 *       Geometry scoring happens exclusively in Layer 3.</li>
 * </ol>
 *
 * <h2>Three-layer scoring</h2>
 * <ol>
 *   <li><b>Layer 1 — Boundary count.</b>  Exponential penalty when the candidate
 *       region has more or fewer distinct boundaries than the reference.</li>
 *   <li><b>Layer 2 — Structural coherence.</b>  Spatial proximity and relative size
 *       consistency of each matched boundary, weighted by structural role.</li>
 *   <li><b>Layer 3 — Geometry.</b>  {@link VectorSignature#similarity} on the
 *       primary boundary (largest contour area in the reference) only.</li>
 * </ol>
 *
 * <h2>Single variant</h2>
 * <p>Runs once at ε = 4 % of perimeter (VECTOR_NORMAL).  Three variants are no
 * longer needed: 87 % of the geometry score is epsilon-independent, and the
 * expansion loop no longer uses geometry for candidate selection.
 */
public final class VectorMatcher {

    // ── Layer weights — must sum to 1.0 ──────────────────────────────────────
    private static final double W_COUNT = 0.15;   // Layer 1: boundary count match
    private static final double W_MATCH = 0.25;   // Layer 2: structural coherence
    private static final double W_GEOM  = 0.60;   // Layer 3: primary geometry

    // ── Layer 1 constants ─────────────────────────────────────────────────────
    /** Exponential decay rate for extra boundaries (scene has more than ref). */
    private static final double CLUSTER_PENALTY_K      = 0.10;
    /**
     * Decay rate for missing boundaries (scene has fewer than ref).
     * Kept gentler because border-pixel extraction can split/merge one boundary
     * differently between 128x128 refs and larger scene renders.
     */
    private static final double CLUSTER_PENALTY_K_MISS = 0.10;
    /** Minimum Layer-1 score when only missing-boundary mismatch is present. */
    private static final double MIN_COUNT_SCORE_MISS   = 0.45;

    // ── Layer 2 constants ─────────────────────────────────────────────────────
    /** Weight of secondary (non-primary) boundaries in Layer 2. */
    private static final double SECONDARY_WEIGHT = 0.30;

    // ── Deduplication constants ───────────────────────────────────────────────
    /**
     * Min bbox IoU to treat two boundaries as the same physical edge.
     * Kept high so inner/outer outline rings are not collapsed.
     */
    private static final double DEDUP_IOU_MIN        = 0.90;
    /**
     * Min contour-area ratio (min/max) for same-edge confirmation.
     * Lower than 0.90 so anti-aliased filled polygons still deduplicate.
     */
    private static final double DEDUP_AREA_RATIO_MIN = 0.75;

    // ── Geometry constant ─────────────────────────────────────────────────────
    private static final double EPSILON = 0.04;   // single variant epsilon

    // ── Debug flag — read once at class load ─────────────────────────────
    private static final boolean VM_DEBUG = System.getProperty("vm.debug") != null;

    private VectorMatcher() {}

    // =========================================================================
    // Public entry points
    // =========================================================================

    public static List<AnalysisResult> match(ReferenceId referenceId,
                                             Mat refMat,
                                             SceneEntry scene,
                                             Set<String> saveVariants,
                                             Path outputDir) {
        SceneDescriptor descriptor = scene.descriptor();
        if (descriptor != null) {
            return matchWithDescriptor(referenceId, refMat, scene, descriptor, saveVariants, outputDir);
        }
        SceneDescriptor temp = SceneDescriptor.build(scene.sceneMat());
        try {
            return matchWithDescriptor(referenceId, refMat, scene, temp, saveVariants, outputDir);
        } finally {
            temp.release();
        }
    }

    public static List<AnalysisResult> match(ReferenceId referenceId,
                                             Mat refMat,
                                             SceneEntry scene,
                                             SceneDescriptor descriptor,
                                             Set<String> saveVariants,
                                             Path outputDir) {
        return matchWithDescriptor(referenceId, refMat, scene, descriptor, saveVariants, outputDir);
    }

    // =========================================================================
    // Core matching
    // =========================================================================

    private static List<AnalysisResult> matchWithDescriptor(ReferenceId referenceId,
                                                             Mat refMat,
                                                             SceneEntry scene,
                                                             SceneDescriptor descriptor,
                                                             Set<String> saveVariants,
                                                             Path outputDir) {
        List<RefCluster> refClusters = buildRefClusters(refMat);

        // Run single NORMAL variant — three variants no longer add value
        VectorVariant variant = VectorVariant.VECTOR_NORMAL;
        AnalysisResult result = runMatch(variant, refClusters,
                descriptor, scene, referenceId, saveVariants, outputDir);

        for (RefCluster rc : refClusters) rc.release();

        // Return as a single-element list — one variant, one result
        return List.of(result);
    }

    // =========================================================================
    // Match execution
    // =========================================================================

    private static AnalysisResult runMatch(VectorVariant variant,
                                           List<RefCluster> refClusters,
                                           SceneDescriptor descriptor,
                                           SceneEntry scene,
                                           ReferenceId referenceId,
                                           Set<String> saveVariants,
                                           Path outputDir) {
        long t0 = System.currentTimeMillis();
        try {
            List<SceneContourEntry> candidates = collectSceneCandidates(descriptor);
            if (candidates.isEmpty()) {
                return zeroResult(variant, referenceId, scene, descriptor, t0);
            }

            // Identify the primary ref cluster (largest contour area) once — reused
            RefCluster primaryRef = findPrimaryCluster(refClusters);
            int        refCount   = refClusters.size();

            // Pre-compute invariants used per-anchor (avoid recomputing in hot loop)
            boolean allAchromatic = true;
            for (RefCluster rc : refClusters) { if (!rc.achromatic) { allAchromatic = false; break; } }

            // Pre-compute the primary ref signature for early-exit type checks (OPT-G)
            VectorSignature primaryRefSig = primaryRef.bestSig(EPSILON);

            // ── Stage 1: Connected-component filter ────────────────────────────
            candidates = CandidateFilter.applyConnectedComponentFilter(candidates);
            if (candidates.isEmpty()) {
                return zeroResult(variant, referenceId, scene, descriptor, t0);
            }

            // ── Stage 1b: Global minimum-size filter ───────────────────────────
            candidates = CandidateFilter.applyGlobalSizeFilter(candidates);
            if (candidates.isEmpty()) {
                return zeroResult(variant, referenceId, scene, descriptor, t0);
            }

            // ── Stage 2: Reference-adaptive morphological opening ──────────────
            // Skipped for outline/line references (computeErosionDepth always returns 0).
            int erosionDepth = CandidateFilter.computeErosionDepth(primaryRef);
            if (erosionDepth > 0) {
                candidates = CandidateFilter.reExtractTopCandidates(candidates, scene, erosionDepth, descriptor);
            }

            // ── OPT-R: Build signatures only for surviving candidates ───
            // Deferred from collectSceneCandidates so contours eliminated by
            // CC filter, global size filter, or re-extraction skip the expensive
            // VectorSignature.buildFromContour work entirely.
            candidates = buildSignatures(candidates, descriptor.sceneArea);

            // Scene diagonal — derived from combinedChromaticMask dimensions if available,
            // otherwise estimated from sceneArea (assumes roughly square scene).
            double sceneW, sceneH;
            if (descriptor.combinedChromaticMask != null) {
                sceneW = descriptor.combinedChromaticMask.cols();
                sceneH = descriptor.combinedChromaticMask.rows();
            } else {
                sceneW = sceneH = Math.sqrt(descriptor.sceneArea);
            }
            double sceneDiag = Math.sqrt(sceneW * sceneW + sceneH * sceneH);

            double bestScore  = 0.0;
            double[] bestLayerScores = new double[]{0, 0, 0}; // [count, match, geom] for best anchor
            Rect   bestBbox   = null;
            SceneContourEntry bestAnchor = null;

            // Track all scored anchors for post-loop re-selection
            List<double[]>          anchorScores      = new ArrayList<>();
            List<Rect>              anchorBboxes      = new ArrayList<>();
            List<SceneContourEntry> anchorEntries     = new ArrayList<>();
            List<List<SceneContourEntry>> anchorMatchedSets = new ArrayList<>();

            for (SceneContourEntry anchor : candidates) {
                Rect anchorBbox = anchor.bbox();

                // ── Anchor-to-ref assignment ──────────────────────────────
                double anchorBboxArea = Math.max(1.0, (double) anchorBbox.width * anchorBbox.height);
                RefCluster anchorRef  = AnchorMatcher.assignAnchorToRef(anchor, anchorBboxArea, refClusters);

                // ── Expansion from anchor ─────────────────────────────────
                AnchorMatcher.MatchResult result = AnchorMatcher.expandFromAnchor(
                        anchor, anchorRef, candidates, refClusters, sceneDiag);
                List<SceneContourEntry> matched = result.matched();
                Rect regionBbox = result.regionBbox();

                // ── Deduplication ─────────────────────────────────────────
                // Collapse bright/dark achromatic pairs that trace the same physical edge
                matched = deduplicateAchromaticPairs(matched);

                // ── OPT-G: Early exit — skip scoring when type hard-gate ──
                // When we already have a strong match and this anchor's shape type
                // is incompatible with the primary ref, the max possible geometry
                // score is 0.15 → combined max ≈ 0.49, well below bestScore.
                double score = 0.0;
                boolean skipped = false;
                if (bestScore >= 0.70) {
                    VectorSignature.ShapeType anchorType = anchor.sig().type;
                    VectorSignature.ShapeType refType    = primaryRefSig.type;
                    boolean typeCompat = (anchorType == refType)
                            || (anchorType == VectorSignature.ShapeType.CLOSED_CONVEX_POLY
                                && refType == VectorSignature.ShapeType.CLOSED_CONCAVE_POLY)
                            || (anchorType == VectorSignature.ShapeType.CLOSED_CONCAVE_POLY
                                && refType == VectorSignature.ShapeType.CLOSED_CONVEX_POLY)
                            || (anchorType == VectorSignature.ShapeType.CIRCLE
                                && refType == VectorSignature.ShapeType.CLOSED_CONVEX_POLY)
                            || (anchorType == VectorSignature.ShapeType.CLOSED_CONVEX_POLY
                                && refType == VectorSignature.ShapeType.CIRCLE)
                            || anchorType == VectorSignature.ShapeType.COMPOUND
                            || refType    == VectorSignature.ShapeType.COMPOUND;
                    if (!typeCompat) {
                        skipped = true;
                    }
                }

                if (!skipped) {
                    // ── Score this candidate ──────────────────────────────────
                    double[] scoreArr = scoreRegion(refClusters, refCount, primaryRef,
                            matched, descriptor, regionBbox, allAchromatic, sceneDiag);
                    score = scoreArr[0];
                    if (score > bestScore) {
                        bestScore       = score;
                        bestBbox        = regionBbox;
                        bestAnchor      = anchor;
                        bestLayerScores = new double[]{scoreArr[1], scoreArr[2], scoreArr[3]};
                    }
                } else {
                    if (score > bestScore) {
                        bestScore  = score;
                        bestBbox   = regionBbox;
                        bestAnchor = anchor;
                    }
                }

                if (VM_DEBUG) {
                    System.out.printf("[VM-ANCHOR] Anchor %s → matched=%d score=%.1f%% region=%s%n",
                            formatBbox(anchorBbox), matched.size(), score * 100, formatBbox(regionBbox));
                }

                anchorScores.add(new double[]{score, anchor.area()});
                anchorBboxes.add(regionBbox);
                anchorEntries.add(anchor);
                anchorMatchedSets.add(matched);
            }

            // ── Anchor re-selection for bbox ────────────────────────────────
            // When a small background element (random circle, line junction) is
            // geometrically similar to the reference it can outscore the real
            // target — the placed shape at scene scale — by a significant margin,
            // causing the bbox to land on the background element.
            //
            // Two paths, gated by "prominence" of the best anchor:
            //
            //   (A) Non-prominent (bbox area < 50 % of largest candidate bbox):
            //       best anchor is a small element — re-select to the largest-bbox
            //       candidate scoring ≥ 70 % of best.
            //
            //   (B) Prominent (bbox area ≥ 50 % of largest):
            //       best anchor is already one of the largest shapes, so the bbox
            //       is likely correct.  Only allow a TIGHT same-size swap: the
            //       candidate must have bbox area ≥ 90 % of the best anchor's AND
            //       score ≥ 90 % of best, with larger contour area.  This handles
            //       inner↔outer contour swaps for outline shapes (e.g. circle
            //       outline produces two near-identical contours) without allowing
            //       jumps to smaller background elements.
            //
            // Uses bounding-box area for the prominence check (outline/thin shapes
            // have small contour area but large spatial extent).
            // Guard: shape-type compatibility (existing gate preserved).
            // This is purely geometry-driven — no colour terms involved.
            if (bestScore >= 0.60 && bestAnchor != null) {
                double bestAnchorBboxArea = (double) bestAnchor.bbox().width * bestAnchor.bbox().height;
                double maxBboxArea = bestAnchorBboxArea;
                for (SceneContourEntry ce : anchorEntries) {
                    double bb = (double) ce.bbox().width * ce.bbox().height;
                    if (bb > maxBboxArea) maxBboxArea = bb;
                }

                VectorSignature.ShapeType bestAnchorType = bestAnchor.sig() != null
                        ? bestAnchor.sig().type : null;

                if (bestAnchorBboxArea < maxBboxArea * 0.50) {
                    // ── Path A: non-prominent — re-select to largest bbox ──────
                    double reselScoreThreshold = bestScore * 0.70;
                    double reselBestBboxArea   = bestAnchorBboxArea;
                    for (int ri = 0; ri < anchorScores.size(); ri++) {
                        double rScore = anchorScores.get(ri)[0];
                        SceneContourEntry reselCandidate = anchorEntries.get(ri);
                        double rBboxArea = (double) reselCandidate.bbox().width * reselCandidate.bbox().height;
                        if (rScore >= reselScoreThreshold && rBboxArea > reselBestBboxArea) {
                            if (!isTypeCompatible(bestAnchorType, reselCandidate)) continue;
                            reselBestBboxArea = rBboxArea;
                            bestBbox    = anchorBboxes.get(ri);
                            bestAnchor  = reselCandidate;
                        }
                    }
                } else {
                    // ── Path B: prominent — tight same-size contour swap ────────
                    double reselScoreThreshold = bestScore * 0.90;
                    double minBboxArea         = bestAnchorBboxArea * 0.90;
                    double reselBestArea       = bestAnchor.area();
                    for (int ri = 0; ri < anchorScores.size(); ri++) {
                        double rScore = anchorScores.get(ri)[0];
                        double rArea  = anchorScores.get(ri)[1];
                        SceneContourEntry reselCandidate = anchorEntries.get(ri);
                        double rBboxArea = (double) reselCandidate.bbox().width * reselCandidate.bbox().height;
                        if (rScore >= reselScoreThreshold && rArea > reselBestArea
                                && rBboxArea >= minBboxArea) {
                            if (!isTypeCompatible(bestAnchorType, reselCandidate)) continue;
                            reselBestArea = rArea;
                            bestBbox    = anchorBboxes.get(ri);
                            bestAnchor  = reselCandidate;
                        }
                    }
                }
            }

            // ── Post-score bbox expansion ─────────────────────────────────
            // Only expand when the match is confident (score ≥ 75%).
            if (bestBbox != null && bestAnchor != null && bestScore >= 0.75) {
                int bestAnchorIdx = anchorEntries.indexOf(bestAnchor);
                List<SceneContourEntry> matched = anchorMatchedSets.get(bestAnchorIdx);
                bestBbox = BboxExpander.expandBbox(bestBbox, bestAnchor, matched,
                        candidates, refClusters, referenceId.name(), descriptor.sceneArea);
            }
            // else: low-confidence match → keep conservative scored bbox

            double scorePercent = bestScore * 100.0;
            long   elapsed      = System.currentTimeMillis() - t0;

            Path savedPath = null;
            if (saveVariants.contains(variant.variantName())) {
                savedPath = writeAnnotated(scene.sceneMat(), bestBbox, variant.variantName(),
                        scorePercent, referenceId, scene, outputDir);
            }

            return new AnalysisResult(variant.variantName(), referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    scorePercent, bestBbox, elapsed, 0L,
                    (int) descriptor.sceneArea, savedPath, false, null,
                    new AnalysisResult.ScoringLayers(
                            bestLayerScores[0] * 100.0,
                            bestLayerScores[1] * 100.0,
                            bestLayerScores[2] * 100.0));

        } catch (Exception e) {
            return AnalysisResult.error(variant.variantName(), referenceId,
                    scene.variantLabel(), scene.category(), scene.backgroundId(),
                    System.currentTimeMillis() - t0,
                    (int) descriptor.sceneArea,
                    e.getMessage());
        }
    }

    // =========================================================================
    // Scoring
    // =========================================================================

    /**
     * Three-layer score for a candidate matched set.
     *
     * @param refClusters  all reference clusters
     * @param refCount     number of ref clusters (pre-dedup ref count)
     * @param primaryRef   ref cluster with largest contour area
     * @param matched      deduplicated scene entries selected for this candidate
     * @param descriptor   scene descriptor (for Fix B contamination)
     * @param regionBbox   candidate bounding box
     * @param allAchromatic true when every ref cluster is achromatic (pre-computed)
     * @param sceneDiag     scene diagonal length (pre-computed, OPT-I)
     */
    /**
     * Returns {@code double[]{combined, countScore, matchScore, geomScore}} — all in [0,1].
     * Index 0 is the weighted combined score; indices 1-3 are the raw per-layer scores
     * (before weighting) for Layer 1 (Boundary Count), Layer 2 (Structural), Layer 3 (Geometry).
     */
    private static double[] scoreRegion(List<RefCluster> refClusters,
                                        int refCount,
                                        RefCluster primaryRef,
                                        List<SceneContourEntry> matched,
                                        SceneDescriptor descriptor,
                                        Rect regionBbox,
                                        boolean allAchromatic,
                                        double sceneDiag) {

        // ── Layer 1: boundary count ───────────────────────────────────────
        int matchedCount = matched.size();
        double countScore;
        if (refCount == 0) {
            countScore = 1.0;
        } else {
            int diff = matchedCount - refCount;
            if (diff > 0) {
                countScore = Math.exp(-CLUSTER_PENALTY_K * diff);              // extra — over-segmentation
            } else {
                countScore = Math.exp(-CLUSTER_PENALTY_K_MISS * Math.abs(diff));
                countScore = Math.max(countScore, MIN_COUNT_SCORE_MISS);
            }
        }

        // ── Fix B: chromatic contamination (achromatic refs only) ─────────
        // Achromatic references (white/grey on black) should not score highly on
        // regions dominated by chromatic content.  The check measures the fraction
        // of chromatic pixels inside the candidate bbox.
        //
        // Floor (0.70): random-line backgrounds produce 40–80+ % chromatic coverage
        // inside a bbox even when the detected shape itself is purely achromatic.
        // Only extreme contamination (> 70 %) is penalised.
        //
        // Cap (0.30): the contamination-adjusted countScore is floored at 0.30
        // so background noise never destroys Layer 1.  Combined with the
        // topology-validated seg rescue in VectorSignature (which boosts
        // Layer 3 geometry to ~0.92 for genuine matches), a 0.30 floor gives
        // combined scores ≥ 85 % for correct self-matches.  A higher floor
        // (e.g. 0.60) can let background candidates outscore the real shape
        // for thin/ambiguous shapes like ARC_HALF.
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

        // ── Layer 2: structural coherence ────────────────────────────────
        double sumContrib  = 0.0;
        double sumWeights  = 0.0;

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
            Rect   refBb        = primaryBbox(rc);
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

            sumContrib += weight * (proxScore * 0.40 + covScore * 0.40 + scaleScore * 0.20);
        }

        double matchScore = (sumWeights > 0) ? sumContrib / sumWeights : 0.0;

        // ── Layer 3: primary boundary geometry ───────────────────────────
        // Try ALL contour signatures from every ref cluster against every matched
        // scene entry.  Using only bestSig (max solidity) per cluster can miss the
        // correct pairing in compound shapes — e.g. a circle-outline cluster's
        // bestSig might be a co-located triangle (higher solidity), causing the
        // circle-ref to compare against the triangle scene contour instead of the
        // circle scene contour.
        //
        // Using the best geometry across ALL ref contours × ALL scene entries is
        // purely structural — it finds the best structural alignment regardless of
        // cluster origin or solidity ranking.
        double geomScore = 0.0;
        SceneContourEntry primaryScene = null;
        double bestGeom = -1.0;
        for (RefCluster rc : refClusters) {
            for (MatOfPoint refContour : rc.contours) {
                VectorSignature rcSig = VectorSignature.buildFromContour(
                        refContour, EPSILON, Double.NaN);
                for (SceneContourEntry e : matched) {
                    double sim = rcSig.similarity(e.sig());
                    if (sim > bestGeom) {
                        bestGeom = sim;
                        primaryScene = e;
                    }
                }
            }
        }
        if (bestGeom >= 0.0) {
            geomScore = bestGeom;
        }

        // If geometry and structural coherence are both near-perfect, do not let
        // count-only over-segmentation suppress an otherwise clear true match.
        if (geomScore >= 0.95 && matchScore >= 0.95) {
            countScore = Math.max(countScore, 0.45);
        }

        double combined = countScore * W_COUNT
                        + matchScore  * W_MATCH
                        + geomScore   * W_GEOM;

        // DEBUG — remove after diagnosis
        if (combined > 0.4 && VM_DEBUG) {
            VectorSignature rs = primaryRef != null ? primaryRef.bestSig(EPSILON) : null;
            SceneContourEntry ps = primaryScene != null ? primaryScene : null;
            System.out.printf("[VM-DEBUG] count=%.3f match=%.3f geom=%.3f combined=%.3f | refType=%s refCirc=%.3f refV=%d | sceneType=%s sceneCirc=%.3f sceneV=%d%n",
                countScore, matchScore, geomScore, combined,
                rs != null ? rs.type : "?", rs != null ? rs.circularity : 0, rs != null ? rs.vertexCount : 0,
                ps != null ? ps.sig().type : "?", ps != null ? ps.sig().circularity : 0, ps != null ? ps.sig().vertexCount : 0);
        }

        return new double[]{
                Math.max(0.0, Math.min(1.0, combined)),
                countScore,
                matchScore,
                geomScore
        };
    }

    // =========================================================================
    // Deduplication
    // =========================================================================

    /**
     * Collapses bright/dark achromatic pairs that trace the same physical boundary.
     *
     * <p>Only fires when:
     * <ul>
     *   <li>One entry is bright achromatic and the other is dark achromatic</li>
     *   <li>Their bbox IoU exceeds {@value #DEDUP_IOU_MIN}</li>
     *   <li>Their contourArea ratio (min/max) exceeds {@value #DEDUP_AREA_RATIO_MIN}
     *       — confirming same boundary, not an inner/outer ring pair</li>
     * </ul>
     * Chromatic entries are never deduplicated against each other, but a chromatic
     * entry and a dark achromatic entry tracing the same physical boundary are
     * collapsed (Pass 2) using the same IoU + area-ratio conditions.
     */
    private static List<SceneContourEntry> deduplicateAchromaticPairs(
            List<SceneContourEntry> entries) {

        // ── Pass 1: bright/dark achromatic pair ───────────────────────────
        SceneContourEntry brightEntry = null;
        SceneContourEntry darkEntry   = null;
        for (SceneContourEntry e : entries) {
            if (e.achromatic()) {
                if (e.brightAchromatic()) brightEntry = e;
                else                    darkEntry   = e;
            }
        }

        List<SceneContourEntry> result = new ArrayList<>(entries);

        if (brightEntry != null && darkEntry != null) {
            Rect   bb     = brightEntry.bbox();
            Rect   db     = darkEntry.bbox();
            double iou    = GeometryUtils.bboxIoU(bb, db);
            double areaB  = brightEntry.area();
            double areaD  = darkEntry.area();
            double ratio  = (areaB > 0 && areaD > 0)
                    ? Math.min(areaB, areaD) / Math.max(areaB, areaD) : 0.0;

            if (iou >= DEDUP_IOU_MIN && ratio >= DEDUP_AREA_RATIO_MIN) {
                SceneContourEntry discard = (areaB >= areaD) ? darkEntry : brightEntry;
                result.remove(discard);
                if (discard == darkEntry) darkEntry = null;
                else brightEntry = null;
            }
        }

        // ── Pass 2: chromatic/dark achromatic pair ────────────────────────
        // A coloured scene shape produces a chromatic cluster (colour side) and a
        // dark achromatic cluster (background side) at the same boundary.  Collapse
        // to the chromatic entry so matchedCount stays in sync with a ref that has
        // already been deduped the same way.
        if (darkEntry != null) {
            Rect   db    = darkEntry.bbox();
            double areaD = darkEntry.area();
            boolean shouldRemoveDark = false;
            for (SceneContourEntry e : result) {
                if (e.achromatic()) continue;
                Rect   cb    = e.bbox();
                double areaC = e.area();
                double iou   = GeometryUtils.bboxIoU(cb, db);
                double ratio = (areaC > 0 && areaD > 0)
                        ? Math.min(areaC, areaD) / Math.max(areaC, areaD) : 0.0;
                if (iou >= DEDUP_IOU_MIN && ratio >= DEDUP_AREA_RATIO_MIN) {
                    shouldRemoveDark = true;
                    break;
                }
            }
            if (shouldRemoveDark) result.remove(darkEntry);
        }

        return result;
    }

    // =========================================================================
    // Reference cluster builder
    // =========================================================================

    static List<RefCluster> buildRefClusters(Mat refBgr) {
        // Use ExperimentalSceneColourClusters — the same extraction path as
        // SceneDescriptor.build() — so ref and scene cluster decomposition
        // are directly comparable.  The non-experimental extractor produces
        // different cluster assignments (e.g. misses cyan/magenta halves on
        // BICOLOUR_RECT_HALVES) which causes systematic ref/scene mismatch.
        List<ColourCluster> raw =
                ExperimentalSceneColourClusters.INSTANCE.extractFromBorderPixels(refBgr);
        List<RefCluster> result = new ArrayList<>();
        double refArea = (double) refBgr.rows() * refBgr.cols();

        for (ColourCluster c : raw) {
            List<MatOfPoint> contours = SceneDescriptor.contoursFromMask(c.mask);
            if (!contours.isEmpty()) {
                double solidity = computeSolidity(contours);
                result.add(new RefCluster(c.hue, c.achromatic, c.brightAchromatic,
                        contours, refArea, solidity));
            }
            c.release();
        }

        // Fallback: no border-pixel clusters found — use full extraction
        if (result.isEmpty()) {
            List<ColourCluster> fallback =
                    ExperimentalSceneColourClusters.INSTANCE.extract(refBgr);
            for (ColourCluster c : fallback) {
                List<MatOfPoint> contours = SceneDescriptor.contoursFromMask(c.mask);
                if (!contours.isEmpty()) {
                    double solidity = computeSolidity(contours);
                    result.add(new RefCluster(c.hue, c.achromatic, c.brightAchromatic,
                            contours, refArea, solidity));
                }
                c.release();
            }
        }

        // Apply the same bright/dark achromatic deduplication used on the scene
        // matched set.  A simple filled shape (e.g. CIRCLE_FILLED, RECT_FILLED) has
        // two clusters — bright border and dark border — both tracing the SAME physical
        // edge from opposite sides.  Without dedup the refCount=2 but the scene
        // matched set (correctly deduped) has count=1, causing a spurious Layer 1
        // penalty of exp(-1.5)≈0.22 on every simple shape.
        result = deduplicateRefClusters(result);

        return result;
    }

    /** Computes solidity of the largest contour in a list. */
    private static double computeSolidity(List<MatOfPoint> contours) {
        if (contours.isEmpty()) return 0.0;
        
        // Find largest contour
        MatOfPoint largest = contours.get(0);
        double maxArea = Imgproc.contourArea(largest);
        for (int i = 1; i < contours.size(); i++) {
            double area = Imgproc.contourArea(contours.get(i));
            if (area > maxArea) {
                maxArea = area;
                largest = contours.get(i);
            }
        }
        
        // Compute solidity = contourArea / convexHullArea
        MatOfInt hull = new MatOfInt();
        Imgproc.convexHull(largest, hull);
        
        // Convert hull indices to points
        Point[] contourPoints = largest.toArray();
        int[] hullIndices = hull.toArray();
        Point[] hullPoints = new Point[hullIndices.length];
        for (int i = 0; i < hullIndices.length; i++) {
            hullPoints[i] = contourPoints[hullIndices[i]];
        }
        
        MatOfPoint hullMat = new MatOfPoint(hullPoints);
        double hullArea = Imgproc.contourArea(hullMat);
        hull.release();
        hullMat.release();
        
        return (hullArea > 0) ? maxArea / hullArea : 0.0;
    }

    /**
     * Collapses ref clusters that trace the same physical boundary from opposite sides.
     *
     * <p><b>Pass 1 — bright/dark achromatic pair:</b> a white/grey shape on a dark
     * background produces a bright achromatic cluster (bright side) and a dark
     * achromatic cluster (dark side) at the same location.
     *
     * <p><b>Pass 2 — chromatic/dark achromatic pair:</b> a coloured shape (e.g. red
     * {@code CIRCLE_FILLED}) on a dark background produces a chromatic cluster (colour
     * side) and a dark achromatic cluster (background side) at the same location.
     * Without this pass, {@code refCount=2} while a white-on-black scene candidate is
     * correctly deduped to {@code matchedCount=1}, causing a spurious
     * {@code exp(-1.5)=0.22} Layer 1 penalty on every simple coloured shape.
     *
     * <p>Both passes use the same safety conditions (bbox IoU > 0.5 AND contourArea
     * ratio > 0.90) so that genuine two-boundary shapes (e.g. {@code HEXAGON_OUTLINE},
     * whose inner void has area ratio ≈ 0.79 &lt; 0.90) are never collapsed.
     */
    private static List<RefCluster> deduplicateRefClusters(List<RefCluster> clusters) {
        RefCluster brightRef = null, darkRef = null;
        for (RefCluster rc : clusters) {
            if (rc.achromatic) {
                if (rc.brightAchromatic) brightRef = rc;
                else                     darkRef   = rc;
            }
        }

        List<RefCluster> result = new ArrayList<>(clusters);

        // ── Pass 1: bright/dark achromatic pair ───────────────────────────
        if (brightRef != null && darkRef != null) {
            Rect   bb    = primaryBbox(brightRef);
            Rect   db    = primaryBbox(darkRef);
            double iou   = GeometryUtils.bboxIoU(bb, db);
            double areaB = brightRef.maxContourArea;
            double areaD = darkRef.maxContourArea;
            double ratio = (areaB > 0 && areaD > 0)
                    ? Math.min(areaB, areaD) / Math.max(areaB, areaD) : 0.0;

            if (iou >= DEDUP_IOU_MIN && ratio >= DEDUP_AREA_RATIO_MIN) {
                RefCluster discard = (areaB >= areaD) ? darkRef : brightRef;
                result.remove(discard);
                discard.release();
                // Track whether darkRef was removed so Pass 2 knows
                if (discard == darkRef) darkRef = null;
                else brightRef = null;
            }
        }

        // ── Pass 2: chromatic/dark achromatic pair ────────────────────────
        // Only runs when a dark achromatic cluster survived Pass 1.
        if (darkRef != null) {
            Rect   db    = primaryBbox(darkRef);
            double areaD = darkRef.maxContourArea;
            boolean shouldRemoveDark = false;
            for (RefCluster rc : result) {
                if (rc.achromatic) continue;   // only compare against chromatic
                Rect   cb    = primaryBbox(rc);
                double areaC = rc.maxContourArea;
                double iou   = GeometryUtils.bboxIoU(cb, db);
                double ratio = (areaC > 0 && areaD > 0)
                        ? Math.min(areaC, areaD) / Math.max(areaC, areaD) : 0.0;
                if (iou >= DEDUP_IOU_MIN && ratio >= DEDUP_AREA_RATIO_MIN) {
                    shouldRemoveDark = true;
                    break;
                }
            }
            if (shouldRemoveDark) {
                result.remove(darkRef);
                darkRef.release();
            }
        }

        return result;
    }

    /** Returns the bounding rect of the primary (largest-area) contour in a ref cluster. */
    private static Rect primaryBbox(RefCluster rc) {
        return rc.primaryBbox;
    }


    // =========================================================================
    // Scene candidate collection
    // =========================================================================

    private static List<SceneContourEntry> collectSceneCandidates(SceneDescriptor descriptor) {
        List<SceneContourEntry> out = new ArrayList<>();
        List<SceneDescriptor.ClusterContours> clusters = descriptor.clusters();

        for (int ci = 0; ci < clusters.size(); ci++) {
            SceneDescriptor.ClusterContours cc = clusters.get(ci);
            if (cc.envelope) continue;

            for (MatOfPoint c : cc.contours) {
                // Skip contours that span > 80% of scene — background fills or noise merges.
                // The previous 50% threshold was too aggressive: shapes rotated 45° (e.g.
                // RECT_SQUARE, LINE_X) can have AABB area of 55–75% of scene area even
                // though their contour area is much smaller.  VectorSignature.similarity()
                // already caps near-full-image contours (normalisedArea > 0.80) at 0.25,
                // providing a secondary guard against background fills.
                Rect   bb     = Imgproc.boundingRect(c);
                double bbArea = (double) bb.width * bb.height;
                if (bbArea > descriptor.sceneArea * 0.80) continue;

                double cArea = Imgproc.contourArea(c);
                // OPT-R: defer VectorSignature build until after filtering stages
                out.add(new SceneContourEntry(c, ci, cc.achromatic, cc.brightAchromatic,
                        cc.hue, null, bb, cArea));
            }
        }
        return out;
    }

    /**
     * OPT-R: Builds VectorSignatures for entries that don't have one yet.
     * Called after CC filter + global size filter so that eliminated contours
     * skip the expensive buildFromContour work entirely.
     */
    private static List<SceneContourEntry> buildSignatures(
            List<SceneContourEntry> entries, double sceneArea) {
        List<SceneContourEntry> result = new ArrayList<>(entries.size());
        for (SceneContourEntry ce : entries) {
            if (ce.sig() != null) { result.add(ce); continue; }
            VectorSignature sig = VectorSignature.buildFromContour(ce.contour(), EPSILON, sceneArea);
            result.add(new SceneContourEntry(ce.contour(), ce.clusterIdx(), ce.achromatic(),
                    ce.brightAchromatic(), ce.clusterHue(), sig, ce.bbox(), ce.area()));
        }
        return result;
    }

    // =========================================================================
    // Selection helpers
    // =========================================================================


    /**
     * Returns the primary ref cluster — the one with the largest contour area.
     * This is the foreground shape boundary regardless of colour.
     */
    private static RefCluster findPrimaryCluster(List<RefCluster> refClusters) {
        RefCluster primary = refClusters.get(0);
        for (RefCluster rc : refClusters) {
            if (rc.maxContourArea > primary.maxContourArea) primary = rc;
        }
        return primary;
    }

    /**
     * Finds the matched entry that best represents the given ref cluster,
     * using bbox-normalised fill ratios for both sides so scale differences
     * between ref (128×128) and scene (640×480) do not affect assignment.
     */
    private static SceneContourEntry findMatchedEntryForRef(RefCluster rc,
                                                             List<SceneContourEntry> matched) {
        Rect   refBb     = primaryBbox(rc);
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

    // =========================================================================
    // Geometry / bbox helpers
    // =========================================================================

    /** Returns true if the shape type is any closed polygon variant. */
    private static boolean isClosedPoly(VectorSignature.ShapeType t) {
        return t == VectorSignature.ShapeType.CLOSED_CONVEX_POLY
            || t == VectorSignature.ShapeType.CLOSED_CONCAVE_POLY;
    }

    /**
     * Shape-type compatibility gate for anchor re-selection.
     * Returns {@code false} (skip candidate) when the candidate's shape type is
     * incompatible with the original best anchor's type.
     */
    private static boolean isTypeCompatible(VectorSignature.ShapeType bestAnchorType,
                                            SceneContourEntry candidate) {
        VectorSignature.ShapeType candType = candidate.sig() != null
                ? candidate.sig().type : null;
        if (bestAnchorType == null || candType == null) return true;
        if (bestAnchorType == candType) return true;
        if (candType == VectorSignature.ShapeType.COMPOUND
                || bestAnchorType == VectorSignature.ShapeType.COMPOUND) return true;
        boolean sameFamily =
                (isClosedPoly(bestAnchorType) && isClosedPoly(candType))
             || (bestAnchorType == VectorSignature.ShapeType.CIRCLE && isClosedPoly(candType))
             || (isClosedPoly(bestAnchorType) && candType == VectorSignature.ShapeType.CIRCLE);
        return sameFamily;
    }


    private static Rect clampRect(Rect r, Mat img) {
        int x  = Math.max(0, r.x);
        int y  = Math.max(0, r.y);
        int x2 = Math.min(img.cols(), r.x + r.width);
        int y2 = Math.min(img.rows(), r.y + r.height);
        return new Rect(x, y, Math.max(0, x2 - x), Math.max(0, y2 - y));
    }

    // =========================================================================
    // Utility
    // =========================================================================

    private static AnalysisResult zeroResult(VectorVariant variant, ReferenceId referenceId,
                                              SceneEntry scene, SceneDescriptor descriptor,
                                              long t0) {
        return new AnalysisResult(variant.variantName(), referenceId,
                scene.variantLabel(), scene.category(), scene.backgroundId(),
                0.0, null, System.currentTimeMillis() - t0, 0L,
                (int) descriptor.sceneArea, null, false, null,
                AnalysisResult.ScoringLayers.ZERO);
    }


    // =========================================================================
    // Annotation writer
    // =========================================================================

    static Path writeAnnotated(Mat scene, Rect bbox, String variant, double score,
                                ReferenceId refId, SceneEntry sceneEntry, Path outputDir) {
        try {
            Path dir = outputDir.resolve("annotated").resolve(sanitise(variant));
            Files.createDirectories(dir);
            String sceneRef = sceneEntry.primaryReferenceId() != null
                    ? sanitise(sceneEntry.primaryReferenceId().name()) : "neg";
            String fname = sanitise(refId.name()) + "_vs_" + sceneRef
                    + "_" + sanitise(sceneEntry.variantLabel()) + ".png";
            Mat m = scene.clone();
            Scalar colour = score >= 70 ? new Scalar(0, 200, 0)
                          : score >= 40 ? new Scalar(0, 200, 200)
                          :               new Scalar(0, 0, 200);
            if (bbox != null && bbox.width > 1 && bbox.height > 1) {
                Imgproc.rectangle(m,
                        new Point(bbox.x, bbox.y),
                        new Point(bbox.x + bbox.width, bbox.y + bbox.height),
                        colour, 1);
            }
            Imgproc.putText(m, shortName(variant),
                    new Point(4, 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.22,
                    new Scalar(200, 200, 200), 1);
            Imgproc.putText(m, String.format("%.1f%%", score),
                    new Point(4, 20), Imgproc.FONT_HERSHEY_SIMPLEX, 0.28, colour, 1);
            // Upscale 4× with nearest-neighbour so contours and boxes stay crisp
            Mat big = new Mat();
            Imgproc.resize(m, big,
                    new Size(m.cols() * 4, m.rows() * 4), 0, 0, Imgproc.INTER_NEAREST);
            m.release();
            Path dest = dir.resolve(fname);
            Imgcodecs.imwrite(dest.toString(), big);
            big.release();
            return outputDir.toAbsolutePath().relativize(dest.toAbsolutePath());
        } catch (Exception e) {
            return null;
        }
    }

    // =========================================================================
    // Public utility stubs — retained for API compatibility with callers
    // =========================================================================

    public static List<MatOfPoint> extractContoursFromBinary(Mat maskedBgr) {
        Mat grey = new Mat(), bin = new Mat();
        Imgproc.cvtColor(maskedBgr, grey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(grey, bin, 20, 255, Imgproc.THRESH_BINARY);
        grey.release();
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(bin, contours, hierarchy,
                Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        hierarchy.release(); bin.release();
        contours.removeIf(c -> Imgproc.contourArea(c) < SceneColourClusters.MIN_CONTOUR_AREA);
        return contours;
    }

    public static Mat extractBinaryRaw(Mat bgrScene) {
        Mat grey = new Mat(), bin = new Mat(), edge = new Mat();
        Imgproc.cvtColor(bgrScene, grey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(grey, bin, 20, 255, Imgproc.THRESH_BINARY);
        Imgproc.Canny(grey, edge, 40.0, 120.0);
        Core.bitwise_or(bin, edge, bin);
        grey.release(); edge.release();
        return bin;
    }

    public static Mat drawContourGraph(Size size, Mat binary,
                                       List<MatOfPoint> contours, double epsilon) {
        Mat out = Mat.zeros((int) size.height, (int) size.width, CvType.CV_8UC3);
        if (binary != null && !binary.empty()) {
            Mat grey3 = new Mat();
            Imgproc.cvtColor(binary, grey3, Imgproc.COLOR_GRAY2BGR);
            Core.multiply(grey3, new Scalar(0.4, 0.4, 0.4), grey3);
            Core.add(out, grey3, out);
            grey3.release();
        }
        int[][] palette = {
            {80,80,255},{80,255,80},{255,80,80},{80,255,255},
            {255,80,255},{255,255,80},{80,160,255},{255,80,160}
        };
        for (int ci = 0; ci < contours.size(); ci++) {
            int[] col = palette[ci % palette.length];
            Scalar edgeCol = new Scalar(col[0], col[1], col[2]);
            MatOfPoint c = contours.get(ci);
            Point[] pts;
            if (epsilon <= 0) {
                pts = c.toArray();
            } else {
                double perim = Imgproc.arcLength(new MatOfPoint2f(c.toArray()), true);
                double eps   = Math.max(epsilon * perim, 2.0);
                MatOfPoint2f approx = new MatOfPoint2f();
                Imgproc.approxPolyDP(new MatOfPoint2f(c.toArray()), approx, eps, true);
                pts = approx.toArray();
                approx.release();
            }
            int n = pts.length;
            if (n == 0) continue;
            for (int i = 0; i < n; i++)
                Imgproc.line(out, pts[i], pts[(i + 1) % n], edgeCol, 1);
            for (Point p : pts)
                Imgproc.circle(out, p, epsilon > 0 ? 2 : 1, new Scalar(255, 255, 255), -1);
        }
        return out;
    }

    public static List<VectorSignature> buildRefSignatures(Mat refBgr, double epsilonFactor) {
        List<RefCluster> clusters = buildRefClusters(refBgr);
        List<VectorSignature> sigs = new ArrayList<>();
        for (RefCluster rc : clusters) sigs.add(rc.bestSig(epsilonFactor));
        for (RefCluster rc : clusters) rc.release();
        return sigs.isEmpty() ? List.of(buildRefSignature(refBgr, epsilonFactor)) : sigs;
    }

    public static VectorSignature buildRefSignature(Mat refBgr, double epsilonFactor) {
        Mat grey = new Mat(), bin = new Mat();
        Imgproc.cvtColor(refBgr, grey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(grey, bin, 20, 255, Imgproc.THRESH_BINARY);
        grey.release();
        VectorSignature sig = VectorSignature.build(bin, epsilonFactor, Double.NaN);
        bin.release();
        return sig;
    }


    private static String sanitise(String s) { return s.replaceAll("[^A-Za-z0-9_\\-]", "_"); }

    private static String shortName(String v) {
        return v.replace("VECTOR_STRICT", "VM_S")
                .replace("VECTOR_NORMAL", "VM_N")
                .replace("VECTOR_LOOSE",  "VM_L");
    }

    private static String formatBbox(Rect r) {
        return String.format("[%d,%d,%dx%d]", r.x, r.y, r.width, r.height);
    }
}
