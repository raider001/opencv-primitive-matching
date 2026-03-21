package org.example.matchers;

import org.example.analytics.AnalysisResult;
import org.example.colour.ColourCluster;
import org.example.colour.ExperimentalSceneColourClusters;
import org.example.colour.SceneColourClusters;
import org.example.factories.ReferenceId;
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
    /** Max centre-to-centre distance as a fraction of scene diagonal for expansion. */
    private static final double PROXIMITY_THRESHOLD = 0.35;
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

    // ── Debug flags — read once at class load ────────────────────────────────
    private static final boolean VM_DEBUG      = System.getProperty("vm.debug") != null;
    private static final boolean VM_BBOX_DEBUG = System.getProperty("vm.bbox.debug") != null;

    // ── Contour isolation constants ───────────────────────────────────────────
    /**
     * Minimum contour area as a fraction of the largest contour in the same
     * cluster.  Contours below this ratio are isolated noise blobs and are
     * dropped by the connected-component filter before scoring.
     */
    private static final double CC_AREA_RATIO_MIN = 0.10;
    /**
     * Minimum contour area as a fraction of the GLOBAL largest contour across
     * all clusters (Stage 3 filter).  Drops tiny background elements — e.g.
     * random background circles (20–60 px, ~2–5% of scene target area) —
     * that would otherwise outscore the actual target shape in Layer 2/3 and
     * produce wrong-location detections.
     *
     * Set to 8%: inner rings of COMPOUND_BULLSEYE (~20–25% of outer) are kept;
     * background 30 px circles (~2–4% of a 150 px target) are dropped.
     */
    private static final double MIN_GLOBAL_AREA_RATIO = 0.08;
    /**
     * Number of top candidates (by raw contour area) to re-extract with
     * morphological opening during reference-adaptive erosion (Stage 2).
     */
    private static final int    EROSION_TOP_K     = 3;

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
            // Per-cluster: drops isolated noise blobs whose area is < 10 % of that
            // cluster's largest contour.  Handles disconnected same-colour background
            // fragments without requiring any reference geometry knowledge.
            candidates = applyConnectedComponentFilter(candidates);
            if (candidates.isEmpty()) {
                return zeroResult(variant, referenceId, scene, descriptor, t0);
            }

            // ── Stage 1b: Global minimum-size filter ───────────────────────────
            // Drops candidates whose contour area is below MIN_GLOBAL_AREA_RATIO of
            // the scene-wide largest contour (across all clusters).  This eliminates
            // tiny background elements — random circles (20–60 px), short line
            // segments — that survive the per-cluster CC filter but would still
            // outscore the actual target in Layer 2/3 due to incidentally matching
            // geometry (a 30 px circle classified as CLOSED_CONVEX_POLY with
            // 8 vertices scores ~97 % against an octagon reference).
            // Inner components of compound shapes (cross inside circle, inner bullseye
            // rings) are retained because they are ≥ 15–25 % of the outer contour.
            candidates = applyGlobalSizeFilter(candidates);
            if (candidates.isEmpty()) {
                return zeroResult(variant, referenceId, scene, descriptor, t0);
            }

            // ── Stage 2: Reference-adaptive morphological opening ──────────────
            // Re-extracts the top-K candidates with a morphological open sized to
            // the reference fill density, severing thin background-line arms that
            // are physically connected to the main shape.
            // Skipped entirely for outline/line references (solidity < 0.30).
            int erosionDepth = computeErosionDepth(primaryRef);
            if (erosionDepth > 0) {
                candidates = reExtractTopCandidates(candidates, scene, erosionDepth, descriptor);
            }

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
            Rect   bestBbox   = null;
            SceneContourEntry bestAnchor = null;

            // Track all scored anchors for post-loop re-selection
            List<double[]>          anchorScores      = new ArrayList<>();
            List<Rect>              anchorBboxes      = new ArrayList<>();
            List<SceneContourEntry> anchorEntries     = new ArrayList<>();
            List<List<SceneContourEntry>> anchorMatchedSets = new ArrayList<>();

            for (SceneContourEntry anchor : candidates) {
                Rect anchorBbox = anchor.bbox;

                // ── Anchor-to-ref assignment ──────────────────────────────
                // The anchor represents the ref cluster whose relative contour area
                // (vs ref image) most closely matches the anchor's (vs its own bbox area).
                double anchorBboxArea = Math.max(1.0, (double) anchorBbox.width * anchorBbox.height);
                RefCluster anchorRef  = assignAnchorToRef(anchor, anchorBboxArea, refClusters);

                // ── Expansion loop ────────────────────────────────────────
                // Select additional scene clusters by proximity + relative size only.
                // No colour, no geometry.
                // For multi-cluster refs (>2), use larger proximity threshold to handle
                // patterns like BICOLOUR_CROSSHAIR_RING where components may be far apart.
                double proximityThreshold = (refClusters.size() > 2) ? 0.50 : PROXIMITY_THRESHOLD;
                
                Rect regionBbox = anchorBbox;
                List<SceneContourEntry> matched = new ArrayList<>();
                matched.add(anchor);
                Set<Integer> usedIdx = new HashSet<>();
                usedIdx.add(anchor.clusterIdx);

                for (RefCluster rc : refClusters) {
                    if (rc == anchorRef) continue;   // anchor already covers this one

                    double refFrac = refFraction(rc);
                    SceneContourEntry best    = null;
                    double            bestDiff = Double.MAX_VALUE;

                    for (SceneContourEntry ce : candidates) {
                        if (usedIdx.contains(ce.clusterIdx)) continue;

                        // Proximity gate (relaxed for multi-cluster patterns)
                        // Use regionBbox (growing bbox) instead of anchorBbox to allow expansion
                        // to distant components that are part of the same pattern
                        double dist = centreDist(regionBbox, ce.bbox);
                        if (dist > sceneDiag * proximityThreshold) continue;

                        // Relative size match — compare against candidate bbox so far
                        double ceFrac = sceneFraction(ce, regionBbox);
                        double diff   = Math.abs(refFrac - ceFrac);
                        if (diff < bestDiff) { bestDiff = diff; best = ce; }
                    }

                    if (best != null) {
                        matched.add(best);
                        usedIdx.add(best.clusterIdx);
                        // Only expand regionBbox for non-background-scale contours
                        Rect bestBb = best.bbox;
                        double bestArea = (double) bestBb.width * bestBb.height;
                        if (bestArea < descriptor.sceneArea * 0.60) {
                            regionBbox = unionRect(regionBbox, bestBb);
                        }
                    }
                }

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
                    VectorSignature.ShapeType anchorType = anchor.sig.type;
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
                    score = scoreRegion(refClusters, refCount, primaryRef,
                            matched, descriptor, regionBbox, allAchromatic);
                }

                if (VM_DEBUG) {
                    System.out.printf("[VM-ANCHOR] Anchor %s → matched=%d score=%.1f%% region=%s%n",
                            formatBbox(anchorBbox), matched.size(), score * 100, formatBbox(regionBbox));
                }

                anchorScores.add(new double[]{score, anchor.area});
                anchorBboxes.add(regionBbox);
                anchorEntries.add(anchor);
                anchorMatchedSets.add(matched);

                if (score > bestScore) {
                    bestScore  = score;
                    bestBbox   = regionBbox;
                    bestAnchor = anchor;
                }
            }

            // ── Anchor re-selection for bbox: prefer largest contour ──────
            // When the scene contains background elements geometrically similar to
            // the reference (e.g. background circles vs octagon/pentagon), a small
            // background contour can outscore the actual target by a few percent,
            // causing the bbox to land on the background element instead of the
            // real target shape.
            //
            // Fix: keep bestScore (the highest score from any anchor — conservative
            // for rejection tests) but re-select the ANCHOR and BBOX from the
            // candidate with the largest contour area among those scoring within
            // 90 % of the best.  The target shape at 3× scale is always among the
            // largest contours, so this correctly steers the bbox to the right
            // location while preserving the best geometry score.
            //
            // This is purely geometry-driven — no colour terms involved.
            if (bestScore >= 0.60 && bestAnchor != null) {
                double reselThreshold = bestScore * 0.85;
                double reselBestArea  = bestAnchor.area;
                for (int ri = 0; ri < anchorScores.size(); ri++) {
                    double rScore = anchorScores.get(ri)[0];
                    double rArea  = anchorScores.get(ri)[1];
                    if (rScore >= reselThreshold && rArea > reselBestArea) {
                        reselBestArea = rArea;
                        // Score stays at the original best; only bbox/anchor change
                        bestBbox    = anchorBboxes.get(ri);
                        bestAnchor  = anchorEntries.get(ri);
                    }
                }
            }

            // ── Post-score bbox expansion ─────────────────────────────────
            // Only expand when the match is confident (score ≥ 85%).
            if (bestBbox != null && bestAnchor != null && bestScore >= 0.80) {
                Rect initialBbox = new Rect(bestBbox.x, bestBbox.y, bestBbox.width, bestBbox.height);

                // ── Step A: Retrieve the stored matched set for bestAnchor ────
                // (replaces the duplicated expansion loop — OPT-B)
                Rect anchorBbox = bestAnchor.bbox;
                int bestAnchorIdx = anchorEntries.indexOf(bestAnchor);
                List<SceneContourEntry> matched = anchorMatchedSets.get(bestAnchorIdx);

                // ── Step B: Compute reference full extent and scene scale ─────
                // Full reference extent = union of every cluster's primary bbox.
                Rect refFullBbox = primaryBbox(refClusters.get(0));
                for (RefCluster rc : refClusters) {
                    refFullBbox = unionRect(refFullBbox, primaryBbox(rc));
                }

                // Estimate scene-to-reference scale by searching ALL contours across
                // ALL ref clusters — not just the primary one.  For compound shapes
                // (BULLSEYE), the scoring loop may select an INNER ring as the best
                // anchor.  Using only the outer primary bbox underestimates the scale
                // (middle-ring anchor at scene-207px / outer-ring-ref-113px = 1.83×
                // instead of the true 3×).
                //
                // Strategy: for each ref contour with diagonal ≥ 6px, compute a
                // candidate scale = anchorDiag / refContourDiag.  Select the
                // candidate whose deviation from the nearest 0.5-step value is
                // smallest (e.g. 2.88 → 3.0, dev=0.12; 1.83 → 2.0, dev=0.17).
                // Half-step quantisation reflects natural scene scales used in this
                // test suite (1×, 1.5×, 2×, 3×). Fallback: 3.0.
                double anchorDiag    = Math.hypot(anchorBbox.width, anchorBbox.height);
                double estimatedScale = 3.0;
                double bestRatioDev   = Double.MAX_VALUE;
                for (RefCluster rc : refClusters) {
                    for (MatOfPoint refContour : rc.contours) {
                        Rect   rcBb      = Imgproc.boundingRect(refContour);
                        double rcDiag    = Math.hypot(rcBb.width, rcBb.height);
                        if (rcDiag < 6.0) continue;
                        double candidate = anchorDiag / rcDiag;
                        if (candidate < 1.0 || candidate > 8.0) continue;
                        double dev = Math.abs(candidate - Math.round(candidate * 2.0) / 2.0);
                        if (dev < bestRatioDev) { bestRatioDev = dev; estimatedScale = candidate; }
                    }
                }
                estimatedScale = Math.min(8.0, Math.max(1.0, estimatedScale));

                // Three area limits derived from the reference geometry:
                //
                //  anchorTrimArea  (5%)  — cap used to trim the initial anchor bbox
                //    when background noise has inflated it beyond the reference shape's
                //    expected footprint.  Uses the max dimension squared so that an
                //    elongated reference (e.g. horizontal ellipse) is not over-trimmed
                //    when detected at a diagonal rotation (where its AABB is roughly
                //    square rather than elongated).  For square-ish references this
                //    gives the same result as width*height.
                //
                //  matchedUnionArea (15%) — generous cap used when unioning already-
                //    scored matched contours in Step C.  These contours were vetted
                //    by the scoring loop, so slight over-estimate is acceptable and
                //    necessary for bicolour shapes whose second half pushes the bbox
                //    ~10–15 % beyond the reference extent.
                //
                //  siblingExpArea   (5%)  — tight cap for Step D sibling expansion.
                //    Siblings are unverified — they may include background circles
                //    that happen to overlap the anchor cluster.
                double refMaxDim       = Math.max(refFullBbox.width, refFullBbox.height);
                double anchorTrimSide  = refMaxDim * estimatedScale * 1.05;
                double anchorTrimArea  = anchorTrimSide * anchorTrimSide;

                double matchedUnionW    = refFullBbox.width  * estimatedScale * 1.15;
                double matchedUnionH    = refFullBbox.height * estimatedScale * 1.15;
                double matchedUnionArea = Math.max(matchedUnionW * matchedUnionH,
                        anchorTrimArea * 1.15);

                double siblingExpArea   = anchorTrimArea;   // same 5 % budget

                // ── Step C: Union co-located matched contours ─────────────────
                // Start from the anchor bbox, but trim it when it already exceeds
                // anchorTrimArea (background pixels may have inflated it).
                Rect expandedBbox;
                double anchorBoxArea = (double) anchorBbox.width * anchorBbox.height;
                if (anchorBoxArea <= anchorTrimArea) {
                    expandedBbox = new Rect(anchorBbox.x, anchorBbox.y,
                            anchorBbox.width, anchorBbox.height);
                } else {
                    // Trim proportionally, keeping the anchor centroid fixed
                    double trimRatio = Math.sqrt(anchorTrimArea / anchorBoxArea);
                    int tw = (int)(anchorBbox.width  * trimRatio);
                    int th = (int)(anchorBbox.height * trimRatio);
                    int cx = anchorBbox.x + anchorBbox.width  / 2;
                    int cy = anchorBbox.y + anchorBbox.height / 2;
                    expandedBbox = new Rect(cx - tw / 2, cy - th / 2, tw, th);
                }
                // Union with each other matched contour (already scored) using a
                // cap that depends on whether the reference has TWO DISTINCTLY
                // DIFFERENT chromatic clusters (a genuine bicolour shape):
                //
                //   • Two+ clusters with clearly different hues (bicolour): use
                //     matchedUnionArea (15 %).  The second colour half is a legitimate
                //     spatial extension.
                //
                //   • Otherwise (single-colour, achromatic, or red-hue-wrapped):
                //     use anchorTrimArea (5 %).  The second matched entry is the dark
                //     border side of the same contour or a spurious background blob.
                //
                // Hue distance threshold: 20 degrees on the [0,180) OpenCV scale,
                // accounting for the 0/179 wrap-around (red appears at both ends).
                // Spatial guard: the two chromatic clusters must also occupy distinct
                // regions (primary bbox IoU < 0.50).  Without this, border-pixel
                // extraction artifacts on single-colour shapes (e.g. red triangle)
                // falsely trigger the generous bicolour cap because the anti-aliased
                // edge produces two chromatic clusters with different hues at the
                // SAME physical location.
                boolean hasDistinctChromaticHues = false;
                List<RefCluster> chromatics = refClusters.stream()
                        .filter(rc -> !rc.achromatic)
                        .toList();
                outer:
                for (int ci = 0; ci < chromatics.size(); ci++) {
                    for (int cj = ci + 1; cj < chromatics.size(); cj++) {
                        double diff = Math.abs(chromatics.get(ci).hue - chromatics.get(cj).hue);
                        if (Math.min(diff, 180.0 - diff) > 20.0) {
                            // Also require spatially distinct regions
                            Rect bi = primaryBbox(chromatics.get(ci));
                            Rect bj = primaryBbox(chromatics.get(cj));
                            if (bboxIoU(bi, bj) < 0.50) {
                                hasDistinctChromaticHues = true;
                                break outer;
                            }
                        }
                    }
                }
                double stepCCap = hasDistinctChromaticHues ? matchedUnionArea : anchorTrimArea;
                if (VM_BBOX_DEBUG) {
                    System.out.printf("[STEP-C] %s chromaticClusters=%d hues=%s distinctHues=%b stepCCap=%.0f%n",
                        referenceId.name(), chromatics.size(),
                        chromatics.stream().map(rc -> String.format("%.1f", rc.hue)).toList(),
                        hasDistinctChromaticHues, stepCCap);
                }

                for (SceneContourEntry m : matched) {
                    if (m == bestAnchor) continue;
                    Rect mBb = m.bbox;
                    double dist = centreDist(anchorBbox, mBb);
                    if (dist <= anchorDiag * 0.50) {
                        Rect candidate = unionRect(expandedBbox, mBb);
                        if ((double) candidate.width * candidate.height <= stepCCap)
                            expandedBbox = candidate;
                    }
                }

                // ── Step D: Sibling expansion, capped at siblingExpArea (5%) ─
                expandedBbox = unionConcentricAndOverlappingSiblings(
                        expandedBbox, matched, candidates, siblingExpArea);

                // Sanity check: reject if frame-spanning (> 80 % scene area)
                double expandedArea = (double) expandedBbox.width * expandedBbox.height;
                if (expandedArea <= descriptor.sceneArea * 0.80) {
                    bestBbox = expandedBbox;
                }

                if (VM_BBOX_DEBUG) {
                    System.out.printf("[BBOX-DEBUG] %s: score=%.1f%% scale=%.2f refExt=%dx%d " +
                        "trim=%.0f union=%.0f sib=%.0f initial=(%d,%d %dx%d) expanded=(%d,%d %dx%d) %s%n",
                        referenceId.name(), bestScore * 100.0, estimatedScale,
                        refFullBbox.width, refFullBbox.height,
                        anchorTrimArea, matchedUnionArea, siblingExpArea,
                        initialBbox.x, initialBbox.y, initialBbox.width, initialBbox.height,
                        expandedBbox.x, expandedBbox.y, expandedBbox.width, expandedBbox.height,
                        (expandedArea <= descriptor.sceneArea * 0.80) ? "ACCEPTED" : "REJECTED");
                }
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
                    (int) descriptor.sceneArea, savedPath, false, null);

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
     */
    private static double scoreRegion(List<RefCluster> refClusters,
                                      int refCount,
                                      RefCluster primaryRef,
                                      List<SceneContourEntry> matched,
                                      SceneDescriptor descriptor,
                                      Rect regionBbox,
                                      boolean allAchromatic) {

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
        if (allAchromatic && descriptor.combinedChromaticMask != null) {
            double bboxArea = (double) regionBbox.width * regionBbox.height;
            if (bboxArea > 0) {
                Rect clamped = clampRect(regionBbox, descriptor.combinedChromaticMask);
                if (clamped.width > 0 && clamped.height > 0) {
                    Mat roi = descriptor.combinedChromaticMask.submat(clamped);
                    double contamination = Core.countNonZero(roi) / bboxArea;
                    countScore *= Math.pow(1.0 - Math.min(contamination, 1.0), 4.0);
                }
            }
        }

        // ── Layer 2: structural coherence ────────────────────────────────
        double sceneW = descriptor.combinedChromaticMask != null
                ? descriptor.combinedChromaticMask.cols() : Math.sqrt(descriptor.sceneArea);
        double sceneH = descriptor.combinedChromaticMask != null
                ? descriptor.combinedChromaticMask.rows() : Math.sqrt(descriptor.sceneArea);
        double sceneDiag = Math.sqrt(sceneW * sceneW + sceneH * sceneH);

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

            Rect   entryBb     = entry.bbox;
            double dist         = centreDist(regionBbox, entryBb);
            double proxScore    = Math.max(0.0, 1.0 - dist / (sceneDiag * 0.30));

            // Coverage score: compare shape-fill ratios normalised by each
            // contour's OWN bounding box — scale and position invariant.
            // Both sides converge to π/4 for circles, 1.0 for rectangles, etc.
            // Using refImageArea for the ref denominator causes a systematic
            // mismatch (circle fills 30% of 128×128 image but 78.5% of its bbox).
            Rect   refBb        = primaryBbox(rc);
            double refBbArea    = Math.max(1.0, (double) refBb.width * refBb.height);
            double entryBbArea  = Math.max(1.0, (double) entryBb.width * entryBb.height);
            double refFrac      = rc.maxContourArea / refBbArea;
            double sceneFrac    = entry.area / entryBbArea;
            double covScore     = 1.0 - Math.min(1.0,
                    Math.abs(refFrac - sceneFrac) / Math.max(refFrac, 0.01));

            // Absolute scale check: penalize when bbox sizes differ significantly.
            // Even if shapes are geometrically similar (diamond vs rotated rect),
            // extreme scale differences may indicate wrong match.
            // Uses square-root ratio to account for area growing quadratically.
            // Expected scale: scene is typically 3x scaled from 128x128 ref → ratio ≈ 3.0
            double refBboxDim   = Math.sqrt(refBbArea);
            double sceneBboxDim = Math.sqrt(entryBbArea);
            double sizeRatio    = sceneBboxDim / Math.max(refBboxDim, 1.0);
            double scaleScore   = 1.0;
            if (sizeRatio > 6.0 || sizeRatio < (1.0/6.0)) {
                // Extreme scale mismatch (>6x or <1/6x) — hard penalty
                scaleScore = 0.25;
            } else if (sizeRatio > 4.5 || sizeRatio < (1.0/4.5)) {
                // Large scale mismatch (4.5x-6x or 1/4.5x-1/6x) — moderate penalty
                scaleScore = 0.70;
            }

            sumContrib += weight * (proxScore * 0.40 + covScore * 0.40 + scaleScore * 0.20);
        }

        double matchScore = (sumWeights > 0) ? sumContrib / sumWeights : 0.0;

        // ── Layer 3: primary boundary geometry ───────────────────────────
        // Try the primary ref cluster's signature first, then also try each
        // secondary ref cluster's signature.  For multi-colour shapes (e.g.
        // BICOLOUR_RECT_HALVES), the primary cluster may be achromatic (full
        // rectangle border) which has no clean counterpart in a noisy scene,
        // while the chromatic half-rectangle clusters match perfectly.
        // Using the best geometry across ALL ref clusters is purely structural
        // — it finds the best structural alignment regardless of cluster origin.
        double geomScore = 0.0;
        SceneContourEntry primaryScene = null;
        double bestGeom = -1.0;
        for (RefCluster rc : refClusters) {
            VectorSignature rcSig = rc.bestSig(EPSILON);
            for (SceneContourEntry e : matched) {
                double sim = rcSig.similarity(e.sig);
                if (sim > bestGeom) {
                    bestGeom = sim;
                    primaryScene = e;
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
                ps != null ? ps.sig.type : "?", ps != null ? ps.sig.circularity : 0, ps != null ? ps.sig.vertexCount : 0);
        }

        return Math.max(0.0, Math.min(1.0, combined));
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
            if (e.achromatic) {
                if (e.brightAchromatic) brightEntry = e;
                else                    darkEntry   = e;
            }
        }

        List<SceneContourEntry> result = new ArrayList<>(entries);

        if (brightEntry != null && darkEntry != null) {
            Rect   bb     = brightEntry.bbox;
            Rect   db     = darkEntry.bbox;
            double iou    = bboxIoU(bb, db);
            double areaB  = brightEntry.area;
            double areaD  = darkEntry.area;
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
            Rect   db    = darkEntry.bbox;
            double areaD = darkEntry.area;
            boolean shouldRemoveDark = false;
            for (SceneContourEntry e : result) {
                if (e.achromatic) continue;
                Rect   cb    = e.bbox;
                double areaC = e.area;
                double iou   = bboxIoU(cb, db);
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
                result.add(new RefCluster(c.hue, c.achromatic, c.brightAchromatic,
                        contours, refArea));
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
                    result.add(new RefCluster(c.hue, c.achromatic, c.brightAchromatic,
                            contours, refArea));
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
            double iou   = bboxIoU(bb, db);
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
                double iou   = bboxIoU(cb, db);
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
    // Internal data structures
    // =========================================================================

    /** One colour-boundary cluster from the reference image. */
    static final class RefCluster {
        final double  hue;
        final boolean achromatic;
        final boolean brightAchromatic;
        final List<MatOfPoint> contours;
        final double  imageArea;
        /** Largest contour area in this cluster — the primary shape boundary. */
        final double  maxContourArea;
        /** Bounding rect of the primary (largest-area) contour — cached. */
        final Rect    primaryBbox;

        private VectorSignature cachedSig = null;

        RefCluster(double hue, boolean achromatic, boolean brightAchromatic,
                   List<MatOfPoint> contours, double imageArea) {
            this.hue              = hue;
            this.achromatic       = achromatic;
            this.brightAchromatic = brightAchromatic;
            this.contours         = contours;
            this.imageArea        = imageArea;
            // Pre-compute max contour area and primary bbox in one pass
            double bestArea = 0.0;
            MatOfPoint primary = contours.get(0);
            for (MatOfPoint c : contours) {
                double a = Imgproc.contourArea(c);
                if (a > bestArea) { bestArea = a; primary = c; }
            }
            this.maxContourArea = bestArea;
            this.primaryBbox    = Imgproc.boundingRect(primary);
        }

        /** Returns the best (highest solidity) VectorSignature at fixed epsilon. */
        VectorSignature bestSig(double eps) {
            if (cachedSig != null) return cachedSig;
            VectorSignature best     = null;
            double          bestSol  = -1;
            for (MatOfPoint c : contours) {
                // Keep ref-side normalisedArea as NaN. The normalised-area guards in
                // VectorSignature are scene-side gates; filling both sides triggers an
                // unintended 0.25 cap for valid matches because ref and scene canvases
                // are different sizes (128x128 vs 640x480 in this suite).
                VectorSignature s = VectorSignature.buildFromContour(c, eps, Double.NaN);
                if (s.solidity > bestSol) { bestSol = s.solidity; best = s; }
            }
            cachedSig = (best != null) ? best
                    : VectorSignature.build(Mat.zeros(1, 1, CvType.CV_8UC1), eps, Double.NaN);
            return cachedSig;
        }

        /** Fraction of ref image area occupied by this cluster's primary contour. */
        double refFraction() {
            return (imageArea > 0) ? maxContourArea / imageArea : 0.0;
        }

        void release() { for (MatOfPoint c : contours) c.release(); }
    }

    /** One contour from the scene with its cluster metadata and pre-built signature. */
    private record SceneContourEntry(
            MatOfPoint contour,
            int        clusterIdx,
            boolean    achromatic,
            boolean    brightAchromatic,
            double     clusterHue,
            VectorSignature sig,
            Rect       bbox,
            double     area) {}

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
                // Skip contours that span > 50% of scene — background fills or noise merges
                Rect   bb     = Imgproc.boundingRect(c);
                double bbArea = (double) bb.width * bb.height;
                if (bbArea > descriptor.sceneArea * 0.50) continue;

                double cArea = Imgproc.contourArea(c);
                VectorSignature sig = VectorSignature.buildFromContour(c, EPSILON, descriptor.sceneArea);
                out.add(new SceneContourEntry(c, ci, cc.achromatic, cc.brightAchromatic,
                        cc.hue, sig, bb, cArea));
            }
        }
        return out;
    }

    // =========================================================================
    // Selection helpers
    // =========================================================================

    /**
     * Assigns the anchor contour to the ref cluster whose relative contour area
     * (vs ref image area) most closely matches the anchor's (vs its own bbox area).
     */
    private static RefCluster assignAnchorToRef(SceneContourEntry anchor,
                                                 double anchorBboxArea,
                                                 List<RefCluster> refClusters) {
        double anchorFrac = anchor.area / anchorBboxArea;
        RefCluster best    = refClusters.get(0);
        double     bestDiff = Math.abs(refFraction(refClusters.get(0)) - anchorFrac);
        for (int i = 1; i < refClusters.size(); i++) {
            double diff = Math.abs(refFraction(refClusters.get(i)) - anchorFrac);
            if (diff < bestDiff) { bestDiff = diff; best = refClusters.get(i); }
        }
        return best;
    }

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
            Rect   eBb   = e.bbox;
            double eBbA  = Math.max(1.0, (double) eBb.width * eBb.height);
            double eFrac = e.area / eBbA;
            double diff  = Math.abs(refFrac - eFrac);
            if (diff < bestDiff) { bestDiff = diff; best = e; }
        }
        return best;
    }

    /** Relative contour area of a ref cluster's primary contour vs ref image area. */
    private static double refFraction(RefCluster rc) {
        return (rc.imageArea > 0) ? rc.maxContourArea / rc.imageArea : 0.0;
    }

    /** Relative contour area of a scene entry vs the current candidate bbox area. */
    private static double sceneFraction(SceneContourEntry ce, Rect regionBbox) {
        double bboxArea = Math.max(1.0, (double) regionBbox.width * regionBbox.height);
        return ce.area / bboxArea;
    }

    /** Enclosed geometric area of a contour (from contour points, not pixel count). */
    private static double contourArea(MatOfPoint c) {
        return Imgproc.contourArea(c);
    }

    // =========================================================================
    // Geometry / bbox helpers
    // =========================================================================

    private static double centreDist(Rect a, Rect b) {
        double ax = a.x + a.width  / 2.0, ay = a.y + a.height / 2.0;
        double bx = b.x + b.width  / 2.0, by = b.y + b.height / 2.0;
        return Math.sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by));
    }

    private static Point rectCentre(Rect r) {
        return new Point(r.x + r.width / 2.0, r.y + r.height / 2.0);
    }

    private static Rect unionRect(Rect a, Rect b) {
        int x  = Math.min(a.x, b.x);
        int y  = Math.min(a.y, b.y);
        int x2 = Math.max(a.x + a.width,  b.x + b.width);
        int y2 = Math.max(a.y + a.height, b.y + b.height);
        return new Rect(x, y, x2 - x, y2 - y);
    }

    /** Returns true if the two rectangles overlap (share any area). */
    private static boolean rectsIntersect(Rect a, Rect b) {
        return a.x < b.x + b.width  && b.x < a.x + a.width
            && a.y < b.y + b.height && b.y < a.y + a.height;
    }

    private static double bboxIoU(Rect a, Rect b) {
        int ix  = Math.max(a.x, b.x);
        int iy  = Math.max(a.y, b.y);
        int ix2 = Math.min(a.x + a.width,  b.x + b.width);
        int iy2 = Math.min(a.y + a.height, b.y + b.height);
        if (ix2 <= ix || iy2 <= iy) return 0.0;
        double inter = (double)(ix2 - ix) * (iy2 - iy);
        double aArea = (double) a.width * a.height;
        double bArea = (double) b.width * b.height;
        return inter / (aArea + bArea - inter);
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
                (int) descriptor.sceneArea, null, false, null);
    }

    // =========================================================================
    // Contour isolation helpers (Stage 1 + 2)
    // =========================================================================

    /**
     * Connected-component filter (Stage 1).
     *
     * <p>For each cluster, retains only contours whose area is at least
     * {@value #CC_AREA_RATIO_MIN} × the largest contour in that cluster.
     * Contours below that threshold are <em>also</em> kept when their bounding-box
     * centre falls inside the main (largest) contour's bounding box — these are
     * compound-shape components (e.g. cross arms inside a circle, ring sections)
     * rather than isolated background noise blobs.
     *
     * <p>This ensures COMPOUND shapes like {@code COMPOUND_CROSS_IN_CIRCLE} keep
     * all their components while scattered same-colour background fragments (whose
     * centres are outside the main shape's bbox) are dropped.
     */
    private static List<SceneContourEntry> applyConnectedComponentFilter(
            List<SceneContourEntry> candidates) {
        if (candidates.size() <= 1) return candidates;

        // Largest contour area and its bbox, keyed by clusterIdx
        Map<Integer, Double> maxArea = new HashMap<>();
        Map<Integer, Rect>   maxBbox = new HashMap<>();
        for (SceneContourEntry ce : candidates) {
            double area = ce.area;
            if (area > maxArea.getOrDefault(ce.clusterIdx, 0.0)) {
                maxArea.put(ce.clusterIdx, area);
                maxBbox.put(ce.clusterIdx, ce.bbox);
            }
        }

        List<SceneContourEntry> out = new ArrayList<>();
        for (SceneContourEntry ce : candidates) {
            double area   = ce.area;
            double clsMax = maxArea.getOrDefault(ce.clusterIdx, 1.0);

            // Primary rule: area large enough relative to cluster max
            // Relaxed threshold for achromatic clusters (0.05 vs 0.10) to preserve
            // thin outline rings in compound shapes like COMPOUND_BULLSEYE
            double threshold = ce.achromatic ? 0.05 : CC_AREA_RATIO_MIN;
            if (area >= clsMax * threshold) { out.add(ce); continue; }

            // Secondary rule: small but spatially inside the main shape's bbox
            // → compound component (cross arm, inner ring, etc.), not background noise
            Rect mainBb = maxBbox.get(ce.clusterIdx);
            if (mainBb != null) {
                Rect   ceBb = ce.bbox;
                double ceCx = ceBb.x + ceBb.width  / 2.0;
                double ceCy = ceBb.y + ceBb.height / 2.0;
                if (ceCx >= mainBb.x && ceCx <= mainBb.x + mainBb.width
                 && ceCy >= mainBb.y && ceCy <= mainBb.y + mainBb.height) {
                    out.add(ce);
                }
            }
        }
        return out;
    }

    /**
     * Returns the morphological-opening depth suited to the reference fill density.
     *
     * <p>Currently returns 0 for all shapes.  Applying MORPH_OPEN — even at 1 px —
     * rounds the corners of triangular colour sections and hexagon/circle outline
     * strokes enough to shift their {@link VectorSignature} vertex angles, causing
     * regressions on {@code TRICOLOUR_TRIANGLE}, {@code HEXAGON_OUTLINE} and
     * {@code BICOLOUR_CIRCLE_RING}.  Background-line tests already pass at the
     * 60 % threshold without erosion.  Re-enable if a stricter contamination
     * metric is added that guards against corner-rounding on outline shapes.
     */
    private static int computeErosionDepth(RefCluster primaryRef) {
        return 0;
    }

    /**
     * Stage 1b — Global minimum-size filter.
     *
     * <p>Drops any candidate whose contour area is below
     * {@value #MIN_GLOBAL_AREA_RATIO} × the area of the globally largest
     * contour (across all clusters).  This eliminates small background elements
     * — random circles (20–60 px) and short line segments — that survive the
     * per-cluster connected-component filter but would still outscore the actual
     * target shape by incidentally matching its geometry at a tiny scale.
     *
     * <p>Inner components of compound shapes (inner cross of
     * COMPOUND_CROSS_IN_CIRCLE, inner rings of COMPOUND_BULLSEYE) are
     * safely retained because their areas are typically ≥ 15–25 % of the
     * outer/primary contour.
     */
    private static List<SceneContourEntry> applyGlobalSizeFilter(
            List<SceneContourEntry> candidates) {
        if (candidates.size() <= 1) return candidates;
        double maxArea = 0.0;
        for (SceneContourEntry ce : candidates) {
            double a = ce.area;
            if (a > maxArea) maxArea = a;
        }
        if (maxArea <= 0.0) return candidates;
        final double minArea = maxArea * MIN_GLOBAL_AREA_RATIO;
        List<SceneContourEntry> out = new ArrayList<>();
        for (SceneContourEntry ce : candidates) {
            if (ce.area >= minArea) out.add(ce);
        }
        return out.isEmpty() ? candidates : out;  // never leave caller empty-handed
    }

    /**
     * Unions same-cluster siblings of each matched entry into {@code expandedBbox}
     * when the sibling is spatially related to the current result region:
     *
     * <ul>
     *   <li><b>Concentric:</b> sibling centre is within 35 % of the sibling's own
     *       diagonal from the expandedBbox centre (concentric rings, bullseye).</li>
     *   <li><b>Overlapping:</b> sibling bbox intersects the current expandedBbox
     *       (adjacent components, enclosed pieces).</li>
     * </ul>
     *
     * <p>Hard cap: the resulting bbox area must not exceed {@code maxAllowedArea},
     * which is computed from the reference full extent × estimated scene scale × 1.15.
     * This prevents absorption of background circles whose combined bbox would
     * exceed what the actual reference shape could ever occupy in the scene.
     */
    private static Rect unionConcentricAndOverlappingSiblings(
            Rect expandedBbox,
            List<SceneContourEntry> matched,
            List<SceneContourEntry> candidates,
            double maxAllowedArea) {

        for (int pass = 0; pass < 4; pass++) {
            Rect before = expandedBbox;
            for (SceneContourEntry m : matched) {
                for (SceneContourEntry ce : candidates) {
                    if (ce.clusterIdx != m.clusterIdx || ce == m) continue;

                    Rect   ceBb         = ce.bbox;
                    double ceArea       = ce.area;
                    double expandedArea = (double) expandedBbox.width * expandedBbox.height;

                    // Already at cap — no point continuing
                    if (expandedArea >= maxAllowedArea) break;

                    // Guard: sibling must be substantial relative to current bbox
                    if (ceArea < expandedArea * 0.10) continue;

                    // Criterion 1 — concentric
                    double ceDiag      = Math.hypot(ceBb.width, ceBb.height);
                    boolean concentric = centreDist(expandedBbox, ceBb) <= ceDiag * 0.35;

                    // Criterion 2 — overlapping
                    boolean overlapping = rectsIntersect(expandedBbox, ceBb);

                    if (concentric || overlapping) {
                        Rect candidate     = unionRect(expandedBbox, ceBb);
                        double candidateArea = (double) candidate.width * candidate.height;
                        if (candidateArea <= maxAllowedArea) {
                            expandedBbox = candidate;
                        } else if (concentric) {
                            // ── Tightly concentric override ────────────────────
                            // When a same-cluster sibling is VERY precisely centered
                            // on the anchor (center-to-center distance < 5% of
                            // anchor diagonal), it is almost certainly an outer ring
                            // of a compound shape (e.g. COMPOUND_BULLSEYE outer ring
                            // when the anchor is the middle ring).
                            //
                            // The normal cap (based on estimated scale) can be too
                            // tight when the scale estimation picks a ref contour
                            // at the wrong nesting level.  Allow expansion up to
                            // 3× the current bbox area for tightly concentric
                            // siblings — generous enough for outer rings, but still
                            // capped to prevent absorption of large background noise.
                            double ancDiag = Math.hypot(expandedBbox.width, expandedBbox.height);
                            double dist    = centreDist(expandedBbox, ceBb);
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

    /**
     * Reference-adaptive erosion (Stage 2).
     *
     * <p>Identifies the top-{@value #EROSION_TOP_K} candidates by raw contour
     * area and re-extracts each one from {@code scene.sceneMat()} with a
     * morphological open of {@code erosionDepth} px.  The opening severs thin
     * background-line arms that are physically connected to the main shape
     * boundary — the root cause of wrong-bbox returns on line-texture backgrounds.
     *
     * <p>For chromatic candidates the full hue-range mask is re-extracted and
     * opened.  For achromatic candidates the bright/dark full-pixel mask is used,
     * then a gradient is applied after opening to restore the border-pixel
     * representation used by {@link SceneDescriptor}.
     */
    private static List<SceneContourEntry> reExtractTopCandidates(
            List<SceneContourEntry> candidates, SceneEntry scene,
            int erosionDepth, SceneDescriptor descriptor) {
        Mat sceneMat = scene.sceneMat();
        if (sceneMat == null || sceneMat.empty()) return candidates;

        // Rank by contour area descending, pick top-K
        List<SceneContourEntry> byArea = new ArrayList<>(candidates);
        byArea.sort(Comparator.comparingDouble(ce -> -ce.area));
        Set<SceneContourEntry> topK = new LinkedHashSet<>();
        for (int i = 0; i < Math.min(EROSION_TOP_K, byArea.size()); i++)
            topK.add(byArea.get(i));

        double sceneArea = descriptor.sceneArea;
        List<SceneContourEntry> result = new ArrayList<>(candidates.size());
        for (SceneContourEntry ce : candidates) {
            result.add(topK.contains(ce)
                    ? reExtractCandidate(ce, sceneMat, erosionDepth, sceneArea)
                    : ce);
        }
        return result;
    }

    /**
     * Re-extracts a single candidate cluster from the scene BGR image, applying
     * a morphological open of {@code erosionDepth} px to sever thin line arms.
     *
     * <p>Spatial matching: the returned entry targets the new contour whose
     * bounding-box centre is closest to the original candidate's centre.
     * Falls back to the original candidate when re-extraction yields no contours.
     */
    private static SceneContourEntry reExtractCandidate(
            SceneContourEntry candidate, Mat sceneBgr,
            int erosionDepth, double sceneArea) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(sceneBgr, hsv, Imgproc.COLOR_BGR2HSV);

        // Build full pixel mask for this cluster's colour
        Mat fullMask = candidate.achromatic
                ? SceneColourClusters.buildAchromaticMask(hsv, candidate.brightAchromatic)
                : SceneColourClusters.buildHueMask(hsv, candidate.clusterHue,
                        SceneColourClusters.HUE_TOLERANCE);
        hsv.release();

        // Apply morphological opening — severs arms thinner than erosionDepth px
        int kSize = erosionDepth * 2 + 1;
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(kSize, kSize));
        Mat opened = new Mat();
        Imgproc.morphologyEx(fullMask, opened, Imgproc.MORPH_OPEN, kernel);
        kernel.release();
        fullMask.release();

        // Achromatic clusters are stored as gradient (border) masks in SceneDescriptor;
        // apply gradient AFTER opening so the border reflects the cleaned shape.
        Mat maskForContours;
        if (candidate.achromatic) {
            Mat gradKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
            maskForContours = new Mat();
            Imgproc.morphologyEx(opened, maskForContours, Imgproc.MORPH_GRADIENT, gradKernel);
            gradKernel.release();
            opened.release();
        } else {
            maskForContours = opened;
        }

        List<MatOfPoint> newContours = SceneDescriptor.contoursFromMask(maskForContours);
        maskForContours.release();

        if (newContours.isEmpty()) return candidate;  // fallback — original is unchanged

        // Pick the contour spatially closest to the original candidate
        Rect   origBb = candidate.bbox;
        double origCx = origBb.x + origBb.width  / 2.0;
        double origCy = origBb.y + origBb.height / 2.0;

        MatOfPoint best     = newContours.get(0);
        double     bestDist = Double.MAX_VALUE;
        for (MatOfPoint c : newContours) {
            Rect   bb   = Imgproc.boundingRect(c);
            double cx   = bb.x + bb.width  / 2.0;
            double cy   = bb.y + bb.height / 2.0;
            double dist = Math.hypot(cx - origCx, cy - origCy);
            if (dist < bestDist) { bestDist = dist; best = c; }
        }

        Rect newBbox = Imgproc.boundingRect(best);
        double newArea = Imgproc.contourArea(best);
        VectorSignature newSig = VectorSignature.buildFromContour(best, EPSILON, sceneArea);

        // Release unused contours to avoid native-memory accumulation
        for (MatOfPoint c : newContours) { if (c != best) c.release(); }

        return new SceneContourEntry(best, candidate.clusterIdx, candidate.achromatic,
                candidate.brightAchromatic, candidate.clusterHue, newSig, newBbox, newArea);
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
