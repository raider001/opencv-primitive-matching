package org.example.matchers.vectormatcher;

import org.example.analytics.AnalysisResult;
import org.example.colour.ColourCluster;
import org.example.colour.ExperimentalSceneColourClusters;
import org.example.colour.SceneColourClusters;
import org.example.factories.ReferenceId;
import org.example.matchers.SceneContourEntry;
import org.example.matchers.SceneDescriptor;
import org.example.matchers.VectorSignature;
import org.example.matchers.VectorVariant;
import org.example.matchers.vectormatcher.components.*;
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

                // ── OPT-G: Early exit — skip scoring when type is incompatible ──
                double score = 0.0;
                boolean skipped = bestScore >= 0.70
                        && !isShapeTypeCompatible(anchor.sig().type, primaryRefSig.type);

                if (!skipped) {
                    // ── Score this candidate ──────────────────────────────────
                    RegionScore regionScore = RegionScorer.score(refClusters, refCount,
                            primaryRef, matched, descriptor, regionBbox, allAchromatic,
                            sceneDiag, EPSILON);
                    score = regionScore.combined();
                    if (score > bestScore) {
                        bestScore       = score;
                        bestBbox        = regionBbox;
                        bestAnchor      = anchor;
                        bestLayerScores = new double[]{
                                regionScore.countScore(),
                                regionScore.matchScore(),
                                regionScore.geomScore()};
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

            // ── Anchor re-selection — correct bbox when best score landed on
            //    a small background element instead of the placed shape ──────
            if (bestScore >= 0.60 && bestAnchor != null) {
                var resel = reSelectAnchor(bestScore, bestAnchor, bestBbox,
                        anchorScores, anchorBboxes, anchorEntries);
                bestBbox   = resel.bbox;
                bestAnchor = resel.anchor;
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


    // =========================================================================
    // Geometry / bbox helpers
    // =========================================================================

    /** Returns true if the shape type is any closed polygon variant. */
    private static boolean isClosedPoly(VectorSignature.ShapeType t) {
        return t == VectorSignature.ShapeType.CLOSED_CONVEX_POLY
            || t == VectorSignature.ShapeType.CLOSED_CONCAVE_POLY;
    }

    /**
     * Shape-type compatibility check for the OPT-G early exit.
     * Returns {@code true} when the anchor and ref types are structurally compatible.
     */
    private static boolean isShapeTypeCompatible(VectorSignature.ShapeType anchor,
                                                 VectorSignature.ShapeType ref) {
        if (anchor == ref) return true;
        if (anchor == VectorSignature.ShapeType.COMPOUND
                || ref == VectorSignature.ShapeType.COMPOUND) return true;
        return (isClosedPoly(anchor) && isClosedPoly(ref))
            || (anchor == VectorSignature.ShapeType.CIRCLE && isClosedPoly(ref))
            || (isClosedPoly(anchor) && ref == VectorSignature.ShapeType.CIRCLE);
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

    /** Lightweight carrier for anchor re-selection results. */
    private record AnchorSelection(SceneContourEntry anchor, Rect bbox) {}

    /**
     * Post-loop anchor re-selection — corrects the bounding box when a small
     * background element outscored the real shape.
     *
     * <p><b>Path A (non-prominent):</b> best anchor bbox &lt; 50 % of the largest
     * candidate → re-select to the largest-bbox candidate scoring ≥ 70 % of best.
     *
     * <p><b>Path B (prominent):</b> best anchor is large → only allow a tight
     * same-size swap (bbox area ≥ 90 %, score ≥ 90 %, larger contour area).
     *
     * <p>Purely geometry-driven — no colour terms.
     */
    private static AnchorSelection reSelectAnchor(
            double bestScore,
            SceneContourEntry bestAnchor,
            Rect bestBbox,
            List<double[]> anchorScores,
            List<Rect> anchorBboxes,
            List<SceneContourEntry> anchorEntries) {

        double bestAnchorBboxArea = (double) bestAnchor.bbox().width * bestAnchor.bbox().height;
        double maxBboxArea = bestAnchorBboxArea;
        for (SceneContourEntry ce : anchorEntries) {
            double bb = (double) ce.bbox().width * ce.bbox().height;
            if (bb > maxBboxArea) maxBboxArea = bb;
        }

        VectorSignature.ShapeType bestAnchorType = bestAnchor.sig() != null
                ? bestAnchor.sig().type : null;

        if (bestAnchorBboxArea < maxBboxArea * 0.50) {
            // Path A: non-prominent — re-select to largest bbox
            double reselScoreThreshold = bestScore * 0.70;
            double reselBestBboxArea   = bestAnchorBboxArea;
            for (int ri = 0; ri < anchorScores.size(); ri++) {
                double rScore = anchorScores.get(ri)[0];
                SceneContourEntry reselCandidate = anchorEntries.get(ri);
                double rBboxArea = (double) reselCandidate.bbox().width * reselCandidate.bbox().height;
                if (rScore >= reselScoreThreshold && rBboxArea > reselBestBboxArea) {
                    if (!isTypeCompatible(bestAnchorType, reselCandidate)) continue;
                    reselBestBboxArea = rBboxArea;
                    bestBbox   = anchorBboxes.get(ri);
                    bestAnchor = reselCandidate;
                }
            }
        } else {
            // Path B: prominent — tight same-size contour swap
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
                    bestBbox   = anchorBboxes.get(ri);
                    bestAnchor = reselCandidate;
                }
            }
        }
        return new AnchorSelection(bestAnchor, bestBbox);
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
