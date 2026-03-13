package org.example.matchers;

import org.example.analytics.AnalysisResult;
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
    private static final double CLUSTER_PENALTY_K      = 0.7;
    /** Steeper decay rate for missing boundaries (scene has fewer than ref). */
    private static final double CLUSTER_PENALTY_K_MISS = 1.5;

    // ── Layer 2 constants ─────────────────────────────────────────────────────
    /** Max centre-to-centre distance as a fraction of scene diagonal for expansion. */
    private static final double PROXIMITY_THRESHOLD = 0.35;
    /** Weight of secondary (non-primary) boundaries in Layer 2. */
    private static final double SECONDARY_WEIGHT = 0.30;

    // ── Deduplication constants ───────────────────────────────────────────────
    /** Min bbox IoU to consider a bright/dark achromatic pair as same physical edge. */
    private static final double DEDUP_IOU_MIN        = 0.50;
    /** Min contourArea ratio (min/max) to confirm same physical edge (not inner/outer ring). */
    private static final double DEDUP_AREA_RATIO_MIN = 0.90;

    // ── Geometry constant ─────────────────────────────────────────────────────
    private static final double EPSILON = 0.04;   // single variant epsilon

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

            double bestScore = 0.0;
            Rect   bestBbox  = null;

            for (SceneContourEntry anchor : candidates) {
                Rect anchorBbox = Imgproc.boundingRect(anchor.contour);

                // ── Anchor-to-ref assignment ──────────────────────────────
                // The anchor represents the ref cluster whose relative contour area
                // (vs ref image) most closely matches the anchor's (vs its own bbox area).
                double anchorBboxArea = Math.max(1.0, (double) anchorBbox.width * anchorBbox.height);
                RefCluster anchorRef  = assignAnchorToRef(anchor, anchorBboxArea, refClusters);

                // ── Expansion loop ────────────────────────────────────────
                // Select additional scene clusters by proximity + relative size only.
                // No colour, no geometry.
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

                        // Proximity gate
                        double dist = centreDist(anchorBbox, Imgproc.boundingRect(ce.contour));
                        if (dist > sceneDiag * PROXIMITY_THRESHOLD) continue;

                        // Relative size match — compare against candidate bbox so far
                        double ceFrac = sceneFraction(ce, regionBbox);
                        double diff   = Math.abs(refFrac - ceFrac);
                        if (diff < bestDiff) { bestDiff = diff; best = ce; }
                    }

                    if (best != null) {
                        matched.add(best);
                        usedIdx.add(best.clusterIdx);
                        // Only expand regionBbox for non-background-scale contours
                        Rect bestBb = Imgproc.boundingRect(best.contour);
                        double bestArea = (double) bestBb.width * bestBb.height;
                        if (bestArea < descriptor.sceneArea * 0.60) {
                            regionBbox = unionRect(regionBbox, bestBb);
                        }
                    }
                }

                // ── Deduplication ─────────────────────────────────────────
                // Collapse bright/dark achromatic pairs that trace the same physical edge
                matched = deduplicateAchromaticPairs(matched);

                // ── Score this candidate ──────────────────────────────────
                double score = scoreRegion(refClusters, refCount, primaryRef,
                        matched, descriptor, regionBbox);

                if (score > bestScore) {
                    bestScore = score;
                    bestBbox  = regionBbox;
                }
            }

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
     */
    private static double scoreRegion(List<RefCluster> refClusters,
                                      int refCount,
                                      RefCluster primaryRef,
                                      List<SceneContourEntry> matched,
                                      SceneDescriptor descriptor,
                                      Rect regionBbox) {

        // ── Layer 1: boundary count ───────────────────────────────────────
        int matchedCount = matched.size();
        double countScore;
        if (refCount == 0) {
            countScore = 1.0;
        } else {
            int diff = matchedCount - refCount;
            countScore = diff > 0
                    ? Math.exp(-CLUSTER_PENALTY_K * 2.0 * diff)        // extra — steep
                    : Math.exp(-CLUSTER_PENALTY_K_MISS * Math.abs(diff)); // missing — steeper
        }

        // ── Fix B: chromatic contamination (achromatic refs only) ─────────
        boolean allAchromatic = refClusters.stream().allMatch(rc -> rc.achromatic);
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

            Rect   entryBb     = Imgproc.boundingRect(entry.contour);
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
            double sceneFrac    = contourArea(entry.contour) / entryBbArea;
            double covScore     = 1.0 - Math.min(1.0,
                    Math.abs(refFrac - sceneFrac) / Math.max(refFrac, 0.01));

            sumContrib += weight * (proxScore * 0.50 + covScore * 0.50);
        }

        double matchScore = (sumWeights > 0) ? sumContrib / sumWeights : 0.0;

        // ── Layer 3: primary boundary geometry ───────────────────────────
        double geomScore = 0.0;
        SceneContourEntry primaryScene =
                findMatchedEntryForRef(primaryRef, matched);
        if (primaryScene != null) {
            VectorSignature refSig   = primaryRef.bestSig(EPSILON);
            VectorSignature sceneSig = primaryScene.sig;
            geomScore = refSig.similarity(sceneSig);
        }

        double combined = countScore * W_COUNT
                        + matchScore  * W_MATCH
                        + geomScore   * W_GEOM;

        // Temporary debug
        System.out.printf("[VM] refCnt=%d matchCnt=%d L1=%.3f L2=%.3f L3=%.3f => %.3f%n",
            refCount, matchedCount, countScore, matchScore, geomScore, combined);

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
            Rect   bb     = Imgproc.boundingRect(brightEntry.contour);
            Rect   db     = Imgproc.boundingRect(darkEntry.contour);
            double iou    = bboxIoU(bb, db);
            double areaB  = contourArea(brightEntry.contour);
            double areaD  = contourArea(darkEntry.contour);
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
            Rect   db    = Imgproc.boundingRect(darkEntry.contour);
            double areaD = contourArea(darkEntry.contour);
            boolean shouldRemoveDark = false;
            for (SceneContourEntry e : result) {
                if (e.achromatic) continue;
                Rect   cb    = Imgproc.boundingRect(e.contour);
                double areaC = contourArea(e.contour);
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
        List<SceneColourClusters.Cluster> raw =
                SceneColourClusters.extractFromBorderPixels(refBgr);
        List<RefCluster> result = new ArrayList<>();
        double refArea = (double) refBgr.rows() * refBgr.cols();

        for (SceneColourClusters.Cluster c : raw) {
            List<MatOfPoint> contours = SceneDescriptor.contoursFromMask(c.mask);
            if (!contours.isEmpty()) {
                result.add(new RefCluster(c.hue, c.achromatic, c.brightAchromatic,
                        contours, refArea));
            }
            c.release();
        }

        // Fallback: no border-pixel clusters found — use full extraction
        if (result.isEmpty()) {
            List<SceneColourClusters.Cluster> fallback = SceneColourClusters.extract(refBgr);
            for (SceneColourClusters.Cluster c : fallback) {
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
        MatOfPoint primary = rc.contours.stream()
                .max(java.util.Comparator.comparingDouble(Imgproc::contourArea))
                .orElse(rc.contours.get(0));
        return Imgproc.boundingRect(primary);
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

        private VectorSignature cachedSig = null;

        RefCluster(double hue, boolean achromatic, boolean brightAchromatic,
                   List<MatOfPoint> contours, double imageArea) {
            this.hue              = hue;
            this.achromatic       = achromatic;
            this.brightAchromatic = brightAchromatic;
            this.contours         = contours;
            this.imageArea        = imageArea;
            this.maxContourArea   = contours.stream()
                    .mapToDouble(c -> Imgproc.contourArea(c))
                    .max().orElse(0.0);
        }

        /** Returns the best (highest solidity) VectorSignature at fixed epsilon. */
        VectorSignature bestSig(double eps) {
            if (cachedSig != null) return cachedSig;
            VectorSignature best     = null;
            double          bestSol  = -1;
            for (MatOfPoint c : contours) {
                VectorSignature s = VectorSignature.buildFromContour(c, eps, Double.NaN);
                if (s.solidity > bestSol) { bestSol = s.solidity; best = s; }
            }
            cachedSig = (best != null) ? best
                    : VectorSignature.build(Mat.zeros(1, 1, CvType.CV_8UC1), eps, imageArea);
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
            VectorSignature sig) {}

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

                VectorSignature sig = VectorSignature.buildFromContour(c, EPSILON, descriptor.sceneArea);
                out.add(new SceneContourEntry(c, ci, cc.achromatic, cc.brightAchromatic,
                        cc.hue, sig));
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
        double anchorFrac = contourArea(anchor.contour) / anchorBboxArea;
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
            Rect   eBb   = Imgproc.boundingRect(e.contour);
            double eBbA  = Math.max(1.0, (double) eBb.width * eBb.height);
            double eFrac = contourArea(e.contour) / eBbA;
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
        return contourArea(ce.contour) / bboxArea;
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
}
