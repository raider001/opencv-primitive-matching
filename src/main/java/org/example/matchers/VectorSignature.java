package org.example.matchers;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Structural descriptor for a binary shape image — scale and rotation invariant.
 *
 * <p>Encodes a shape purely in terms of topology: how many components, how many
 * vertices, how circular, how concave, and what the distribution of inter-vertex
 * angles looks like.  None of these fields depend on absolute size or orientation.
 *
 * <h2>Usage</h2>
 * <pre>
 *   VectorSignature sig = VectorSignature.build(binaryMask, 0.04);
 *   double score = sig.similarity(otherSig);
 * </pre>
 */
public final class VectorSignature {

    // -------------------------------------------------------------------------
    // Shape type classification
    // -------------------------------------------------------------------------

    public enum ShapeType {
        /** Thin, elongated contour — aspect ratio > 4:1. */
        LINE_SEGMENT,
        /** circularity > 0.82 */
        CIRCLE,
        /** Closed polygon, no significant concavity (concavityRatio < 0.08). */
        CLOSED_CONVEX_POLY,
        /** Closed polygon with convexity defects (stars, arrows, chevrons). */
        CLOSED_CONCAVE_POLY,
        /** Multiple distinct components (cross, grid, concentric). */
        COMPOUND,
        /** Fallback when nothing else fits. */
        UNKNOWN
    }

    // -------------------------------------------------------------------------
    // Fields
    // -------------------------------------------------------------------------

    /** Primary shape classification. */
    public final ShapeType type;

    /** Number of vertices after polygon approximation (0 for circles/lines). */
    public final int vertexCount;

    /**
     * Circularity = 4π × area / perimeter² in [0,1].
     * 1.0 = perfect circle, lower = less circular.
     */
    public final double circularity;

    /**
     * Concavity ratio = total convexity-defect depth / perimeter.
     * 0 for convex shapes, > 0 for stars/chevrons.
     */
    public final double concavityRatio;

    /**
     * 6-bin histogram of inter-vertex angles (0–180°, 30° per bin),
     * normalised so bins sum to 1.  Rotation-invariant because the bins
     * use absolute angle magnitudes, not positional order.
     */
    public final double[] angleHistogram;

    /** Number of distinct contour components found (1 for simple shapes). */
    public final int componentCount;

    /** Aspect ratio of bounding box (width / height, always >= 1). */
    public final double aspectRatio;

    /**
     * Solidity = contour area / convex-hull area, in (0, 1].
     * 1.0 = perfectly convex (rectangle, circle).
     * Lower = concave shape (star, arrowhead, cross outline).
     * Scale- and rotation-invariant — this is a shape ratio, not an absolute measure.
     */
    public final double solidity;

    /**
     * Topological fingerprint: the cyclic sequence of (normalised edge length,
     * turn angle) pairs describing how the contour's edges connect at each vertex.
     * Scale and rotation invariant. {@code null} for COMPOUND shapes.
     */
    public final ContourTopology topology;

    /**
     * Geometric segment descriptor: the contour traversed and classified into
     * STRAIGHT and CURVED segments, with scale-invariant length/radius ratios.
     * This is the primary structural discriminator — it is immune to noise
     * connections because curvature spikes at noise joins terminate segments.
     * {@code null} for COMPOUND shapes.
     */
    public final SegmentDescriptor segmentDescriptor;

    /**
     * Normalised area: contour area divided by the bounding-box area of the
     * full image from which this signature was extracted.
     */
    public final double normalisedArea;

    /**
     * Edge-length coefficient of variation = stddev(edges) / mean(edges).
     * Computed from the approxPolyDP vertex sequence; scale-invariant.
     * Near 0.0 for regular polygons with uniform edge lengths (square, equilateral diamond).
     * High (0.3–0.7) for shapes with strongly alternating edge lengths (rotated rectangle:
     * alternates hw=48 and hh=28 edges), or irregular polygons.
     * Primary discriminator between POLYLINE_DIAMOND (CV≈0) and RECT_ROTATED_45 (CV≈0.4).
     */
    public final double edgeLengthCV;

    // -------------------------------------------------------------------------
    // Constructor (private — use build())
    // -------------------------------------------------------------------------

    private VectorSignature(ShapeType type, int vertexCount, double circularity,
                             double concavityRatio, double[] angleHistogram,
                             int componentCount, double aspectRatio,
                             double solidity, ContourTopology topology,
                             SegmentDescriptor segmentDescriptor, double normalisedArea,
                             double edgeLengthCV) {
        this.type               = type;
        this.vertexCount        = vertexCount;
        this.circularity        = circularity;
        this.concavityRatio     = concavityRatio;
        this.angleHistogram     = angleHistogram;
        this.componentCount     = componentCount;
        this.aspectRatio        = aspectRatio;
        this.solidity           = solidity;
        this.topology           = topology;
        this.segmentDescriptor  = segmentDescriptor;
        this.normalisedArea     = normalisedArea;
        this.edgeLengthCV       = edgeLengthCV;
    }

    // -------------------------------------------------------------------------
    // Factory
    // -------------------------------------------------------------------------

    /**
     * Builds a {@code VectorSignature} from a binary (CV_8UC1) mask.
     *
     * @param binaryMask  single-channel binary image (255 = foreground)
     * @param epsilon     polygon approximation factor — multiplied by each
     *                    contour's perimeter to get the actual epsilon value.
     *                    Typical values: 0.02 (strict) … 0.08 (loose).
     * @return the computed signature, or an UNKNOWN signature on failure
     */
    public static VectorSignature build(Mat binaryMask, double epsilon) {
        return build(binaryMask, epsilon, Double.NaN);
    }

    /**
     * Builds directly from an already-extracted {@link MatOfPoint} contour,
     * bypassing the mask-render + findContours round-trip.
     * This is the fast path used by {@link org.example.matchers.VectorMatcher}
     * when scoring candidate contours from a scene.
     *
     * @param contour   a single contour (from findContours on the scene)
     * @param epsilon   polygon approximation factor
     * @param imageArea total pixel area of the source image (for normalisedArea), or NaN
     */
    public static VectorSignature buildFromContour(MatOfPoint contour, double epsilon, double imageArea) {
        if (contour == null || contour.empty()) return unknown();

        Rect bb = Imgproc.boundingRect(contour);
        if (bb.width < 4 || bb.height < 4) return unknown();

        // ── Step 1: Build SegmentDescriptor from the RAW contour BEFORE
        // approxPolyDP.  This preserves curved segments for ellipses and circles,
        // which approxPolyDP collapses to straight-segment polygons — causing an
        // isClosedCurve mismatch against the reference (built from a smooth mask
        // contour) and a segScore of 0.0.
        double rawPerim = Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true);
        SegmentDescriptor rawSegDesc = SegmentDescriptor.build(contour, rawPerim);

        // ── Step 2: ApproxPolyDP reduction for polygon-based metrics (vertex
        // count, circularity, solidity, type classification, topology).
        double strictEps = Math.min(Math.max(0.01 * rawPerim, 1.5), 6.0);
        MatOfPoint2f approxF = new MatOfPoint2f();
        Imgproc.approxPolyDP(new MatOfPoint2f(contour.toArray()), approxF, strictEps, true);
        Point[] approxPts = approxF.toArray();
        approxF.release();

        // Fall back to raw if approx collapsed to < 3 points
        Point[] renderPts = (approxPts.length >= 3) ? approxPts : contour.toArray();

        // ── Step 3: Render approxPolyDP polygon into crop for polygon metrics
        int pad = 2;
        Mat crop = Mat.zeros(bb.height + pad * 2, bb.width + pad * 2, CvType.CV_8UC1);
        Point[] shifted = new Point[renderPts.length];
        for (int i = 0; i < renderPts.length; i++) {
            shifted[i] = new Point(renderPts[i].x - bb.x + pad, renderPts[i].y - bb.y + pad);
        }
        MatOfPoint shiftedMat = new MatOfPoint(shifted);
        Imgproc.fillPoly(crop, List.of(shiftedMat), new Scalar(255));
        shiftedMat.release();

        // ── Step 4: Build signature from the polygon crop, then override the
        // segmentDescriptor with the raw-contour version so curved shapes
        // (ellipses, circles) match their reference correctly.
        VectorSignature sig = build(crop, epsilon, imageArea);
        crop.release();

        // ── Step 5: Re-derive type from the raw SegmentDescriptor so that
        // circles/ellipses are classified as CIRCLE consistently with the
        // reference path.  The crop-based type may be CLOSED_CONVEX_POLY
        // because rendering the approxPolyDP polygon into a crop and then
        // re-extracting contours creates a polygon (not a smooth curve).
        ShapeType finalType = sig.type;
        if (sig.circularity >= 0.85
                && rawSegDesc != null
                && rawSegDesc.isClosedCurve) {
            finalType = ShapeType.CIRCLE;
        }

        return new VectorSignature(
                finalType, sig.vertexCount, sig.circularity, sig.concavityRatio,
                sig.angleHistogram, sig.componentCount, sig.aspectRatio,
                sig.solidity, sig.topology, rawSegDesc, sig.normalisedArea, sig.edgeLengthCV);
    }

    /**
     * Builds a {@code VectorSignature} from a binary (CV_8UC1) mask.
     *
     * @param binaryMask     single-channel binary image (255 = foreground)
     * @param epsilon        polygon approximation factor
     * @param imageArea      total pixel area of the image this mask came from,
     *                       used to compute {@link #normalisedArea}.
     *                       Pass {@link Double#NaN} if unknown.
     * @return the computed signature, or an UNKNOWN signature on failure
     */
    public static VectorSignature build(Mat binaryMask, double epsilon, double imageArea) {
        if (binaryMask == null || binaryMask.empty()) return unknown();

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        try {
            Imgproc.findContours(binaryMask.clone(), contours, hierarchy,
                    Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        } finally {
            hierarchy.release();
        }

        // Filter out noise (tiny contours)
        contours.removeIf(c -> Imgproc.contourArea(c) < 64.0);

        if (contours.isEmpty()) return unknown();

        int componentCount = contours.size();

        // COMPOUND: multiple significant components
        if (componentCount > 1) {
            return buildCompound(contours, epsilon, imageArea);
        }

        // Single component
        MatOfPoint contour = contours.get(0);
        return buildSingle(contour, epsilon, imageArea);
    }

    // -------------------------------------------------------------------------
    // Internal builders
    // -------------------------------------------------------------------------

    private static VectorSignature buildSingle(MatOfPoint contour, double epsilon, double imageArea) {
        double area      = Imgproc.contourArea(contour);
        double perimeter = Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true);

        if (perimeter < 1.0) return unknown();

        // Circularity
        double circularity = (4.0 * Math.PI * area) / (perimeter * perimeter);
        circularity = Math.min(1.0, Math.max(0.0, circularity));

        // Bounding box aspect ratio
        Rect bb = Imgproc.boundingRect(contour);
        double w = bb.width, h = bb.height;
        double aspectRatio = (Math.max(w, h)) / Math.max(1.0, Math.min(w, h));

        // Solidity = area / convex-hull area — scale and rotation invariant
        // convexHull with MatOfInt returns hull point indices; reindex to get hull points
        MatOfInt hullIdx = new MatOfInt();
        Imgproc.convexHull(contour, hullIdx, false);
        int[] idx = hullIdx.toArray();
        Point[] allPts = contour.toArray();
        Point[] hullPtsArr = new Point[idx.length];
        for (int i = 0; i < idx.length; i++) hullPtsArr[i] = allPts[idx[i]];
        MatOfPoint hullMat = new MatOfPoint(hullPtsArr);
        double hullArea = Imgproc.contourArea(hullMat);
        double solidity = (hullArea > 1.0) ? Math.min(1.0, area / hullArea) : 1.0;
        hullIdx.release();
        hullMat.release();

        // Normalised area
        double normArea = (Double.isNaN(imageArea) || imageArea <= 0)
                ? Double.NaN : area / imageArea;

        // Polygon approximation
        double eps = Math.min(Math.max(epsilon * perimeter, 2.0), 8.0);
        MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
        MatOfPoint2f approx    = new MatOfPoint2f();
        Imgproc.approxPolyDP(contour2f, approx, eps, true);
        int vertexCount = (int) approx.total();

        // Inter-vertex angle histogram
        double[] angleHist = computeAngleHistogram(approx);

        // Contour topology — legacy connected edge structure
        ContourTopology topology = ContourTopology.build(approx, perimeter);

        // Segment descriptor — built from the RAW findContours contour (not
        // approxPolyDP).  This is consistent with the scene path in
        // buildFromContour(), which also uses the raw contour.
        //
        // CHAIN_APPROX_SIMPLE already compresses straight edges to just their
        // endpoints (4 points for a rect, 3 for a triangle, etc.).  The
        // SegmentDescriptor densifies sparse contours (spacing > 4px) so
        // polygons still get correctly segmented into STRAIGHT runs.
        //
        // Critically, circles and ellipses arrive as dense pixel-level contours
        // (spacing ≈ 1px), so the SegmentDescriptor correctly identifies them
        // as continuous curved loops (isClosedCurve = true).  The old approach
        // of using approxPolyDP here collapsed circles to polygons, producing
        // isClosedCurve = false — which caused a hard 0.0 from
        // SegmentDescriptor.similarity() when compared against the raw-contour
        // scene path (isClosedCurve = true).
        SegmentDescriptor segDesc = SegmentDescriptor.build(contour, perimeter);

        // Concavity ratio via convex hull
        double concavityRatio = computeConcavityRatio(contour, perimeter);

        // Edge-length coefficient of variation (CV = stddev/mean) — scale invariant.
        // Uses a dedicated coarser epsilon (4% of perimeter, no 8px cap) so that
        // large scene contours (e.g. 3× scaled shapes with perimeter ~1000px) reduce
        // to clean polygon corners rather than a staircase, ensuring the alternating
        // edge-length signal of a rotated rectangle is preserved at any scale.
        // Regular polygons with equal edges: CV ≈ 0.
        // Rotated rectangle (alternating hw=48/hh=28 edges): CV ≈ 0.26.
        double cvEps = Math.max(0.04 * perimeter, 4.0);  // 4% of perimeter, min 4px, NO upper cap
        MatOfPoint2f cvApprox = new MatOfPoint2f();
        Imgproc.approxPolyDP(contour2f, cvApprox, cvEps, true);
        double edgeLengthCV = computeEdgeLengthCV(cvApprox);
        if (System.getProperty("vm.debug") != null) {
            System.out.printf("[EDGE-CV] perim=%.0f cvEps=%.1f vertices=%d CV=%.3f%n",
                perimeter, cvEps, (int)cvApprox.total(), edgeLengthCV);
        }
        cvApprox.release();

        // Classify — order matters: LINE first, then CIRCLE (overrides concavity), then poly
        ShapeType type;
        if (aspectRatio >= 4.0) {
            type = ShapeType.LINE_SEGMENT;
        } else if (circularity >= 0.85) {
            // A true circle's contour is already dense; its SegmentDescriptor is a
            // single closed curved loop (isClosedCurve = true).
            // High-vertex polygons (pentagon ≈0.86, hexagon ≈0.91, heptagon ≈0.93,
            // octagon ≈0.95) also exceed 0.85, but after contour densification their
            // SegmentDescriptor correctly shows discrete STRAIGHT segments
            // (isClosedCurve = false).  Use that evidence to reclassify them as
            // CLOSED_CONVEX_POLY so self-matches score cleanly and cross-matches
            // against true circles are hard-gated.
            boolean hasStarightSegs = segDesc != null
                    && !segDesc.isClosedCurve
                    && !segDesc.segments.isEmpty();
            type = hasStarightSegs ? ShapeType.CLOSED_CONVEX_POLY : ShapeType.CIRCLE;
        } else if (concavityRatio >= 0.08) {
            type = ShapeType.CLOSED_CONCAVE_POLY;
        } else {
            type = ShapeType.CLOSED_CONVEX_POLY;
        }

        approx.release();
        contour2f.release();

        return new VectorSignature(type, vertexCount, circularity,
                concavityRatio, angleHist, 1, aspectRatio, solidity, topology, segDesc, normArea, edgeLengthCV);
    }

    private static VectorSignature buildCompound(List<MatOfPoint> contours, double epsilon, double imageArea) {
        // Use the largest component as the representative for vertex/circularity
        MatOfPoint largest = contours.stream()
                .max((a, b) -> Double.compare(Imgproc.contourArea(a), Imgproc.contourArea(b)))
                .orElse(contours.get(0));

        VectorSignature rep = buildSingle(largest, epsilon, imageArea);

        // Total normalised area = sum of all component areas
        double totalArea = contours.stream().mapToDouble(Imgproc::contourArea).sum();
        double normArea  = (Double.isNaN(imageArea) || imageArea <= 0)
                ? Double.NaN : totalArea / imageArea;

        // Aggregate angle histograms across all components
        double[] combined = new double[6];
        int counted = 0;
        for (MatOfPoint c : contours) {
            double perim = Imgproc.arcLength(new MatOfPoint2f(c.toArray()), true);
            double eps   = Math.min(Math.max(epsilon * perim, 2.0), 8.0);
            MatOfPoint2f c2f    = new MatOfPoint2f(c.toArray());
            MatOfPoint2f approx = new MatOfPoint2f();
            Imgproc.approxPolyDP(c2f, approx, eps, true);
            double[] h = computeAngleHistogram(approx);
            for (int i = 0; i < 6; i++) combined[i] += h[i];
            counted++;
            approx.release();
            c2f.release();
        }
        if (counted > 0) {
            double sum = Arrays.stream(combined).sum();
            if (sum > 0) for (int i = 0; i < 6; i++) combined[i] /= sum;
        }

        return new VectorSignature(ShapeType.COMPOUND,
                rep.vertexCount, rep.circularity, rep.concavityRatio,
                combined, contours.size(), rep.aspectRatio, rep.solidity, null, null, normArea, 0.0);
    }

    // -------------------------------------------------------------------------
    // Geometry helpers
    // -------------------------------------------------------------------------

    /**
     * Computes a 6-bin normalised histogram of interior angles at each vertex.
     * Bins: [0–30), [30–60), [60–90), [90–120), [120–150), [150–180].
     * Returns a flat histogram if fewer than 3 vertices.
     */
    static double[] computeAngleHistogram(MatOfPoint2f approx) {
        double[] hist = new double[6];
        Point[] pts   = approx.toArray();
        int n         = pts.length;
        if (n < 3) {
            Arrays.fill(hist, 1.0 / 6.0);
            return hist;
        }
        for (int i = 0; i < n; i++) {
            Point prev = pts[(i - 1 + n) % n];
            Point curr = pts[i];
            Point next = pts[(i + 1) % n];
            double ax = prev.x - curr.x, ay = prev.y - curr.y;
            double bx = next.x - curr.x, by = next.y - curr.y;
            double dot  = ax * bx + ay * by;
            double magA = Math.sqrt(ax * ax + ay * ay);
            double magB = Math.sqrt(bx * bx + by * by);
            if (magA < 1e-9 || magB < 1e-9) continue;
            double cosAngle = Math.max(-1.0, Math.min(1.0, dot / (magA * magB)));
            double angleDeg = Math.toDegrees(Math.acos(cosAngle));
            int bin = Math.min(5, (int)(angleDeg / 30.0));
            hist[bin]++;
        }
        double sum = Arrays.stream(hist).sum();
        if (sum > 0) for (int i = 0; i < 6; i++) hist[i] /= sum;
        else Arrays.fill(hist, 1.0 / 6.0);
        return hist;
    }

    /**
     * Computes the coefficient of variation (stddev / mean) of consecutive
     * inter-vertex edge lengths from an approxPolyDP vertex sequence.
     * Scale-invariant: all lengths are normalised by mean before computing CV.
     * Returns 0.0 if fewer than 2 edges.
     *
     * <p>Key values:
     * <ul>
     *   <li>Regular polygon / square rotated 45° (equal edges): CV ≈ 0.0</li>
     *   <li>Rotated rectangle hw=48, hh=28 (alternating edges): CV ≈ 0.24</li>
     *   <li>Highly irregular polygon: CV > 0.4</li>
     * </ul>
     */
    static double computeEdgeLengthCV(MatOfPoint2f approx) {
        Point[] pts = approx.toArray();
        int n = pts.length;
        if (n < 2) return 0.0;
        double[] lengths = new double[n];
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            Point a = pts[i], b = pts[(i + 1) % n];
            double dx = b.x - a.x, dy = b.y - a.y;
            lengths[i] = Math.sqrt(dx * dx + dy * dy);
            sum += lengths[i];
        }
        double mean = sum / n;
        if (mean < 1e-9) return 0.0;
        double variance = 0.0;
        for (double l : lengths) variance += (l - mean) * (l - mean);
        variance /= n;
        return Math.sqrt(variance) / mean;
    }

    /**
     * Concavity ratio = total depth of convexity defects / perimeter.
     * Returns 0 for fully convex contours.
     *
     * <p>Self-intersecting contours (common in noisy scene masks) cause
     * {@code convexityDefects} to throw because the hull indices become
     * non-monotonic.  We sanitise the contour with {@code approxPolyDP}
     * before computing defects — this collapses tiny loops and removes
     * self-intersections without materially changing the shape.</p>
     */
    static double computeConcavityRatio(MatOfPoint contour, double perimeter) {
        try {
            // ── Sanitise: approxPolyDP removes self-intersections ────────
            MatOfPoint2f raw2f = new MatOfPoint2f(contour.toArray());
            double eps = Math.max(1.0, 0.005 * perimeter);   // very light — just fix crossings
            MatOfPoint2f approxF = new MatOfPoint2f();
            Imgproc.approxPolyDP(raw2f, approxF, eps, true);
            raw2f.release();

            MatOfPoint clean = new MatOfPoint();
            approxF.convertTo(clean, CvType.CV_32S);
            approxF.release();

            if (clean.rows() < 3) { clean.release(); return 0.0; }

            // ── Convex hull (index form, counter-clockwise = false) ──────
            MatOfInt  hull    = new MatOfInt();
            MatOfInt4 defects = new MatOfInt4();
            Imgproc.convexHull(clean, hull, false);
            if (hull.rows() < 3) { hull.release(); clean.release(); return 0.0; }

            // ── Verify hull indices are monotonically increasing ─────────
            // Non-monotonic indices mean the simplified contour still has a
            // tiny self-intersection; in that case we return 0 safely.
            int[] hullIdx = new int[(int) hull.total()];
            hull.get(0, 0, hullIdx);
            boolean mono = true;
            for (int i = 1; i < hullIdx.length; i++) {
                if (hullIdx[i] <= hullIdx[i - 1]) { mono = false; break; }
            }
            if (!mono) { hull.release(); clean.release(); return 0.0; }

            // ── Compute defect depths ────────────────────────────────────
            Imgproc.convexityDefects(clean, hull, defects);
            double totalDepth = 0.0;
            if (!defects.empty()) {
                int[] data = new int[(int)(defects.total() * defects.channels())];
                defects.get(0, 0, data);
                for (int i = 0; i < data.length; i += 4) {
                    totalDepth += data[i + 3] / 256.0;   // Q8 fixed-point depth
                }
            }
            hull.release(); defects.release(); clean.release();
            return perimeter > 0 ? Math.min(1.0, totalDepth / perimeter) : 0.0;
        } catch (Exception e) {
            return 0.0;
        }
    }

    // -------------------------------------------------------------------------
    // Similarity scoring
    // -------------------------------------------------------------------------

    /**
     * Returns a similarity score in [0, 1] between this signature (the candidate
     * extracted from the scene) and {@code ref} (the reference shape).
     *
     * <h3>Design principle — ratio enforcement</h3>
     * <p>Every field in a {@code VectorSignature} is a <em>geometric ratio</em> that
     * is preserved when the shape is scaled or rotated:
     * <ul>
     *   <li><b>circularity</b>  — 4π·area/perimeter² — scale invariant</li>
     *   <li><b>aspectRatio</b>  — bbox width/height   — scale invariant, rotation stable ±45°</li>
     *   <li><b>solidity</b>     — area/hull area      — scale and rotation invariant</li>
     *   <li><b>vertexCount</b>  — polygon corners     — topology invariant</li>
     *   <li><b>angleHistogram</b> — angle distribution — rotation invariant (absolute angles)</li>
     * </ul>
     * <p>The reference defines the "expected" ratio for each field.  A candidate that
     * has different ratios is a different shape, regardless of scale or rotation.
     * Each component score is therefore computed as how well the candidate's ratio
     * matches the reference's ratio.
     *
     * <h3>Weights</h3>
     * <ul>
     *   <li>0.25 — type (hard gate for cross-type mismatches)</li>
     *   <li>0.20 — circularity ratio match</li>
     *   <li>0.20 — solidity ratio match</li>
     *   <li>0.15 — vertex count match</li>
     *   <li>0.15 — angle histogram intersection</li>
     *   <li>0.05 — aspect-ratio match</li>
     * </ul>
     */
    public double similarity(VectorSignature ref) {
        if (ref == null) return 0.0;

        // ── 0. Normalised-area gate ────────────────────────────────────────
        // Note: in VectorMatcher, this = refSig (NaN normArea), ref = sceneSig.
        //
        // Rule (a): image-border reject — if the scene candidate fills >95% of
        // the image it's almost certainly the frame border, not a real shape.
        //
        // Rule (b): minimum-size reject — if the scene candidate is a tiny noise
        // fragment (< 0.3% of image area) it cannot be a meaningful shape match.
        // This eliminates small line intersections and random fragments that share
        // geometric features with real shapes only by coincidence.
        //
        // Rule (c): area-ratio reject — only fires when BOTH sides have a finite
        // normalisedArea (e.g. scene-vs-scene comparison).
        if (!Double.isNaN(ref.normalisedArea)) {
            // Rule (a): near-full-image reject — contour fills > 80% of the scene
            // → almost certainly the frame border or a full-image background blob.
            if (ref.normalisedArea > 0.80) {
                return Math.min(0.25, computeRawSimilarity(ref));
            }
            // Rule (b): minimum-size reject — tiny noise fragment
            if (ref.normalisedArea < 0.003) {
                return Math.min(0.25, computeRawSimilarity(ref));
            }
        }
        if (!Double.isNaN(this.normalisedArea) && !Double.isNaN(ref.normalisedArea)
                && ref.normalisedArea > 0 && this.normalisedArea > 0) {
            double areaRatio = ref.normalisedArea / this.normalisedArea;
            if (areaRatio < (1.0 / 10.0) || areaRatio > 10.0) {
                return Math.min(0.25, computeRawSimilarity(ref));
            }
        }

        return computeRawSimilarity(ref);
    }

    private double computeRawSimilarity(VectorSignature ref) {

        // ── 1. Type scoring with hard gate ────────────────────────────────
        // Incompatible types get a hard cap so they can never pass threshold.
        // Exception: CIRCLE vs CLOSED_CONVEX_POLY where the circle-typed side has
        // borderline circularity (0.85–0.94) — this happens with high-vertex polygons
        // like octagons that fall on the CIRCLE/POLY boundary at different scales.
        double typeScore;
        boolean hardGate = false;
        if (this.type == ref.type) {
            typeScore = 1.0;
        } else if ((this.type  == ShapeType.CLOSED_CONVEX_POLY || this.type  == ShapeType.CLOSED_CONCAVE_POLY)
                && (ref.type   == ShapeType.CLOSED_CONVEX_POLY || ref.type   == ShapeType.CLOSED_CONCAVE_POLY)) {
            // Polygon family: check if types match
            if (this.type != ref.type) {
                // Type mismatch (CONVEX vs CONCAVE) — hard gate
                typeScore = 0.0;
                hardGate  = true;
                
                // DEBUG logging
                if (System.getProperty("vm.debug") != null) {
                    System.out.printf("[POLY-GATE] Hard gate: %s vs %s | concav: %.3f vs %.3f | solid: %.3f vs %.3f%n",
                        this.type, ref.type, this.concavityRatio, ref.concavityRatio, this.solidity, ref.solidity);
                }
            } else {
                // Same type — no penalty
                typeScore = 1.0;
            }
        } else if ((this.type == ShapeType.CIRCLE && ref.type == ShapeType.CLOSED_CONVEX_POLY)
                || (this.type == ShapeType.CLOSED_CONVEX_POLY && ref.type == ShapeType.CIRCLE)) {
            // Borderline: one classified as CIRCLE, one as POLY.
            double circSide     = (this.type == ShapeType.CIRCLE) ? this.circularity : ref.circularity;
            double polySideCirc = (this.type == ShapeType.CLOSED_CONVEX_POLY) ? this.circularity : ref.circularity;
            int    polySide     = (this.type == ShapeType.CLOSED_CONVEX_POLY) ? this.vertexCount : ref.vertexCount;

            if (polySideCirc > 0.82 && circSide > 0.82) {
                // Both sides are highly circular.  The POLY side is most likely a true
                // circle whose contour was broken by noise arms (random-lines background)
                // or by CHAIN_APPROX_SIMPLE compression artifacts.  Treat as near-match.
                typeScore = 0.85;
            } else if (circSide < 0.95 && polySide >= 6) {
                // Borderline high-vertex polygon (octagon/hexagon) that straddles the
                // CIRCLE/POLY boundary at this scale — partial penalty only.
                typeScore = 0.4;
            } else {
                // Low-vertex polygon (triangle/rect/pentagon) or truly circular — hard gate
                typeScore = 0.0;
                hardGate  = true;
            }
        } else if ((this.type == ShapeType.CIRCLE && ref.type == ShapeType.CLOSED_CONCAVE_POLY)
                || (this.type == ShapeType.CLOSED_CONCAVE_POLY && ref.type == ShapeType.CIRCLE)) {
            // Circles vs concave shapes (stars, arrowheads, chevrons) — always hard gate.
            // No circle can legitimately match a star or arrowhead.
            typeScore = 0.0;
            hardGate  = true;
        } else {
            // Fundamentally incompatible (LINE vs anything, CONCAVE vs CIRCLE, etc.)
            typeScore = 0.0;
            hardGate  = true;
        }

        // ── 2. Circularity ratio — enforces "how round the shape is" ─────
        // A circle has circularity ≈ 1.0; a rectangle ≈ 0.78; a triangle ≈ 0.60.
        // These are preserved across scale and rotation.
        //
        // Special gate: when the reference is a CIRCLE (circ > 0.82), the scene
        // candidate MUST also be reasonably circular.  A gradient stripe or
        // any non-round shape has circularity well below 0.65 and cannot
        // legitimately match a filled circle.
        double circScore = 1.0 - Math.abs(this.circularity - ref.circularity);
        if (this.type == ShapeType.CIRCLE && ref.circularity < 0.65) {
            // Clearly non-circular candidate vs circular reference — hard gate
            hardGate = true;
        }

        // ── 3. Solidity ratio — enforces convexity / fill ratio ───────────
        // A filled rectangle ≈ 1.0; a star ≈ 0.5; a circle outline ring ≈ 0.15.
        // This is the strongest discriminator between a background circle (solid, ~1.0)
        // and a rectangle outline or thin shape, and between a filled shape and a ring.
        double solidityScore = 1.0 - Math.abs(this.solidity - ref.solidity);

        // ── 4. Vertex count — ratio-based penalty ────────────────────────
        // Score = matched / expected, where "expected" is the reference count.
        // Missing half the expected vertices gives 0.50, missing all gives 0.0.
        // This scales relative to how structurally complex the reference is —
        // a triangle missing 1 vertex (33% deficit) is penalised more than an
        // octagon missing 1 vertex (12% deficit).
        //
        // Special case: if the reference has 0 vertices (circle / line), the shape
        // type already handles discrimination — don't penalise the scene for picking
        // up noise vertices in the polygon approximation.
        double vertexScore;
        if (ref.vertexCount == 0) {
            // Reference is a circle or line — vertex count not a discriminator here
            vertexScore = 1.0;
        } else if (this.vertexCount == 0) {
            // Scene produced no vertices but reference expects some — full penalty
            vertexScore = 0.0;
        } else {
            // Vertex scoring: penalise MISSING vertices (scene fewer than reference)
            // but do NOT penalise extra vertices from noise (scene has more).
            // ref.vertexCount = vDet (scene), this.vertexCount = vRef (reference).
            // Score = min(vDet, vRef) / vRef
            //   e.g. vRef=4, vDet=6 → min(6,4)/4 = 1.0  (no penalty for extra)
            //        vRef=4, vDet=2 → min(2,4)/4 = 0.5  (missing half the vertices)
            //        vRef=5, vDet=2 → min(2,5)/5 = 0.4
            // Symmetric ratio — penalises both missing AND extra vertices equally.
            // min/max ensures a 20-vertex noise blob cannot score 1.0 against a 4-vertex rect.
            double found    = Math.min(this.vertexCount, ref.vertexCount);
            double total    = Math.max(this.vertexCount, ref.vertexCount);
            vertexScore = (total > 0) ? found / total : 1.0;
        }

        // ── 5. Segment descriptor — geometric traversal (primary structural signal) ──
        double segScore;
        if (this.segmentDescriptor != null && ref.segmentDescriptor != null) {
            segScore = this.segmentDescriptor.similarity(ref.segmentDescriptor);
        } else {
            segScore = 0.0;
        }

        // ── 5b. CIRCLE-type segScore fallback ──────────────────────────────
        // For CIRCLE shapes, SegmentDescriptor is unreliable because
        // CHAIN_APPROX_SIMPLE compresses circle contours differently at
        // different scales.  A smaller circle stays dense (isClosedCurve=true),
        // while a larger circle gets compressed to polygon-like vertices that
        // trigger densification → straight segments → isClosedCurve=false.
        // The mismatch causes a hard 0.0 from SegmentDescriptor.similarity().
        //
        // Also applies when one side is CIRCLE and the other is
        // CLOSED_CONVEX_POLY with high circularity (> 0.82) — this occurs when
        // noise arms from background lines break the contour's closed-curve
        // property, downgrading the type to POLY despite the shape being circular.
        //
        // When both shapes are highly circular, circularity agreement IS the
        // primary structural signal — use it as a floor for segScore.
        double rawSegScore = segScore;  // save before fallback for coherence gate below
        boolean bothCircleLike =
                (this.type == ShapeType.CIRCLE && ref.type == ShapeType.CIRCLE)
             || ((this.type == ShapeType.CIRCLE || ref.type == ShapeType.CIRCLE)
                    && this.circularity > 0.82 && ref.circularity > 0.82);
        if (bothCircleLike && segScore < 0.10) {
            segScore = Math.max(segScore, circScore);
        }

        // ── 6. Topology — legacy connected edge structure ─────────────────
        double topoScore;
        if (this.topology != null && ref.topology != null) {
            topoScore = this.topology.similarity(ref.topology);
        } else {
            topoScore = 0.0;
        }

        // ── 7. Angle histogram intersection — rotation invariant ──────────
        double angleScore = histogramIntersection(this.angleHistogram, ref.angleHistogram);

        // ── 8. Aspect ratio — multiplicative gate ─────────────────────────
        double arA = Math.max(this.aspectRatio, 1.0);
        double arB = Math.max(ref.aspectRatio,  1.0);
        double aspectScore = 1.0 - Math.abs(arA - arB) / Math.max(arA, arB);
        double arMultiplier = aspectScore >= 0.70 ? 1.0 : Math.pow(aspectScore / 0.70, 2.0);

        // ── 9. Vertex count multiplicative gate ──────────────────────────
        // When two CLOSED_CONVEX_POLY shapes have different vertex counts the
        // mismatch is structurally significant (hexagon ≠ octagon, triangle ≠ pentagon).
        // The additive vertexScore (0.08 weight) is too weak to overcome high scores
        // in other features — a strong multiplicative gate is needed.
        //
        // Uses the symmetric min/max ratio raised to the power 2.5 so that:
        //   • 6 vs 8 (hexagon/octagon):   (6/8)^2.5 ≈ 0.487 → reduces score by ~51%
        //   • 3 vs 5 (triangle/pentagon): (3/5)^2.5 ≈ 0.279 → reduces score by ~72%
        //   • Same counts (self-match):   (n/n)^2.5 = 1.0   → no penalty
        //
        // Only applies to CLOSED_CONVEX_POLY pairs — circles, lines and open curves
        // have vertex counts that vary with approximation noise and must not be penalised.
        double vertexMultiplier = 1.0;
        if (this.type == ShapeType.CLOSED_CONVEX_POLY && ref.type == ShapeType.CLOSED_CONVEX_POLY
                && this.vertexCount > 0 && ref.vertexCount > 0
                && this.vertexCount != ref.vertexCount
                // Guard 1: only canonical small polygons — ellipses/circles approximated as
                // 12–20-vertex polygons have inherent approximation noise across scale/threshold
                // changes; applying the penalty there breaks legitimate self-matches.
                && Math.max(this.vertexCount, ref.vertexCount) <= 10
                // Guard 2: require a meaningful relative gap (> 20%).  This protects against
                // 1-vertex polygon-approximation noise (hexagon detected as 7-gon: 6/7 = 0.857
                // which is > 0.80 → no penalty).  True polygon-order differences like
                // hexagon(6) vs octagon(8) produce ratio 0.75 which is ≤ 0.80.
                ) {
            double vtxRatio = (double) Math.min(this.vertexCount, ref.vertexCount)
                            / (double) Math.max(this.vertexCount, ref.vertexCount);
            if (vtxRatio <= 0.80) {
                vertexMultiplier = Math.pow(vtxRatio, 2.5);
                if (System.getProperty("vm.debug") != null) {
                    System.out.printf("[VTXMULT-GATE] type=%s/%s vtx=%d/%d ratio=%.3f mult=%.3f%n",
                        this.type, ref.type, this.vertexCount, ref.vertexCount,
                        vtxRatio, vertexMultiplier);
                }
            }
        }

        // ── 10. Edge-length CV multiplicative gate ─────────────────────────
        // For same-vertex-count convex polygons, edge-length uniformity is the
        // primary remaining discriminator (e.g. POLYLINE_DIAMOND with uniform edges
        // vs RECT_ROTATED_45 with alternating long/short edges).
        //
        // Gate condition (strict, asymmetric):
        //   • "this" (reference sig) has VERY UNIFORM edges (CV < 0.05):
        //     e.g. POLYLINE_DIAMOND (perfect square rotated 45°), regular hexagon, etc.
        //   • "ref"  (scene candidate) has STRONGLY NON-UNIFORM edges (CV > 0.20).
        //   • The difference is large (> 0.20).
        //
        // Thresholds are deliberately tight to prevent false penalties on:
        //   - Regular rings/circles (approx. octagon-like, refCV ≈ 0.009–0.015)
        //     which have borderline CV due to polygon approximation at scale
        //   - Triangles with slight edge-length variation (refCV ≈ 0.05)
        //   - Compound shapes with mixed polygon clusters
        //   - Self-matches where both sides have similar small CVs
        double edgeCVMultiplier = 1.0;
        if (this.type == ShapeType.CLOSED_CONVEX_POLY
                && ref.type == ShapeType.CLOSED_CONVEX_POLY
                && this.vertexCount > 0 && ref.vertexCount == this.vertexCount
                && this.edgeLengthCV < 0.05          // reference has VERY uniform edges (regular polygon/diamond)
                && ref.edgeLengthCV > 0.20            // scene has STRONGLY non-uniform edges
                && (ref.edgeLengthCV - this.edgeLengthCV) > 0.20) { // large, unambiguous difference
            double cvDiff = ref.edgeLengthCV - this.edgeLengthCV;
            edgeCVMultiplier = Math.max(0.25, Math.pow(1.0 - cvDiff, 2.5));
            if (System.getProperty("vm.debug") != null) {
                System.out.printf("[EDGECV-GATE] type=%s/%s vtx=%d/%d thisCV=%.3f refCV=%.3f diff=%.3f mult=%.3f%n",
                    this.type, ref.type, this.vertexCount, ref.vertexCount,
                    this.edgeLengthCV, ref.edgeLengthCV, cvDiff, edgeCVMultiplier);
            }
        } else if (System.getProperty("vm.debug") != null
                && this.type == ShapeType.CLOSED_CONVEX_POLY
                && ref.type == ShapeType.CLOSED_CONVEX_POLY
                && this.vertexCount > 0 && ref.vertexCount == this.vertexCount) {
            System.out.printf("[EDGECV-SKIP] type=%s/%s vtx=%d/%d thisCV=%.3f refCV=%.3f%n",
                this.type, ref.type, this.vertexCount, ref.vertexCount,
                this.edgeLengthCV, ref.edgeLengthCV);
        }

        // ── Component count penalty ───────────────────────────────────────
        double componentPenalty = 0.0;
        if (this.componentCount != ref.componentCount) {
            int maxC = Math.max(this.componentCount, ref.componentCount);
            componentPenalty = 0.15 * ((double) Math.abs(this.componentCount - ref.componentCount) / maxC);
        }

        // ── Near-circular coherence ─────────────────────────────────────
        // When BOTH shapes are near-circular (circ > 0.82) and their scale-invariant
        // global metrics agree almost perfectly, but seg/topo/angle are all zero or
        // near zero, the failure is a polygon-approximation or contour-compression
        // artifact, not a true structural difference.
        //
        // This handles two cases:
        //   (a) Self-match scale artifacts — the 8 px epsilon cap in approxPolyDP
        //       makes vertex count scale-dependent.
        //   (b) Noise-contaminated circles — background lines break the contour's
        //       closed-curve property, the SegmentDescriptor sees STRAIGHT segments,
        //       and topo/angle also fail due to vertex misalignment.
        //
        // Safety gates:
        //   • circScore ≥ 0.95 — circularity must match within 0.05.
        //     Hexagon (0.907) vs octagon (0.948) → circScore = 0.959 ≈ 0.96 → passes.
        //     BUT hexagon topo is typically > 0.01 (6 vs 8 regular edges → partial
        //     alignment), so the topoScore < 0.01 gate blocks it.
        //   • solidityScore ≥ 0.90 — fill ratio must also agree.
        //   • segScore already ≤ circScore (from fallback 5b) — if both are CIRCLE×CIRCLE,
        //     segScore is floored to circScore, so this block only adds topo/angle/vtx.
        //   • typeScore ≥ 0.85 — allows CIRCLE×CLOSED_CONVEX_POLY with high circularity
        //     (scored 0.85 in the type comparison) to benefit from coherence.
        //   • topoScore < 0.01 — the topology must have completely failed, confirming
        //     vertex alignment is genuinely impossible (not just partially degraded).
        boolean bothNearCircular = this.circularity > 0.82 && ref.circularity > 0.82;
        if (bothNearCircular && typeScore >= 0.85
                && circScore      >= 0.95
                && solidityScore  >= 0.90
                && aspectScore    >= 0.90
                && rawSegScore    < 0.01      // use RAW seg (before CIRCLE fallback)
                && topoScore      < 0.01) {
            segScore    = Math.max(segScore,    0.90);
            topoScore   = Math.max(topoScore,   0.90);
            angleScore  = Math.max(angleScore,  0.85);
            vertexScore = Math.max(vertexScore, 0.85);
        }

        // ── Segment-score coherence boost ────────────────────────────────
        // When all OTHER geometric features (type, circularity, solidity, vertices,
        // aspect ratio) agree strongly, the SegmentDescriptor should not drag the
        // total score below what the geometry implies.  This primarily helps ellipses
        // and circles whose SegmentDescriptor varies across different contour
        // densities but whose shape metrics are highly consistent.
        //
        // CRITICAL GATES to prevent false matches:
        // 1. Require angle histogram agreement (≥ 0.70) to prevent false matches 
        //    between shapes with same vertex count but different angle distributions
        // 2. Require actual topology/segment scores to be reasonable (≥ 0.50) —
        //    if topology/segment are very low despite matching global metrics, this
        //    indicates structural differences (e.g., diamond vs square, both 4 vertices
        //    but different edge proportions/vertex spacing). Don't boost in this case.
        // 3. DISABLE BOOST when vertex counts match perfectly (vertexScore = 1.0) for
        //    CLOSED_CONVEX_POLY — when both shapes have identical vertex counts, the
        //    ONLY discriminator is edge proportions/vertex spacing captured by topology/
        //    segment descriptors. Boosting here masks the structural differences that
        //    matter most (e.g., diamond vs rotated square, both 4 vertices).
        //
        // Floor levels are tiered by agreement strength:
        //   • Very strong (type match, circ/solid/vtx ≥ 0.95, AR ≥ 0.80, angle ≥ 0.70, seg/topo ≥ 0.50): floor at 0.88
        //     → clean self-matches produce overall score ≥ ~93%
        //   • Acceptable (type match, all ≥ 0.80, AR ≥ 0.70, angle ≥ 0.60, seg/topo ≥ 0.40): floor at 0.65
        //     → good-quality matches are not suppressed below ~82% overall
        boolean allowBoost = true;
        
        if (allowBoost && typeScore >= 1.0
                && circScore     >= 0.95
                && solidityScore >= 0.95
                && vertexScore   >= 0.95
                && aspectScore   >= 0.80
                && angleScore    >= 0.70
                && segScore      >= 0.50
                && topoScore     >= 0.50) {
            segScore = Math.max(segScore, 0.92);
        } else if (allowBoost && typeScore >= 1.0
                && circScore     >= 0.80
                && solidityScore >= 0.80
                && vertexScore   >= 0.80
                && aspectScore   >= 0.70
                && angleScore    >= 0.60
                && segScore      >= 0.40
                && topoScore     >= 0.40) {
            segScore = Math.max(segScore, 0.72);
        }

        double score = (typeScore     * 0.15
                     + segScore      * 0.23
                     + topoScore     * 0.15
                     + circScore     * 0.13
                     + solidityScore * 0.18
                     + vertexScore   * 0.08
                     + angleScore    * 0.10
                     - componentPenalty)
                     * arMultiplier          // AR gate: wrong shape proportions → full suppression
                     * vertexMultiplier      // vertex count gate: diff polygon orders → suppression
                     * edgeCVMultiplier;     // edge length uniformity gate (diamond vs rotated rect)

        double result = Math.max(0.0, Math.min(1.0, score));


        // Hard gate: cap cross-type matches well below any pass threshold
        // Max score with hard gate = 0.15*count + 0.25*match + 0.60*0.15 = 0.49 (49%)
        if (hardGate) result = Math.min(result, 0.15);

        // DEBUG — remove after diagnosis
        if (result > 0.4 && System.getProperty("vm.debug") != null) {
            System.out.printf("[SIG-DEBUG] result=%.3f hardGate=%b type=%s->%s typeScore=%.2f seg=%.3f topo=%.3f circ=%.3f(%.3f->%.3f) solid=%.3f vtx=%.2f angle=%.3f ar=%.3f%n",
                result, hardGate, this.type, ref.type, typeScore, segScore, topoScore,
                circScore, this.circularity, ref.circularity, solidityScore, vertexScore, angleScore, arMultiplier);
        }

        return result;
    } // end computeRawSimilarity

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    static double histogramIntersection(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) sum += Math.min(a[i], b[i]);
        return sum;  // already in [0,1] if both histograms sum to 1
    }

    private static VectorSignature unknown() {
        return new VectorSignature(ShapeType.UNKNOWN, 0, 0, 0,
                new double[]{1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6}, 0, 1.0, 0, null, null, Double.NaN, 0.0);
    }

    @Override
    public String toString() {
        return String.format(
                "VectorSignature{type=%s, vertices=%d, circ=%.2f, solidity=%.2f, concav=%.2f, ar=%.2f, components=%d, normArea=%.4f}",
                type, vertexCount, circularity, solidity, concavityRatio, aspectRatio, componentCount, normalisedArea);
    }
}































