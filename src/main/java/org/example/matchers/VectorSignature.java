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

    // -------------------------------------------------------------------------
    // Constructor (private — use build())
    // -------------------------------------------------------------------------

    private VectorSignature(ShapeType type, int vertexCount, double circularity,
                             double concavityRatio, double[] angleHistogram,
                             int componentCount, double aspectRatio,
                             double solidity, ContourTopology topology,
                             SegmentDescriptor segmentDescriptor, double normalisedArea) {
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

        // Pre-reduce with STRICT approxPolyDP before rendering into the crop.
        // This collapses the 7-8 pixel-stepping points per corner down to the
        // true geometric corners (e.g. 4 for a rect, 3 for a triangle).
        double perim     = Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true);
        double strictEps = Math.max(0.02 * perim, 2.0);
        MatOfPoint2f approxF = new MatOfPoint2f();
        Imgproc.approxPolyDP(new MatOfPoint2f(contour.toArray()), approxF, strictEps, true);
        Point[] approxPts = approxF.toArray();
        approxF.release();

        // Fall back to raw if approx collapsed to < 3 points
        Point[] renderPts = (approxPts.length >= 3) ? approxPts : contour.toArray();

        // Shift to crop-local coordinates
        int pad = 2;
        Mat crop = Mat.zeros(bb.height + pad * 2, bb.width + pad * 2, CvType.CV_8UC1);
        Point[] shifted = new Point[renderPts.length];
        for (int i = 0; i < renderPts.length; i++) {
            shifted[i] = new Point(renderPts[i].x - bb.x + pad, renderPts[i].y - bb.y + pad);
        }

        // Use fillPoly — handles non-simple polygons that approxPolyDP can produce,
        // unlike drawContours which throws a convexityDefects self-intersection error.
        MatOfPoint shiftedMat = new MatOfPoint(shifted);
        Imgproc.fillPoly(crop, List.of(shiftedMat), new Scalar(255));
        shiftedMat.release();

        VectorSignature sig = build(crop, epsilon, imageArea);
        crop.release();
        return sig;
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
        double eps = Math.max(epsilon * perimeter, 2.0);
        MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
        MatOfPoint2f approx    = new MatOfPoint2f();
        Imgproc.approxPolyDP(contour2f, approx, eps, true);
        int vertexCount = (int) approx.total();

        // Inter-vertex angle histogram
        double[] angleHist = computeAngleHistogram(approx);

        // Contour topology — legacy connected edge structure
        ContourTopology topology = ContourTopology.build(approx, perimeter);

        // Segment descriptor — built from the approxPolyDP-reduced contour so that
        // we start from clean corners (4 for a rect, 3 for a triangle) rather than
        // hundreds of pixel-level stepping points that confuse the traversal.
        // We use the STRICT epsilon (0.02) regardless of the variant epsilon so the
        // descriptor always sees the true geometric corners.
        double strictEps = Math.max(0.02 * perimeter, 2.0);
        MatOfPoint2f strictApprox = new MatOfPoint2f();
        Imgproc.approxPolyDP(contour2f, strictApprox, strictEps, true);
        MatOfPoint strictContour = new MatOfPoint(strictApprox.toArray());
        SegmentDescriptor segDesc = SegmentDescriptor.build(strictContour, perimeter);
        strictApprox.release();
        strictContour.release();

        // Concavity ratio via convex hull
        double concavityRatio = computeConcavityRatio(contour, perimeter);

        // Classify — order matters: LINE first, then CIRCLE (overrides concavity), then poly
        ShapeType type;
        if (aspectRatio >= 4.0) {
            type = ShapeType.LINE_SEGMENT;
        } else if (circularity >= 0.85) {
            type = ShapeType.CIRCLE;
        } else if (concavityRatio >= 0.08) {
            type = ShapeType.CLOSED_CONCAVE_POLY;
        } else {
            type = ShapeType.CLOSED_CONVEX_POLY;
        }

        approx.release();
        contour2f.release();

        return new VectorSignature(type, vertexCount, circularity,
                concavityRatio, angleHist, 1, aspectRatio, solidity, topology, segDesc, normArea);
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
            double eps   = Math.max(epsilon * perim, 2.0);
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
                combined, contours.size(), rep.aspectRatio, rep.solidity, null, null, normArea);
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
     * Concavity ratio = total depth of convexity defects / perimeter.
     * Returns 0 for fully convex contours.
     */
    static double computeConcavityRatio(MatOfPoint contour, double perimeter) {
        try {
            MatOfInt    hull    = new MatOfInt();
            MatOfInt4   defects = new MatOfInt4();
            MatOfPoint2f c2f   = new MatOfPoint2f(contour.toArray());
            Imgproc.convexHull(contour, hull, false);
            if (hull.rows() < 3) { hull.release(); c2f.release(); return 0.0; }
            Imgproc.convexityDefects(contour, hull, defects);
            double totalDepth = 0.0;
            if (!defects.empty()) {
                int[] data = new int[(int)(defects.total() * defects.channels())];
                defects.get(0, 0, data);
                for (int i = 0; i < data.length; i += 4) {
                    totalDepth += data[i + 3] / 256.0;  // depth is Q8 fixed-point
                }
            }
            hull.release(); defects.release(); c2f.release();
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
            // Same broad polygon family but concavity differs — moderate penalty
            typeScore = 0.5;
        } else if ((this.type == ShapeType.CIRCLE && ref.type == ShapeType.CLOSED_CONVEX_POLY)
                || (this.type == ShapeType.CLOSED_CONVEX_POLY && ref.type == ShapeType.CIRCLE)) {
            // Borderline: one classified as CIRCLE, one as POLY.
            // If the circular one has circ < 0.95 it's likely a high-vertex polygon (octagon etc).
            double circSide = (this.type == ShapeType.CIRCLE) ? this.circularity : ref.circularity;
            if (circSide < 0.95) {
                // Borderline polygon — partial penalty only, no hard gate
                typeScore = 0.4;
            } else {
                // Truly circular vs clearly polygonal — hard gate
                typeScore = 0.0;
                hardGate  = true;
            }
        } else {
            // Fundamentally incompatible (LINE vs anything, CONCAVE vs CIRCLE, etc.)
            typeScore = 0.0;
            hardGate  = true;
        }

        // ── 2. Circularity ratio — enforces "how round the shape is" ─────
        // A circle has circularity ≈ 1.0; a rectangle ≈ 0.78; a triangle ≈ 0.60.
        // These are preserved across scale and rotation.
        double circScore = 1.0 - Math.abs(this.circularity - ref.circularity);

        // ── 3. Solidity ratio — enforces convexity / fill ratio ───────────
        // A filled rectangle ≈ 1.0; a star ≈ 0.5; a circle outline ring ≈ 0.15.
        // This is the strongest discriminator between a background circle (solid, ~1.0)
        // and a rectangle outline or thin shape, and between a filled shape and a ring.
        double solidityScore = 1.0 - Math.abs(this.solidity - ref.solidity);

        // ── 4. Vertex count — topology invariant ─────────────────────────
        // For small vertex counts (3=triangle, 4=rect, 5=pentagon) each additional
        // vertex is highly significant. Use a stepped absolute-delta penalty that
        // is steeper at low counts and levels out for high-vertex shapes.
        int minV = Math.min(this.vertexCount, ref.vertexCount);
        int maxV = Math.max(this.vertexCount, ref.vertexCount);
        double vertexScore;
        if (maxV == 0) {
            vertexScore = 1.0;
        } else if (minV <= 6) {
            // Low vertex count: treat each delta vertex as a large jump
            // 0 diff → 1.0, 1 diff → 0.70, 2 diff → 0.40, 3+ → 0.10
            int delta = maxV - minV;
            vertexScore = Math.max(0.0, 1.0 - delta * 0.30);
        } else {
            // Higher vertex counts: relative penalty (octagons vs hexagons etc)
            double relDelta = (double)(maxV - minV) / Math.max(1, minV);
            vertexScore = Math.max(0.0, 1.0 - relDelta * relDelta);
        }

        // ── 5. Segment descriptor — geometric traversal (primary structural signal) ──
        // Traverses the raw contour classifying segments as STRAIGHT or CURVED,
        // with noise connections naturally terminated at curvature spikes.
        // This is noise-resistant and scale/rotation invariant.
        double segScore;
        if (this.segmentDescriptor != null && ref.segmentDescriptor != null) {
            segScore = this.segmentDescriptor.similarity(ref.segmentDescriptor);
        } else {
            segScore = 0.5;
        }

        // ── 6. Topology — legacy connected edge structure ─────────────────
        // Still useful as corroboration alongside segmentDescriptor.
        double topoScore;
        if (this.topology != null && ref.topology != null) {
            topoScore = this.topology.similarity(ref.topology);
        } else if (this.topology == null && ref.topology == null) {
            topoScore = 1.0;
        } else {
            topoScore = 0.5;
        }

        // ── 7. Angle histogram intersection — rotation invariant ──────────
        double angleScore = histogramIntersection(this.angleHistogram, ref.angleHistogram);

        // ── 7. Aspect ratio — scale invariant ─────────────────────────────
        double arA = Math.max(this.aspectRatio, 1.0);
        double arB = Math.max(ref.aspectRatio,  1.0);
        double aspectScore = 1.0 - Math.abs(arA - arB) / Math.max(arA, arB);

        // ── Component count penalty ───────────────────────────────────────
        double componentPenalty = 0.0;
        if (this.componentCount != ref.componentCount) {
            int maxC = Math.max(this.componentCount, ref.componentCount);
            componentPenalty = 0.15 * ((double) Math.abs(this.componentCount - ref.componentCount) / maxC);
        }

        // Weights — segmentDescriptor is the primary structural discriminator
        double score = typeScore     * 0.15   // type classification
                     + segScore      * 0.35   // geometric segment traversal (PRIMARY)
                     + topoScore     * 0.15   // legacy topology (corroboration)
                     + circScore     * 0.13   // circularity ratio
                     + solidityScore * 0.12   // solidity ratio
                     + vertexScore   * 0.06   // vertex count tiebreaker
                     + angleScore    * 0.03   // angle histogram
                     + aspectScore   * 0.01   // aspect ratio
                     - componentPenalty;

        double result = Math.max(0.0, Math.min(1.0, score));

        // Hard gate: cap cross-type matches well below the 50% pass threshold
        if (hardGate) result = Math.min(result, 0.35);

        return result;
    }

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
                new double[]{1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6}, 0, 1.0, 0, null, null, Double.NaN);
    }

    @Override
    public String toString() {
        return String.format(
                "VectorSignature{type=%s, vertices=%d, circ=%.2f, solidity=%.2f, concav=%.2f, ar=%.2f, components=%d, normArea=%.4f}",
                type, vertexCount, circularity, solidity, concavityRatio, aspectRatio, componentCount, normalisedArea);
    }
}




