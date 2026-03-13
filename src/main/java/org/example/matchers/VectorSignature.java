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

        // ── Step 1: Build SegmentDescriptor from the RAW contour BEFORE
        // approxPolyDP.  This preserves curved segments for ellipses and circles,
        // which approxPolyDP collapses to straight-segment polygons — causing an
        // isClosedCurve mismatch against the reference (built from a smooth mask
        // contour) and a segScore of 0.0.
        double rawPerim = Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true);
        SegmentDescriptor rawSegDesc = SegmentDescriptor.build(contour, rawPerim);

        // ── Step 2: ApproxPolyDP reduction for polygon-based metrics (vertex
        // count, circularity, solidity, type classification, topology).
        double strictEps = Math.max(0.02 * rawPerim, 2.0);
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

        return new VectorSignature(
                sig.type, sig.vertexCount, sig.circularity, sig.concavityRatio,
                sig.angleHistogram, sig.componentCount, sig.aspectRatio,
                sig.solidity, sig.topology, rawSegDesc, sig.normalisedArea);
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
            // Same broad polygon family but concavity classification differs.
            // Noisy backgrounds add spurious concavities to otherwise-convex shapes,
            // so we use a light penalty (0.70) rather than a moderate one (0.50).
            typeScore = 0.70;
        } else if ((this.type == ShapeType.CIRCLE && ref.type == ShapeType.CLOSED_CONVEX_POLY)
                || (this.type == ShapeType.CLOSED_CONVEX_POLY && ref.type == ShapeType.CIRCLE)) {
            // Borderline: one classified as CIRCLE, one as POLY.
            // Only allow partial credit if BOTH of the following hold:
            //   (a) the circular side has borderline circularity (< 0.95) — i.e. it is
            //       really a high-vertex polygon (octagon, hexagon) that straddles the
            //       CIRCLE/POLY boundary at this scale.
            //   (b) the polygon side has many vertices (≥ 6) — low-vertex shapes
            //       (triangle=3, rect=4, pentagon=5) cannot legitimately be confused
            //       with a circle. Background random circles must not match these.
            double circSide = (this.type == ShapeType.CIRCLE) ? this.circularity : ref.circularity;
            int    polySide = (this.type == ShapeType.CLOSED_CONVEX_POLY) ? this.vertexCount : ref.vertexCount;
            if (circSide < 0.95 && polySide >= 6) {
                // Borderline high-vertex polygon — partial penalty only
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
        // AR is nearly invariant for a shape (scale-independent, rotation-stable
        // up to 90°). A rect with AR=1.30 should never match a blob with AR=1.82.
        // Applied as a multiplier so a wrong AR suppresses the entire score.
        // Only fires when mismatch > 30% to avoid penalising shapes drawn at
        // slightly different proportions between ref and scene.
        double arA = Math.max(this.aspectRatio, 1.0);
        double arB = Math.max(ref.aspectRatio,  1.0);
        double aspectScore = 1.0 - Math.abs(arA - arB) / Math.max(arA, arB);
        // Multiplier: only applied when aspectScore < 0.70 (> 30% mismatch).
        // At 0.70 → 1.0. At 0.50 → 0.510. At 0.35 → 0.250.
        double arMultiplier = aspectScore >= 0.70 ? 1.0 : Math.pow(aspectScore / 0.70, 2.0);

        // ── Component count penalty ───────────────────────────────────────
        double componentPenalty = 0.0;
        if (this.componentCount != ref.componentCount) {
            int maxC = Math.max(this.componentCount, ref.componentCount);
            componentPenalty = 0.15 * ((double) Math.abs(this.componentCount - ref.componentCount) / maxC);
        }

        // ── Segment-score coherence boost ────────────────────────────────
        // When all OTHER geometric features (type, circularity, solidity, vertices,
        // aspect ratio) agree strongly (each > 0.80), the SegmentDescriptor should
        // not drag the total score below what the geometry implies.  This primarily
        // helps ellipses and circles whose SegmentDescriptor varies across different
        // contour densities but whose shape metrics are highly consistent.
        // The boost floors segScore at 0.60 only when the coherence is very high
        // (all metrics ≥ 0.80 and type = exact match), preventing over-penalisation.
        if (typeScore >= 1.0
                && circScore     >= 0.80
                && solidityScore >= 0.80
                && vertexScore   >= 0.80
                && aspectScore   >= 0.70) {
            segScore = Math.max(segScore, 0.60);
        }

        // Weights — segmentDescriptor is the primary structural discriminator.
        // Aspect ratio is applied as a multiplier rather than an additive term
        // so a significantly wrong AR suppresses the entire score, not just adds
        // a small penalty.  The remaining weights sum to 1.0.
        double score = (typeScore     * 0.15   // type classification (hard gate applies)
                     + segScore      * 0.30   // geometric segment traversal (PRIMARY)
                     + topoScore     * 0.10   // topology (corroboration)
                     + circScore     * 0.13   // circularity ratio
                     + solidityScore * 0.20   // solidity ratio — primary filled/outline discriminator
                     + vertexScore   * 0.08   // vertex count (symmetric min/max)
                     + angleScore    * 0.04   // angle histogram (rotation invariant)
                     - componentPenalty)
                     * arMultiplier;          // AR gate: wrong shape proportions → full suppression

        double result = Math.max(0.0, Math.min(1.0, score));

        // Temporary debug
        System.out.printf("[SIG] type=%.2f seg=%.2f topo=%.2f circ=%.2f solid=%.2f vtx=%.2f ang=%.2f arMult=%.2f => %.3f  [refT=%s detT=%s refC=%.2f detC=%.2f]%n",
            typeScore, segScore, topoScore, circScore, solidityScore, vertexScore, angleScore, arMultiplier, result, ref.type, this.type, ref.circularity, this.circularity);

        // Hard gate: cap cross-type matches well below any pass threshold
        if (hardGate) result = Math.min(result, 0.25);

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
                new double[]{1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6}, 0, 1.0, 0, null, null, Double.NaN);
    }

    @Override
    public String toString() {
        return String.format(
                "VectorSignature{type=%s, vertices=%d, circ=%.2f, solidity=%.2f, concav=%.2f, ar=%.2f, components=%d, normArea=%.4f}",
                type, vertexCount, circularity, solidity, concavityRatio, aspectRatio, componentCount, normalisedArea);
    }
}




