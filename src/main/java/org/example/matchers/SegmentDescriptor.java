package org.example.matchers;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Geometric segment descriptor — replaces {@link ContourTopology}.
 *
 * <h2>Core idea</h2>
 * <p>Instead of treating a contour as a flat sequence of approximated polygon
 * vertices, this class <em>traverses</em> the raw contour points and classifies
 * each run of points into one of two segment types:
 *
 * <ul>
 *   <li><b>STRAIGHT</b> — a run where consecutive direction changes stay near 0°.
 *       Multiple raw points can be merged into one straight segment as long as the
 *       cumulative bearing drift stays below {@link #STRAIGHT_TURN_DEG}.</li>
 *   <li><b>CURVED</b> — a run where the curvature (1/r, measured from three
 *       consecutive points via the circumradius formula) is consistent.  The segment
 *       records the derived normalised x-radius and y-radius of the best-fit ellipse,
 *       making it possible to distinguish circles from ellipses.</li>
 * </ul>
 *
 * <h2>Why this fixes the noise problem</h2>
 * <p>A background line that connects to a circle via a 1–2 px bridge causes a sharp
 * curvature spike at the join point.  The traversal detects this as a curvature
 * discontinuity and ends the current curved segment there — the noise arm is never
 * merged into the shape's segment list.
 *
 * <h2>Matching</h2>
 * <p>Two {@code SegmentDescriptor}s are compared by aligning their segment lists
 * cyclically and scoring:
 * <ul>
 *   <li>Segment type match (STRAIGHT vs CURVED)</li>
 *   <li>Normalised length ratio (scale-invariant)</li>
 *   <li>For CURVED: normalised radius ratio (xrad/perimeter, yrad/perimeter)</li>
 *   <li>Turn angle at junctions between segments</li>
 * </ul>
 */
public final class SegmentDescriptor {

    // ── Tuning constants ────────────────────────────────────────────────────

    /** Max cumulative bearing drift (degrees) to stay within a STRAIGHT segment. */
    private static final double STRAIGHT_TURN_DEG = 12.0;

    /**
     * Max relative curvature variation to stay within a single CURVED segment.
     * κ = 1/r; variation = |κ_i - κ_mean| / κ_mean.
     */
    private static final double CURVE_KAPPA_VAR = 0.55;

    /** Min number of raw points for a segment to be kept (noise filter). */
    private static final int MIN_SEG_POINTS = 3;

    // ── Segment type ────────────────────────────────────────────────────────

    public enum SegType { STRAIGHT, CURVED }

    // ── Segment record ──────────────────────────────────────────────────────

    /**
     * One geometric segment extracted from the contour.
     *
     * <ul>
     *   <li>{@link #type}      — STRAIGHT or CURVED</li>
     *   <li>{@link #normLen}   — arc length / total perimeter (scale-invariant)</li>
     *   <li>{@link #normXRad} / {@link #normYRad} — for CURVED segments: the best-fit
     *       ellipse x/y radii divided by total perimeter.  For STRAIGHT: 0.</li>
     *   <li>{@link #junctionAngle} — exterior turn angle at the <em>end</em> of this
     *       segment (degrees, 0–180).  This is the angle between this segment's exit
     *       direction and the next segment's entry direction.</li>
     * </ul>
     */
    public static final class Seg {
        public final SegType type;
        public final double  normLen;
        public final double  normXRad;
        public final double  normYRad;
        public final double  junctionAngle;

        Seg(SegType type, double normLen, double normXRad, double normYRad, double junctionAngle) {
            this.type          = type;
            this.normLen       = normLen;
            this.normXRad      = normXRad;
            this.normYRad      = normYRad;
            this.junctionAngle = junctionAngle;
        }

        @Override public String toString() {
            if (type == SegType.CURVED)
                return String.format("CURVED(len=%.3f xr=%.3f yr=%.3f jct=%.1f°)",
                        normLen, normXRad, normYRad, junctionAngle);
            return String.format("STRAIGHT(len=%.3f jct=%.1f°)", normLen, junctionAngle);
        }
    }

    // ── Fields ───────────────────────────────────────────────────────────────

    /** Ordered list of segments (cyclic — last connects back to first). */
    public final List<Seg> segments;

    /** True when the whole contour is a single curved loop (circle / ellipse). */
    public final boolean isClosedCurve;

    // ── Constructor ──────────────────────────────────────────────────────────

    private SegmentDescriptor(List<Seg> segments, boolean isClosedCurve) {
        this.segments      = segments;
        this.isClosedCurve = isClosedCurve;
    }

    // =========================================================================
    // Factory
    // =========================================================================

    /**
     * Builds a {@code SegmentDescriptor} from a raw contour (not approximated —
     * we want every pixel-level point so the geometry is accurate).
     *
     * @param contour   raw contour from {@code findContours}
     * @param perimeter arc-length of the closed contour
     */
    public static SegmentDescriptor build(MatOfPoint contour, double perimeter) {
        if (contour == null || contour.empty() || perimeter < 1.0)
            return empty();
        return build(contour.toArray(), perimeter);
    }

    /**
     * Builds a {@code SegmentDescriptor} from pre-extracted contour points,
     * avoiding a redundant {@code contour.toArray()} native→Java copy when the
     * caller already holds the point array (OPT-J).
     *
     * @param rawPts    contour points (from {@code MatOfPoint.toArray()})
     * @param perimeter arc-length of the closed contour
     */
    public static SegmentDescriptor build(Point[] rawPts, double perimeter) {
        if (rawPts == null || rawPts.length < 3 || perimeter < 1.0)
            return empty();
        int n = rawPts.length;

        // ── Densify sparse contours ───────────────────────────────────────
        // CHAIN_APPROX_SIMPLE compresses every straight side of a polygon to
        // just its two endpoints, so an octagon arrives with only 8 points.
        // With MIN_SEG_POINTS = 3, those 8 points cannot form 8 STRAIGHT
        // runs; they collapse into a single pseudo-curve and the shape is
        // mistakenly flagged as isClosedCurve = true (indistinguishable from
        // a circle).  Densification interpolates ≥5 evenly-spaced points along
        // each edge, giving the traversal enough samples to correctly identify
        // straight vs. curved runs.
        // Trigger: average inter-point spacing > 4 px (all CHAIN_APPROX_SIMPLE
        // polygons) while true circles/ellipses are already dense (spacing ~1–2 px).
        Point[] pts;
        if (perimeter / n > 4.0) {
            pts = densifyContour(rawPts, 5);
            n   = pts.length;
        } else {
            pts = rawPts;
        }
        if (n < 6) return empty();   // still too few even after densification

        // ── Step 1: compute per-point curvature and direction ────────────
        double[] kappa  = new double[n]; // curvature at each point
        double[] dir    = new double[n]; // bearing of the edge leaving point i (radians)

        for (int i = 0; i < n; i++) {
            Point p0 = pts[(i - 1 + n) % n];
            Point p1 = pts[i];
            Point p2 = pts[(i + 1) % n];

            kappa[i] = circumCurvature(p0, p1, p2);

            double dx = p2.x - p0.x, dy = p2.y - p0.y;
            dir[i] = Math.atan2(dy, dx);
        }

        // ── Step 2: traverse and classify runs ───────────────────────────
        // We decide the type of each run by its curvature profile.
        // A run starts STRAIGHT until curvature builds up consistently,
        // at which point it transitions to CURVED.

        List<int[]> rawSegs = new ArrayList<>(); // each int[] = {startIdx, endIdx, type (0=S,1=C)}

        int i = 0;
        while (i < n) {
            // Determine segment type by looking ahead
            double k0 = kappa[i];
            boolean startCurved = k0 > 0.01; // non-zero curvature at start = probably curved

            int segStart = i;
            int segType; // 0=straight, 1=curved

            if (startCurved) {
                // Try to extend a CURVED run
                segType = 1;
                double kMean = k0;
                int count = 1;
                int j = i + 1;
                while (j < n) {
                    double kj   = kappa[j % n];
                    double kNew = (kMean * count + kj) / (count + 1);
                    double var  = Math.abs(kj - kNew) / Math.max(kNew, 1e-6);
                    if (var > CURVE_KAPPA_VAR) break;
                    kMean = kNew;
                    count++;
                    j++;
                    if (j - segStart >= n) break; // gone full circle
                }
                int segEnd = Math.min(j, segStart + n - 1);
                if (segEnd - segStart < MIN_SEG_POINTS) {
                    // Too short — absorb into next segment by forcing straight
                    segType = 0;
                    segEnd  = i + 1;
                }
                rawSegs.add(new int[]{segStart, segEnd, segType});
                i = segEnd;
            } else {
                // Try to extend a STRAIGHT run
                segType = 0;
                double bearingStart = dir[i];
                int j = i + 1;
                double cumTurn = 0;
                while (j < n) {
                    double diff = angleDiffDeg(dir[j % n], bearingStart);
                    cumTurn += Math.abs(diff - cumTurn); // running cumulative drift
                    if (cumTurn > STRAIGHT_TURN_DEG) break;
                    j++;
                    if (j - segStart >= n) break;
                }
                int segEnd = Math.min(j, segStart + n - 1);
                if (segEnd - segStart < MIN_SEG_POINTS) segEnd = i + 1;
                rawSegs.add(new int[]{segStart, segEnd, segType});
                i = segEnd;
            }

            if (i >= n) break;
        }

        // ── Step 3: merge tiny segments into neighbours ──────────────────
        rawSegs = mergeShortSegments(rawSegs, n, MIN_SEG_POINTS * 2);

        // ── Step 4: check if this is a closed curve (circle / ellipse) ───
        // A closed curve = the whole contour is one or two CURVED segments
        long curvedCount = rawSegs.stream().filter(s -> s[2] == 1).count();
        boolean isClosedCurve = (rawSegs.size() <= 2 && curvedCount == rawSegs.size());

        // ── Step 5: build Seg objects ─────────────────────────────────────
        List<Seg> segs = new ArrayList<>();
        int numSegs = rawSegs.size();

        for (int si = 0; si < numSegs; si++) {
            int[] s     = rawSegs.get(si);
            int   sType = s[2];
            int   sFrom = s[0];
            int   sTo   = Math.min(s[1], n - 1);

            // Arc length of this segment
            double segLen = 0;
            for (int k = sFrom; k < sTo; k++) {
                Point pa = pts[k % n], pb = pts[(k + 1) % n];
                double dx = pb.x - pa.x, dy = pb.y - pa.y;
                segLen += Math.sqrt(dx * dx + dy * dy);
            }
            double normLen = segLen / Math.max(perimeter, 1.0);

            // Junction angle at end of this segment (between this segment's exit and next's entry)
            int[]  nextSeg   = rawSegs.get((si + 1) % numSegs);
            double exitBear  = dir[sTo   % n];
            double entryBear = dir[nextSeg[0] % n];
            double jctAngle  = Math.abs(angleDiffDeg(exitBear, entryBear));
            jctAngle = Math.min(180.0, jctAngle);

            // Curve parameters
            double normXRad = 0, normYRad = 0;
            if (sType == 1) {
                // Fit an ellipse to the points in this segment
                double[] radii = fitEllipseRadii(pts, sFrom, sTo, n);
                normXRad = radii[0] / Math.max(perimeter, 1.0);
                normYRad = radii[1] / Math.max(perimeter, 1.0);
            }

            segs.add(new Seg(sType == 0 ? SegType.STRAIGHT : SegType.CURVED,
                    normLen, normXRad, normYRad, jctAngle));
        }

        return new SegmentDescriptor(segs, isClosedCurve);
    }

    // =========================================================================
    // Similarity  —  cyclic alignment of segment lists
    // =========================================================================

    /**
     * Returns a similarity score in [0, 1] between this descriptor and {@code ref}.
     *
     * <ul>
     *   <li>Both closed curves → compare radii ratios only</li>
     *   <li>Otherwise: try all cyclic alignments of the shorter list against the
     *       longer, score each alignment, return the best</li>
     * </ul>
     */
    public double similarity(SegmentDescriptor ref) {
        if (ref == null) return 0.0;
        if (segments.isEmpty() && ref.segments.isEmpty()) return 1.0;
        if (segments.isEmpty() || ref.segments.isEmpty()) return 0.0;

        // Both closed curves (circles / ellipses)
        if (this.isClosedCurve && ref.isClosedCurve) {
            return closedCurveSimilarity(ref);
        }
        // One closed, one not
        if (this.isClosedCurve != ref.isClosedCurve) return 0.0;

        int na = this.segments.size();
        int nb = ref.segments.size();
        int refCount = nb; // nb is always the reference side (caller passes ref)
        int delta = Math.abs(na - nb);

        // ── Ratio-based missing-segment penalty ───────────────────────────
        // Distinguish two cases:
        //
        // (A) Scene has FEWER segments than reference — structurally incomplete.
        //     Penalise hard: cap = matched/expected.  Hard reject below 50%.
        //     e.g. ref=4 (rect), scene=2 → cap=0.50, hard reject at scene=1.
        //
        // (B) Scene has MORE segments than reference — noise added extra segments.
        //     Softer penalty: we still match the best ref-count segments from the
        //     scene, but cap at 1.0 - (extra/total)*0.4 so pure noise can't win.
        //     e.g. ref=1 (circle), scene=3 → cap = 1.0 - (2/3)*0.4 = 0.73.
        double cap;
        if (na < nb) {
            // Scene is missing segments — strict fraction-based cap
            double matchedFraction = (double) na / refCount;
            if (matchedFraction < 0.5) return 0.0; // fewer than half — hard reject
            cap = matchedFraction;
        } else if (na > nb) {
            // Scene has extra noise segments — soft penalty
            double extraFraction = (double)(na - nb) / na;
            cap = Math.max(0.40, 1.0 - extraFraction * 0.40);
        } else {
            cap = 1.0; // exact match
        }

        List<Seg> aSegs = na <= nb ? this.segments : ref.segments;
        List<Seg> bSegs = na <= nb ? ref.segments  : this.segments;
        int sm = Math.min(na, nb);
        int lg = Math.max(na, nb);

        double best = 0.0;
        for (int offset = 0; offset < lg; offset++) {
            double sum = 0.0;
            for (int i = 0; i < sm; i++) {
                Seg a = aSegs.get(i);
                Seg b = bSegs.get((i + offset) % lg);
                sum += segScore(a, b);
            }
            // Average over the reference count — unmatched reference segments
            // implicitly contribute 0 (they are missing from the scene).
            double score = sum / Math.max(refCount, 1);
            if (score > best) best = score;
        }

        return Math.min(1.0, best * cap);
    }

    /** Score two individual segments against each other. Returns [0, 1]. */
    private static double segScore(Seg a, Seg b) {
        // Type mismatch — no partial credit; straight and curved are fundamentally different
        if (a.type != b.type) return 0.0;

        // Normalised length similarity
        double lenSim = Math.max(0.0, 1.0 - Math.abs(a.normLen - b.normLen) / Math.max(a.normLen, b.normLen + 1e-9));

        // Junction angle similarity
        double jctSim = Math.max(0.0, 1.0 - Math.abs(a.junctionAngle - b.junctionAngle) / 180.0);

        if (a.type == SegType.STRAIGHT) {
            return 0.50 * lenSim + 0.50 * jctSim;
        } else {
            // Curved: also compare radii ratios
            double xRadSim = radiiSim(a.normXRad, b.normXRad);
            double yRadSim = radiiSim(a.normYRad, b.normYRad);
            return 0.30 * lenSim + 0.30 * jctSim + 0.20 * xRadSim + 0.20 * yRadSim;
        }
    }

    private static double radiiSim(double ra, double rb) {
        double maxR = Math.max(ra, rb);
        if (maxR < 1e-9) return 1.0;
        return Math.max(0.0, 1.0 - Math.abs(ra - rb) / maxR);
    }

    private double closedCurveSimilarity(SegmentDescriptor ref) {
        // Aggregate normalised radii across all curved segments
        double[] myR   = aggregateRadii(this.segments);
        double[] refR  = aggregateRadii(ref.segments);
        double xSim = radiiSim(myR[0], refR[0]);
        double ySim = radiiSim(myR[1], refR[1]);
        // Ratio xrad/yrad — distinguishes circle (ratio≈1) from ellipse
        double myRatio  = myR[1]  > 1e-9 ? myR[0]  / myR[1]  : 1.0;
        double refRatio = refR[1] > 1e-9 ? refR[0] / refR[1] : 1.0;
        double ratioSim = Math.max(0.0, 1.0 - Math.abs(myRatio - refRatio) / Math.max(myRatio, refRatio + 1e-9));
        return 0.35 * xSim + 0.35 * ySim + 0.30 * ratioSim;
    }

    private static double[] aggregateRadii(List<Seg> segs) {
        double sumX = 0, sumY = 0; int n = 0;
        for (Seg s : segs) {
            if (s.type == SegType.CURVED) { sumX += s.normXRad; sumY += s.normYRad; n++; }
        }
        if (n == 0) return new double[]{0, 0};
        return new double[]{sumX / n, sumY / n};
    }

    // =========================================================================
    // Geometry helpers
    // =========================================================================

    /**
     * Circumradius curvature κ = 1/R from three consecutive points.
     * Uses the formula: κ = 4·Area(triangle) / (|ab|·|bc|·|ca|).
     */
    private static double circumCurvature(Point p0, Point p1, Point p2) {
        double ax = p1.x - p0.x, ay = p1.y - p0.y;
        double bx = p2.x - p1.x, by = p2.y - p1.y;
        double cx = p0.x - p2.x, cy = p0.y - p2.y;
        double cross = Math.abs(ax * by - ay * bx); // 2 * triangle area
        double lenA  = Math.sqrt(ax * ax + ay * ay);
        double lenB  = Math.sqrt(bx * bx + by * by);
        double lenC  = Math.sqrt(cx * cx + cy * cy);
        double denom = lenA * lenB * lenC;
        return denom < 1e-9 ? 0.0 : (2.0 * cross) / denom;
    }

    /**
     * Signed angle difference between two bearings in degrees (result in [-180, 180]).
     */
    private static double angleDiffDeg(double bearA, double bearB) {
        double d = Math.toDegrees(bearA - bearB);
        while (d >  180) d -= 360;
        while (d < -180) d += 360;
        return d;
    }

    /**
     * Fit an ellipse to a sub-range of raw contour points and return [xRadius, yRadius].
     * Uses OpenCV's fitEllipse if there are enough points, otherwise falls back to
     * the circumradius of the centroid.
     */
    private static double[] fitEllipseRadii(Point[] pts, int from, int to, int n) {
        int count = to - from;
        if (count < 5) {
            // Not enough for fitEllipse — estimate radius from mean curvature
            double kSum = 0; int kCount = 0;
            for (int i = from; i < to; i++) {
                Point p0 = pts[(i - 1 + n) % n];
                Point p1 = pts[i % n];
                Point p2 = pts[(i + 1) % n];
                double k = circumCurvature(p0, p1, p2);
                if (k > 1e-9) { kSum += k; kCount++; }
            }
            double r = kCount > 0 ? 1.0 / (kSum / kCount) : 0;
            return new double[]{r, r};
        }

        try {
            MatOfPoint2f mat = new MatOfPoint2f();
            List<Point> sub = new ArrayList<>(count);
            for (int i = from; i < to; i++) sub.add(pts[i % n]);
            mat.fromList(sub);
            RotatedRect ellipse = Imgproc.fitEllipse(mat);
            mat.release();
            double xRad = ellipse.size.width  / 2.0;
            double yRad = ellipse.size.height / 2.0;
            return new double[]{Math.max(xRad, yRad), Math.min(xRad, yRad)};
        } catch (Exception e) {
            return new double[]{0, 0};
        }
    }

    /**
     * Merge segments that are too short into their neighbours (same type preferred).
     */
    private static List<int[]> mergeShortSegments(List<int[]> segs, int n, int minLen) {
        if (segs.size() <= 1) return segs;
        boolean changed = true;
        while (changed && segs.size() > 1) {
            changed = false;
            for (int i = 0; i < segs.size(); i++) {
                int[] s = segs.get(i);
                if (s[1] - s[0] < minLen) {
                    // Merge into next
                    int next = (i + 1) % segs.size();
                    int[] ns = segs.get(next);
                    int[] merged = new int[]{s[0], ns[1], ns[2]};
                    segs.remove(i);
                    if (next > i) next--;
                    segs.set(next < 0 ? 0 : next, merged);
                    changed = true;
                    break;
                }
            }
        }
        return segs;
    }

    /**
     * Interpolates {@code minPtsPerEdge} evenly-spaced points along each edge of
     * a polygon contour, plus additional points so no edge has fewer than
     * {@code ceil(edgeLen / 3)} samples.  This turns a sparse CHAIN_APPROX_SIMPLE
     * polygon (e.g. 8 corners for an octagon) into a dense point cloud where
     * straight sides produce near-zero curvature runs that the traversal reliably
     * classifies as STRAIGHT segments.
     */
    private static Point[] densifyContour(Point[] pts, int minPtsPerEdge) {
        int n = pts.length;
        List<Point> out = new ArrayList<>(n * minPtsPerEdge);
        for (int i = 0; i < n; i++) {
            out.add(pts[i]);
            Point p0 = pts[i];
            Point p1 = pts[(i + 1) % n];
            double dx  = p1.x - p0.x, dy = p1.y - p0.y;
            double len = Math.sqrt(dx * dx + dy * dy);
            int steps  = Math.max(minPtsPerEdge, (int) Math.ceil(len / 3.0));
            for (int j = 1; j < steps; j++) {
                double t = (double) j / steps;
                out.add(new Point(p0.x + t * dx, p0.y + t * dy));
            }
        }
        return out.toArray(new Point[0]);
    }

    private static SegmentDescriptor empty() {
        return new SegmentDescriptor(List.of(), false);
    }

    // =========================================================================

    @Override
    public String toString() {
        return "SegmentDescriptor{segs=" + segments.size()
                + ", closedCurve=" + isClosedCurve + ", " + segments + "}";
    }
}

