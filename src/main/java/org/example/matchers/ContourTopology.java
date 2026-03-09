package org.example.matchers;

import org.opencv.core.*;

/**
 * Connected-edge descriptor for a contour.
 *
 * <p>For each vertex {@code i} in the approximated polygon, stores only the
 * two quantities that describe its <em>direct connection</em> to the next vertex:
 * <ul>
 *   <li>{@link #normLen}[i]   — length of edge i→i+1, divided by total perimeter
 *                               (scale-invariant)</li>
 *   <li>{@link #turnAngle}[i] — exterior turn angle at vertex i in degrees [0°,180°]
 *                               (rotation-invariant: uses the magnitude of the turn,
 *                               not its compass bearing)</li>
 * </ul>
 *
 * <h2>Why edges-only is sufficient</h2>
 * <p>The all-pairs approach (storing distance/angle to every other vertex) is O(N⁴)
 * to match and gives marginal benefit beyond what cheap scalars already provide.
 * The real discriminating power comes from the connected structure:
 * <ul>
 *   <li>A triangle has 3 edges at ~120° turns, each ~1/3 of the perimeter</li>
 *   <li>A rectangle has 4 edges at ~90° turns, alternating long/short</li>
 *   <li>A star has 10 edges alternating short-steep / long-shallow turns</li>
 * </ul>
 * These patterns are unique per shape and fully captured by the edge sequence.
 * Matching is O(N²) — cyclic alignment of two N-length sequences.
 *
 * <h2>Matching</h2>
 * <p>See {@link #similarity(ContourTopology)}.
 */
public final class ContourTopology {

    /** normLen[i] = length of edge i→(i+1) / total perimeter. Scale-invariant. */
    public final double[] normLen;

    /**
     * turnAngle[i] = exterior turn angle at vertex i, degrees [0,180].
     * 0° = straight on, 90° = right angle, 180° = spike/cusp.
     * Rotation-invariant because it is the magnitude of the direction change.
     */
    public final double[] turnAngle;

    /** Number of vertices (= normLen.length). */
    public final int n;

    /** True for circles/ellipses: fewer than 3 vertices after polygon approximation. */
    public final boolean isCircular;

    // -------------------------------------------------------------------------

    private ContourTopology(double[] normLen, double[] turnAngle, boolean isCircular) {
        this.normLen    = normLen;
        this.turnAngle  = turnAngle;
        this.n          = normLen.length;
        this.isCircular = isCircular;
    }

    // -------------------------------------------------------------------------
    // Factory
    // -------------------------------------------------------------------------

    /**
     * Builds the edge descriptor from an already-approximated polygon.
     *
     * @param approx    vertices from {@code Imgproc.approxPolyDP}
     * @param perimeter full arc-length of the original contour (for normalisation)
     */
    public static ContourTopology build(MatOfPoint2f approx, double perimeter) {
        Point[] pts = approx.toArray();
        int n = pts.length;

        if (n < 3 || perimeter < 1.0) {
            return new ContourTopology(new double[0], new double[0], true);
        }

        double[] len   = new double[n];
        double[] turns = new double[n];

        for (int i = 0; i < n; i++) {
            Point prev = pts[(i - 1 + n) % n];
            Point curr = pts[i];
            Point next = pts[(i + 1) % n];

            // Normalised edge length: curr → next
            double ex = next.x - curr.x, ey = next.y - curr.y;
            len[i] = Math.sqrt(ex * ex + ey * ey) / perimeter;

            // Turn angle at curr: angle between incoming (prev→curr) and outgoing (curr→next)
            double ax = curr.x - prev.x, ay = curr.y - prev.y; // incoming direction
            double bx = next.x - curr.x, by = next.y - curr.y; // outgoing direction
            double magA = Math.sqrt(ax * ax + ay * ay);
            double magB = Math.sqrt(bx * bx + by * by);
            if (magA < 1e-9 || magB < 1e-9) {
                turns[i] = 0;
            } else {
                double cos = (ax * bx + ay * by) / (magA * magB);
                cos = Math.max(-1.0, Math.min(1.0, cos));
                // Interior angle → exterior turn = 180 - interior
                turns[i] = 180.0 - Math.toDegrees(Math.acos(cos));
            }
        }

        return new ContourTopology(len, turns, false);
    }

    // -------------------------------------------------------------------------
    // Similarity  —  O(N²) cyclic alignment
    // -------------------------------------------------------------------------

    /**
     * Returns a similarity score in [0,1] against a reference edge descriptor.
     *
     * <ul>
     *   <li>Both circular → 1.0</li>
     *   <li>One circular, one not → 0.0</li>
     *   <li>Vertex counts differ by more than 2 → 0.0 (structurally incompatible)</li>
     *   <li>Otherwise: try all N cyclic offsets of this sequence against the reference,
     *       score each alignment as the average per-edge similarity, return the best.</li>
     * </ul>
     *
     * <p>Per-edge similarity at each position is:
     * <pre>
     *   edgeSim   = 1 - |normLen_a - normLen_b|          (how similar the edge lengths are)
     *   turnSim   = 1 - |turnAngle_a - turnAngle_b|/180  (how similar the corner sharpness is)
     *   pairScore = 0.5 * edgeSim + 0.5 * turnSim
     * </pre>
     *
     * <p>A hard structural cap of 0.25 per extra/missing vertex is applied so that
     * a triangle (3v) can never score above 0.75 against a rectangle (4v) regardless
     * of how well the edges align.
     */
    public double similarity(ContourTopology ref) {
        if (ref == null) return 0.0;

        if (this.isCircular && ref.isCircular) return 1.0;
        if (this.isCircular != ref.isCircular) return 0.0;

        int na = this.n, nb = ref.n;
        int delta = Math.abs(na - nb);
        if (delta > 2) return 0.0;

        // Hard cap: 0.25 per missing/extra vertex
        double cap = 1.0 - delta * 0.25;

        // Try all cyclic offsets of the smaller against the larger
        double[] aLen   = na <= nb ? this.normLen   : ref.normLen;
        double[] aTurn  = na <= nb ? this.turnAngle : ref.turnAngle;
        double[] bLen   = na <= nb ? ref.normLen    : this.normLen;
        double[] bTurn  = na <= nb ? ref.turnAngle  : this.turnAngle;
        int sm = Math.min(na, nb);
        int lg = Math.max(na, nb);

        double best = 0.0;
        for (int offset = 0; offset < lg; offset++) {
            double sum = 0.0;
            for (int i = 0; i < sm; i++) {
                int j = (i + offset) % lg;
                double edgeSim = Math.max(0.0, 1.0 - Math.abs(aLen[i]  - bLen[j]));
                double turnSim = Math.max(0.0, 1.0 - Math.abs(aTurn[i] - bTurn[j]) / 180.0);
                sum += 0.5 * edgeSim + 0.5 * turnSim;
            }
            double score = sum / sm;
            if (score > best) best = score;
        }

        return best * cap;
    }

    // -------------------------------------------------------------------------

    @Override
    public String toString() {
        return String.format("ContourTopology{n=%d, circular=%b}", n, isCircular);
    }
}
