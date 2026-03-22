package org.example.matchers.vectormatcher.components;

import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.imgproc.Imgproc;

/**
 * Common geometric utility methods for bounding box operations and
 * contour alignment diagnostics.
 */
public final class GeometryUtils {

    /**
     * Computes the union of two bounding rectangles.
     *
     * @param a first rectangle
     * @param b second rectangle
     * @return smallest rectangle containing both a and b
     */
    public static Rect unionRect(Rect a, Rect b) {
        int x1 = Math.min(a.x, b.x);
        int y1 = Math.min(a.y, b.y);
        int x2 = Math.max(a.x + a.width, b.x + b.width);
        int y2 = Math.max(a.y + a.height, b.y + b.height);
        return new Rect(x1, y1, x2 - x1, y2 - y1);
    }

    /**
     * Computes the centre-to-centre distance between two bounding rectangles.
     *
     * @param a first rectangle
     * @param b second rectangle
     * @return Euclidean distance between rectangle centroids
     */
    public static double centreDist(Rect a, Rect b) {
        double ax = a.x + a.width / 2.0;
        double ay = a.y + a.height / 2.0;
        double bx = b.x + b.width / 2.0;
        double by = b.y + b.height / 2.0;
        return Math.hypot(ax - bx, ay - by);
    }

    /**
     * Computes the Intersection-over-Union (IoU) of two bounding rectangles.
     *
     * @param a first rectangle
     * @param b second rectangle
     * @return IoU value in [0, 1], where 1 means perfect overlap
     */
    public static double bboxIoU(Rect a, Rect b) {
        int ix1 = Math.max(a.x, b.x);
        int iy1 = Math.max(a.y, b.y);
        int ix2 = Math.min(a.x + a.width, b.x + b.width);
        int iy2 = Math.min(a.y + a.height, b.y + b.height);

        if (ix2 <= ix1 || iy2 <= iy1) return 0.0;  // no overlap

        double inter = (double) (ix2 - ix1) * (iy2 - iy1);
        double areaA = (double) a.width * a.height;
        double areaB = (double) b.width * b.height;
        double union = areaA + areaB - inter;

        return union > 0 ? inter / union : 0.0;
    }

    /**
     * Formats a bounding rectangle as a compact string.
     *
     * @param r rectangle to format
     * @return string like "(x,y w×h)"
     */
    public static String formatBbox(Rect r) {
        if (r == null) return "null";
        return String.format("(%d,%d %dx%d)", r.x, r.y, r.width, r.height);
    }

    /**
     * Returns {@code true} if the two rectangles overlap (share any area).
     */
    public static boolean rectsIntersect(Rect a, Rect b) {
        return a.x < b.x + b.width  && b.x < a.x + a.width
            && a.y < b.y + b.height && b.y < a.y + a.height;
    }

    // =========================================================================
    // Contour Alignment Score (CAS) — diagnostic probe
    // =========================================================================

    /**
     * Result of the contour alignment check.
     *
     * @param positionMatch fraction of projected ref vertices that land within
     *                      tolerance of a scene contour vertex (0–1)
     * @param angleMatch    mean vertex-angle agreement at matched vertices (0–1)
     * @param combined      positionMatch × angleMatch (0–1)
     * @param rotationDeg   estimated rotation (degrees) from ref to scene
     * @param scale         estimated scale factor from ref to scene
     */
    public record ContourAlignmentScore(
            double positionMatch, double angleMatch, double combined,
            double rotationDeg, double scale) {}

    /**
     * Computes a Contour Alignment Score between a reference contour and a
     * scene contour.
     *
     * <h2>Algorithm</h2>
     * <ol>
     *   <li>Estimate rotation from {@code minAreaRect} angle difference (with
     *       4 hypotheses for rectangular ambiguity: Δ, Δ±90°, Δ+180°).</li>
     *   <li>Estimate scale from minAreaRect dimension ratios.</li>
     *   <li>Build affine transform (scale + rotation + translation) mapping
     *       ref centroid → scene centroid.</li>
     *   <li>Project ref's approxPolyDP vertices into scene space.</li>
     *   <li>For each projected vertex, find the nearest scene approxPolyDP
     *       vertex.  If within distance tolerance AND vertex-angle agreement
     *       is good, count as a matched vertex.</li>
     *   <li>CAS = positionMatch × angleMatch.  The angle check is critical —
     *       it prevents DIAMOND (uniform edges, vertex angles ≈ 90°) from
     *       scoring 100% against RECT_ROTATED_45 (non-uniform edges, vertex
     *       angles ≈ 56°/124°) even though both have 4 vertices at the same
     *       cardinal positions.</li>
     * </ol>
     *
     * @param refContour   reference contour (MatOfPoint)
     * @param sceneContour scene contour (MatOfPoint)
     * @param epsilon      polygon approximation factor
     * @return alignment score, or null if either contour is too small
     */
    public static ContourAlignmentScore computeAlignment(
            MatOfPoint refContour, MatOfPoint sceneContour, double epsilon) {

        if (refContour == null || sceneContour == null
                || refContour.empty() || sceneContour.empty()) return null;

        Point[] refPts   = refContour.toArray();
        Point[] scenePts = sceneContour.toArray();
        if (refPts.length < 3 || scenePts.length < 3) return null;

        MatOfPoint2f ref2f   = new MatOfPoint2f(refPts);
        MatOfPoint2f scene2f = new MatOfPoint2f(scenePts);

        // ── 1. minAreaRect for rotation + scale estimation ──────────────
        RotatedRect refRect   = Imgproc.minAreaRect(ref2f);
        RotatedRect sceneRect = Imgproc.minAreaRect(scene2f);

        // Estimate scale from the larger dimension of the minAreaRect
        double refDim   = Math.max(refRect.size.width, refRect.size.height);
        double sceneDim = Math.max(sceneRect.size.width, sceneRect.size.height);
        double scale = (refDim > 1.0) ? sceneDim / refDim : 1.0;

        // ── 2. ApproxPolyDP vertices for both sides ─────────────────────
        double refPerim   = Imgproc.arcLength(ref2f, true);
        double scenePerim = Imgproc.arcLength(scene2f, true);

        double refEps   = Math.min(Math.max(epsilon * refPerim, 2.0), 8.0);
        double sceneEps = Math.min(Math.max(epsilon * scenePerim, 2.0), 8.0);

        MatOfPoint2f refApprox   = new MatOfPoint2f();
        MatOfPoint2f sceneApprox = new MatOfPoint2f();
        Imgproc.approxPolyDP(ref2f, refApprox, refEps, true);
        Imgproc.approxPolyDP(scene2f, sceneApprox, sceneEps, true);

        ref2f.release();
        scene2f.release();

        Point[] refVerts   = refApprox.toArray();
        Point[] sceneVerts = sceneApprox.toArray();
        refApprox.release();
        sceneApprox.release();

        if (refVerts.length < 3 || sceneVerts.length < 3) return null;

        // ── 3. Compute vertex angles at each vertex ──────────────────────
        double[] refAngles   = computeVertexAngles(refVerts);
        double[] sceneAngles = computeVertexAngles(sceneVerts);

        // ── 4. Ref and scene centroids ──────────────────────────────────
        Point refCentroid   = centroid(refVerts);
        Point sceneCentroid = centroid(sceneVerts);

        // ── 5. Try 4 rotation hypotheses — pick the one with best CAS ──
        // minAreaRect angle is in [-90, 0) for OpenCV; the ambiguity for
        // rectangular shapes means the true rotation could be Δ, Δ±90°, Δ+180°.
        double baseAngle = sceneRect.angle - refRect.angle;
        double[] hypotheses = {
                baseAngle, baseAngle + 90, baseAngle - 90, baseAngle + 180
        };

        ContourAlignmentScore bestCas = null;
        for (double rotDeg : hypotheses) {
            double rotRad = Math.toRadians(rotDeg);
            double cosR = Math.cos(rotRad);
            double sinR = Math.sin(rotRad);

            // Project ref vertices: translate to origin, scale+rotate, translate to scene centroid
            Point[] projected = new Point[refVerts.length];
            for (int i = 0; i < refVerts.length; i++) {
                double dx = refVerts[i].x - refCentroid.x;
                double dy = refVerts[i].y - refCentroid.y;
                projected[i] = new Point(
                        sceneCentroid.x + scale * (cosR * dx - sinR * dy),
                        sceneCentroid.y + scale * (sinR * dx + cosR * dy)
                );
            }

            // Distance tolerance: 15% of scene contour perimeter / vertex count
            // (approximate inter-vertex spacing)
            double distTol = 0.15 * scenePerim / Math.max(sceneVerts.length, 1);
            distTol = Math.max(distTol, 5.0);  // minimum 5 pixels

            // ── 6. Match projected vertices to scene vertices ──────────
            int posMatches   = 0;
            double angleSum  = 0.0;
            int angleMatches = 0;

            for (int i = 0; i < projected.length; i++) {
                // Find nearest scene vertex
                int nearestIdx = -1;
                double nearestDist = Double.MAX_VALUE;
                for (int j = 0; j < sceneVerts.length; j++) {
                    double d = Math.hypot(projected[i].x - sceneVerts[j].x,
                                          projected[i].y - sceneVerts[j].y);
                    if (d < nearestDist) {
                        nearestDist = d;
                        nearestIdx = j;
                    }
                }

                if (nearestDist <= distTol) {
                    posMatches++;

                    // Compare vertex angles — both in [0, 180] degrees
                    double refAngle   = refAngles[i];
                    double sceneAngle = sceneAngles[nearestIdx];
                    double angleDiff  = Math.abs(refAngle - sceneAngle);
                    // Angle agreement: 1.0 at 0° diff, 0.0 at ≥ 45° diff
                    double agreement = Math.max(0.0, 1.0 - angleDiff / 45.0);
                    angleSum += agreement;
                    angleMatches++;
                }
            }

            double posScore   = (double) posMatches / Math.max(projected.length, 1);
            double angleScore = (angleMatches > 0) ? angleSum / angleMatches : 0.0;
            double combined   = posScore * angleScore;

            if (bestCas == null || combined > bestCas.combined()) {
                bestCas = new ContourAlignmentScore(
                        posScore, angleScore, combined, rotDeg, scale);
            }
        }

        return bestCas;
    }

    /**
     * Result of the boundary alignment check.
     *
     * @param boundaryMatch fraction of projected ref vertices that land within
     *                      tolerance of the scene contour boundary (0–1)
     * @param angleMatch    mean vertex-angle agreement at boundary-matched
     *                      vertices (0–1). Angle is measured from the ref
     *                      vertex's local geometry against the scene contour's
     *                      local tangent at the nearest boundary point.
     * @param combined      boundaryMatch × angleMatch (0–1)
     * @param rotationDeg   estimated rotation (degrees) from ref to scene
     * @param scale         estimated scale factor from ref to scene
     */
    public record BoundaryAlignmentScore(
            double boundaryMatch, double angleMatch, double combined,
            double rotationDeg, double scale) {}

    /**
     * Noise-resistant vertex projection check against the scene contour boundary.
     *
     * <p>Unlike {@link #computeAlignment}, which matches projected ref vertices
     * against scene <em>approxPolyDP vertices</em>, this method checks if projected
     * ref vertices fall <b>near the raw scene contour boundary</b>.  This makes it
     * tolerant of background contamination that adds extra vertices to the scene
     * contour — the check only verifies "are the reference's corners present
     * somewhere on the scene contour?"
     *
     * <h2>Use case</h2>
     * <p>IRREGULAR_QUAD on BG_RANDOM_LINES: the background lines physically merge
     * with the shape's edges, creating a 7+ vertex polygon instead of the clean
     * 4-vertex quad.  The vertex-to-vertex CAS fails (different vertex count), but
     * this boundary check can still verify that all 4 of the quad's corners lie on
     * the contaminated contour.
     *
     * <h2>Algorithm</h2>
     * <ol>
     *   <li>Estimate rotation + scale from minAreaRect (same as CAS).</li>
     *   <li>Project ref's approxPolyDP vertices into scene space.</li>
     *   <li>For each projected vertex, compute the minimum distance to the raw
     *       scene contour (using nearest-point search on contour points).</li>
     *   <li>If within tolerance, the vertex is "on the boundary".</li>
     *   <li>At each boundary-matched vertex, compare the ref vertex angle against
     *       the local scene contour angle (measured from the 3 nearest sequential
     *       contour points around the match point).</li>
     * </ol>
     *
     * @param refContour   reference contour (MatOfPoint)
     * @param sceneContour scene contour (MatOfPoint) — raw, NOT approxPolyDP
     * @param epsilon      polygon approximation factor (for ref vertex extraction)
     * @return alignment score, or null if either contour is too small
     */
    public static BoundaryAlignmentScore computeBoundaryAlignment(
            MatOfPoint refContour, MatOfPoint sceneContour, double epsilon) {

        if (refContour == null || sceneContour == null
                || refContour.empty() || sceneContour.empty()) return null;

        Point[] refPts   = refContour.toArray();
        Point[] scenePts = sceneContour.toArray();
        if (refPts.length < 3 || scenePts.length < 3) return null;

        MatOfPoint2f ref2f   = new MatOfPoint2f(refPts);
        MatOfPoint2f scene2f = new MatOfPoint2f(scenePts);

        // ── 1. minAreaRect for rotation + scale estimation ──────────────
        RotatedRect refRect   = Imgproc.minAreaRect(ref2f);
        RotatedRect sceneRect = Imgproc.minAreaRect(scene2f);

        double refDim   = Math.max(refRect.size.width, refRect.size.height);
        double sceneDim = Math.max(sceneRect.size.width, sceneRect.size.height);
        double scale = (refDim > 1.0) ? sceneDim / refDim : 1.0;

        // ── 2. Ref approxPolyDP vertices (clean polygon corners) ────────
        double refPerim = Imgproc.arcLength(ref2f, true);
        double refEps   = Math.min(Math.max(epsilon * refPerim, 2.0), 8.0);
        MatOfPoint2f refApprox = new MatOfPoint2f();
        Imgproc.approxPolyDP(ref2f, refApprox, refEps, true);
        Point[] refVerts = refApprox.toArray();
        refApprox.release();

        // Scene perimeter for tolerance calculation
        double scenePerim = Imgproc.arcLength(scene2f, true);
        ref2f.release();
        scene2f.release();

        if (refVerts.length < 3) return null;

        // ── 3. Compute ref vertex angles ────────────────────────────────
        double[] refAngles = computeVertexAngles(refVerts);

        // ── 4. Centroids ────────────────────────────────────────────────
        Point refCentroid   = centroid(refVerts);
        Point sceneCentroid = centroid(scenePts);

        // ── 5. Try 4 rotation hypotheses ────────────────────────────────
        double baseAngle = sceneRect.angle - refRect.angle;
        double[] hypotheses = {
                baseAngle, baseAngle + 90, baseAngle - 90, baseAngle + 180
        };

        // Distance tolerance: proportional to scene perimeter —
        // a projected vertex within 3% of the scene perimeter from the
        // contour boundary is considered "on the boundary"
        double distTol = Math.max(0.03 * scenePerim, 5.0);

        BoundaryAlignmentScore bestBas = null;
        for (double rotDeg : hypotheses) {
            double rotRad = Math.toRadians(rotDeg);
            double cosR = Math.cos(rotRad);
            double sinR = Math.sin(rotRad);

            // Project ref vertices into scene space
            Point[] projected = new Point[refVerts.length];
            for (int i = 0; i < refVerts.length; i++) {
                double dx = refVerts[i].x - refCentroid.x;
                double dy = refVerts[i].y - refCentroid.y;
                projected[i] = new Point(
                        sceneCentroid.x + scale * (cosR * dx - sinR * dy),
                        sceneCentroid.y + scale * (sinR * dx + cosR * dy)
                );
            }

            // ── 6. Check each projected vertex against raw scene contour ─
            int boundaryMatches = 0;
            double angleSum     = 0.0;
            int angleCount      = 0;

            for (int i = 0; i < projected.length; i++) {
                // Find the nearest point on the raw scene contour
                int nearestIdx = -1;
                double nearestDist = Double.MAX_VALUE;
                for (int j = 0; j < scenePts.length; j++) {
                    double d = Math.hypot(projected[i].x - scenePts[j].x,
                                          projected[i].y - scenePts[j].y);
                    if (d < nearestDist) {
                        nearestDist = d;
                        nearestIdx = j;
                    }
                }

                if (nearestDist <= distTol) {
                    boundaryMatches++;

                    // Measure local contour angle at the match point using
                    // multi-scale windows.  On contaminated contours, a single
                    // window may land across a background junction — trying
                    // multiple scales increases the chance of measuring the
                    // true corner angle.
                    int n = scenePts.length;
                    double bestAgreement = 0.0;
                    for (int ws : new int[]{
                            Math.max(2, n / 40),   // fine scale (~2.5% of contour)
                            Math.max(3, n / 20),   // medium scale (~5%)
                            Math.max(5, n / 10)    // coarse scale (~10%)
                    }) {
                        Point before = scenePts[(nearestIdx - ws + n) % n];
                        Point at     = scenePts[nearestIdx];
                        Point after  = scenePts[(nearestIdx + ws) % n];

                        double ax = before.x - at.x, ay = before.y - at.y;
                        double bx = after.x  - at.x, by = after.y  - at.y;
                        double dot  = ax * bx + ay * by;
                        double magA = Math.sqrt(ax * ax + ay * ay);
                        double magB = Math.sqrt(bx * bx + by * by);
                        double sceneAngle;
                        if (magA < 1e-9 || magB < 1e-9) {
                            sceneAngle = 180.0;
                        } else {
                            double cosA = Math.max(-1.0, Math.min(1.0, dot / (magA * magB)));
                            sceneAngle = Math.toDegrees(Math.acos(cosA));
                        }

                        double angleDiff = Math.abs(refAngles[i] - sceneAngle);
                        // Wider tolerance (60°) for contaminated boundaries
                        double agreement = Math.max(0.0, 1.0 - angleDiff / 60.0);
                        bestAgreement = Math.max(bestAgreement, agreement);
                    }
                    angleSum += bestAgreement;
                    angleCount++;
                }
            }

            double bScore = (double) boundaryMatches / projected.length;
            double aScore = (angleCount > 0) ? angleSum / angleCount : 0.0;
            double combined = bScore * aScore;

            if (bestBas == null || combined > bestBas.combined()) {
                bestBas = new BoundaryAlignmentScore(
                        bScore, aScore, combined, rotDeg, scale);
            }
        }

        return bestBas;
    }

    /**
     * Computes the interior angle at each vertex of a closed polygon.
     *
     * @param pts polygon vertices (ordered, closed)
     * @return angle in degrees [0, 180] at each vertex
     */
    static double[] computeVertexAngles(Point[] pts) {
        int n = pts.length;
        double[] angles = new double[n];
        for (int i = 0; i < n; i++) {
            Point prev = pts[(i - 1 + n) % n];
            Point curr = pts[i];
            Point next = pts[(i + 1) % n];
            double ax = prev.x - curr.x, ay = prev.y - curr.y;
            double bx = next.x - curr.x, by = next.y - curr.y;
            double dot  = ax * bx + ay * by;
            double magA = Math.sqrt(ax * ax + ay * ay);
            double magB = Math.sqrt(bx * bx + by * by);
            if (magA < 1e-9 || magB < 1e-9) {
                angles[i] = 180.0;
                continue;
            }
            double cosAngle = Math.max(-1.0, Math.min(1.0, dot / (magA * magB)));
            angles[i] = Math.toDegrees(Math.acos(cosAngle));
        }
        return angles;
    }

    /**
     * Computes the centroid of a set of points.
     */
    static Point centroid(Point[] pts) {
        double cx = 0, cy = 0;
        for (Point p : pts) { cx += p.x; cy += p.y; }
        return new Point(cx / pts.length, cy / pts.length);
    }

    private GeometryUtils() {}  // static utility class
}

