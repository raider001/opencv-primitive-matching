package org.example.matchers.vectormatcher;

import org.opencv.core.Rect;

/**
 * Common geometric utility methods for bounding box operations.
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

    private GeometryUtils() {}  // static utility class
}

