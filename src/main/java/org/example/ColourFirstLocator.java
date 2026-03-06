package org.example;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Colour-First Region Proposal engine — Milestone 15.
 *
 * <p>Applies an HSV colour threshold to a scene to generate a small set of
 * candidate bounding windows.  Each candidate is guaranteed to be at least
 * {@value #MIN_SIDE}×{@value #MIN_SIDE} pixels (the reference tile size).
 * Any subsequent matcher runs <em>only inside</em> these windows rather than
 * searching the full scene.
 *
 * <h2>Algorithm</h2>
 * <ol>
 *   <li>Apply {@link ColourPreFilter#applyToScene} with the given tolerance.</li>
 *   <li>Morphological close (5×5 kernel) to merge nearby blobs.</li>
 *   <li>{@code findContours} → bounding rect of each contour.</li>
 *   <li>Merge overlapping rects (union where IoU > 0).</li>
 *   <li>Pad each rect to ≥ {@value #MIN_SIDE}×{@value #MIN_SIDE},
 *       centred on the blob centroid, clamped to scene bounds.</li>
 *   <li>Filter out rects whose area is below {@code minArea}.</li>
 *   <li>Sort descending by area (largest / most colour-matched blob first).</li>
 * </ol>
 *
 * <p>If no candidates survive filtering, a single full-scene rect is returned
 * so that callers always have at least one window to search.
 *
 * <p>This class is <b>stateless and thread-safe</b>.
 */
public final class ColourFirstLocator {

    /** Minimum side length of any proposed window (matches the reference tile size). */
    public static final int MIN_SIDE = 128;

    /** Default minimum blob area (px²) to consider as a candidate. */
    public static final int DEFAULT_MIN_AREA = 100;

    private ColourFirstLocator() {}

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /**
     * Proposes candidate search windows based on colour presence.
     *
     * @param scene        full-colour BGR scene Mat
     * @param refId        reference whose foreground colour drives the threshold
     * @param hueTolerance use {@link ColourPreFilter#LOOSE} or {@link ColourPreFilter#TIGHT}
     * @param minArea      minimum blob area in pixels; blobs smaller than this are discarded
     * @return ordered list of candidate rects (largest first); never empty —
     *         falls back to the full-scene rect if no colour blobs are found
     */
    public static List<Rect> propose(Mat scene, ReferenceId refId,
                                     double hueTolerance, int minArea) {
        Mat mask = ColourPreFilter.applyToScene(scene, refId, hueTolerance);
        List<Rect> candidates = extractCandidates(mask, scene.cols(), scene.rows(), minArea);
        mask.release();
        return candidates;
    }

    /**
     * Convenience overload using {@link #DEFAULT_MIN_AREA}.
     */
    public static List<Rect> propose(Mat scene, ReferenceId refId, double hueTolerance) {
        return propose(scene, refId, hueTolerance, DEFAULT_MIN_AREA);
    }

    /**
     * Returns the single best (largest-area) candidate, or the full-scene rect
     * if no colour blobs are found.
     */
    public static Rect proposeFirst(Mat scene, ReferenceId refId, double hueTolerance) {
        List<Rect> candidates = propose(scene, refId, hueTolerance);
        return candidates.isEmpty()
                ? new Rect(0, 0, scene.cols(), scene.rows())
                : candidates.get(0);
    }

    // -------------------------------------------------------------------------
    // Core extraction logic
    // -------------------------------------------------------------------------

    private static List<Rect> extractCandidates(Mat mask, int sceneW, int sceneH, int minArea) {
        // Morphological close to bridge gaps between nearby blobs
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Mat closed = new Mat();
        Imgproc.morphologyEx(mask, closed, Imgproc.MORPH_CLOSE, kernel);
        kernel.release();

        // Find contours
        List<MatOfPoint> contours  = new ArrayList<>();
        Mat              hierarchy = new Mat();
        Imgproc.findContours(closed, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        closed.release();
        hierarchy.release();

        // Compute bounding rect for each contour
        List<Rect> rects = new ArrayList<>();
        for (MatOfPoint c : contours) {
            double area = Imgproc.contourArea(c);
            if (area >= minArea) {
                rects.add(Imgproc.boundingRect(c));
            }
        }

        if (rects.isEmpty()) {
            // No colour blobs found — fall back to full scene
            return List.of(new Rect(0, 0, sceneW, sceneH));
        }

        // Merge overlapping rects
        List<Rect> merged = mergeOverlapping(rects);

        // Pad each rect to MIN_SIDE × MIN_SIDE, centred, clamped to scene
        List<Rect> padded = new ArrayList<>(merged.size());
        for (Rect r : merged) {
            padded.add(padAndClamp(r, sceneW, sceneH));
        }

        // Sort by area descending (largest / best blob first)
        padded.sort(Comparator.comparingInt((Rect r) -> r.width * r.height).reversed());
        return padded;
    }

    // -------------------------------------------------------------------------
    // Rect helpers
    // -------------------------------------------------------------------------

    /**
     * Iteratively merges any pair of overlapping rects until no overlaps remain.
     */
    private static List<Rect> mergeOverlapping(List<Rect> input) {
        List<Rect> result = new ArrayList<>(input);
        boolean changed = true;
        while (changed) {
            changed = false;
            outer:
            for (int i = 0; i < result.size(); i++) {
                for (int j = i + 1; j < result.size(); j++) {
                    if (overlaps(result.get(i), result.get(j))) {
                        Rect u = union(result.get(i), result.get(j));
                        result.remove(j);
                        result.set(i, u);
                        changed = true;
                        break outer;
                    }
                }
            }
        }
        return result;
    }

    private static boolean overlaps(Rect a, Rect b) {
        return a.x < b.x + b.width  && a.x + a.width  > b.x
            && a.y < b.y + b.height && a.y + a.height > b.y;
    }

    private static Rect union(Rect a, Rect b) {
        int x1 = Math.min(a.x, b.x);
        int y1 = Math.min(a.y, b.y);
        int x2 = Math.max(a.x + a.width,  b.x + b.width);
        int y2 = Math.max(a.y + a.height, b.y + b.height);
        return new Rect(x1, y1, x2 - x1, y2 - y1);
    }

    /**
     * Expands {@code r} to at least {@value #MIN_SIDE}×{@value #MIN_SIDE},
     * keeping it centred on the blob, then clamps to scene bounds.
     */
    private static Rect padAndClamp(Rect r, int sceneW, int sceneH) {
        int cx = r.x + r.width  / 2;
        int cy = r.y + r.height / 2;
        int hw = Math.max(r.width,  MIN_SIDE) / 2;
        int hh = Math.max(r.height, MIN_SIDE) / 2;

        int x = Math.max(0, cx - hw);
        int y = Math.max(0, cy - hh);
        int w = Math.min(Math.max(r.width,  MIN_SIDE), sceneW - x);
        int h = Math.min(Math.max(r.height, MIN_SIDE), sceneH - y);
        return new Rect(x, y, Math.max(1, w), Math.max(1, h));
    }
}

