package org.example;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Colour-First Region Proposal engine — Milestones 15 &amp; 21.
 *
 * <p>Applies an HSV colour threshold to a scene to generate a small set of
 * candidate bounding windows.  Each candidate is guaranteed to be at least
 * {@value #MIN_SIDE}×{@value #MIN_SIDE} pixels (the reference tile size).
 * Any subsequent matcher runs <em>only inside</em> these windows rather than
 * searching the full scene.
 *
 * <h2>Single-colour algorithm (Milestone 15)</h2>
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
 * <h2>Multi-colour algorithm (Milestone 21)</h2>
 * <ol>
 *   <li>For each distinct foreground colour in the reference, run the single-colour
 *       algorithm above to produce a per-channel candidate list.</li>
 *   <li>Union all per-channel candidate lists.</li>
 *   <li>Run a second merge pass across the combined list to collapse any cross-channel
 *       overlaps.</li>
 *   <li>Sort by area descending.</li>
 * </ol>
 *
 * <p>If no candidates survive filtering across all channels, a single full-scene rect
 * is returned so that callers always have at least one window to search.
 *
 * <p>This class is <b>stateless and thread-safe</b>.
 */
public final class ColourFirstLocator {

    /** Minimum side length of any proposed window (matches the reference tile size). */
    public static final int MIN_SIDE = 128;

    /** Default minimum blob area (px²) to consider as a candidate. */
    public static final int DEFAULT_MIN_AREA = 100;

    /**
     * Default minimum hue separation (OpenCV 0–179 scale) for two colours to be
     * considered distinct enough to warrant separate proposal windows.
     * At 25° the 8-palette hues (spaced ~45° apart) are cleanly separated.
     */
    public static final int DEFAULT_MIN_HUE_SEP = 25;

    /**
     * Default minimum saturation (0–255) for a colour cluster to be treated as a
     * distinct foreground channel.  Below this threshold the colour is near-grey
     * and its hue is unreliable.
     */
    public static final int DEFAULT_MIN_SATURATION = 40;

    private ColourFirstLocator() {}

    // -------------------------------------------------------------------------
    // Public API — single-colour (Milestone 15)
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
    // Public API — multi-colour (Milestone 21)
    // -------------------------------------------------------------------------

    /**
     * Proposes candidate windows for an explicit list of BGR foreground colours.
     * One set of windows is proposed per colour channel, then all sets are merged.
     *
     * @param scene        full-colour BGR scene Mat
     * @param bgrColours   explicit list of BGR colours to threshold (e.g. from
     *                     {@link ReferenceImageFactory#foregroundColours(ReferenceId)})
     * @param hueTolerance hue tolerance in degrees (use {@link ColourPreFilter#LOOSE}/TIGHT)
     * @param minArea      minimum blob area in px²
     * @return merged, deduplicated, area-sorted list of candidate rects; never empty
     */
    public static List<Rect> propose(Mat scene, List<Scalar> bgrColours,
                                     double hueTolerance, int minArea) {
        if (bgrColours == null || bgrColours.isEmpty()) {
            return List.of(new Rect(0, 0, scene.cols(), scene.rows()));
        }
        List<Rect> combined = new ArrayList<>();
        for (Scalar bgr : bgrColours) {
            ColourRange range = ColourPreFilter.extractColourRange(bgr, hueTolerance);
            Mat mask = ColourPreFilter.apply(scene, range.hsvLower(), range.hsvUpper());
            List<Rect> candidates = extractCandidates(mask, scene.cols(), scene.rows(), minArea);
            mask.release();
            // Only add real candidates (not full-scene fallbacks) to the combined pool
            for (Rect r : candidates) {
                if (r.width < scene.cols() || r.height < scene.rows()) {
                    combined.add(r);
                }
            }
        }
        if (combined.isEmpty()) {
            return List.of(new Rect(0, 0, scene.cols(), scene.rows()));
        }
        List<Rect> merged = mergeOverlapping(combined);
        merged.sort(Comparator.comparingInt((Rect r) -> r.width * r.height).reversed());
        return merged;
    }

    /**
     * Auto-extracts distinct foreground colours from the reference and proposes
     * merged candidate windows for all channels.
     *
     * <p>First checks {@link ReferenceImageFactory#foregroundColours(ReferenceId)} for
     * explicit registration.  If only one colour is registered (single-colour reference),
     * delegates to the single-colour {@link #propose(Mat, ReferenceId, double, int)}.
     * Otherwise runs the multi-colour pipeline.
     *
     * @param scene         full-colour BGR scene Mat
     * @param refId         reference ID (used to look up registered colours)
     * @param hueTolerance  hue tolerance (use {@link ColourPreFilter#LOOSE}/TIGHT)
     * @param minArea       minimum blob area in px²
     * @param minHueSep     minimum hue separation to treat two colours as distinct
     *                      (ignored when explicit registration covers > 1 colour)
     * @param minSaturation minimum saturation; colours below this are skipped
     * @return merged, deduplicated candidate rects; never empty
     */
    public static List<Rect> proposeMulti(Mat scene, ReferenceId refId,
                                          double hueTolerance, int minArea,
                                          int minHueSep, int minSaturation) {
        List<Scalar> colours = ReferenceImageFactory.foregroundColours(refId);
        if (colours.size() == 1) {
            // Single-colour ref — identical behaviour to Milestone 15
            return propose(scene, refId, hueTolerance, minArea);
        }
        // Filter out near-grey colours below the saturation floor
        List<Scalar> saturated = colours.stream()
                .filter(bgr -> {
                    double[] hsv = bgrToHsv(bgr);
                    return hsv[1] * 255 >= minSaturation;
                })
                .toList();
        if (saturated.isEmpty()) {
            return propose(scene, refId, hueTolerance, minArea);
        }
        return propose(scene, saturated, hueTolerance, minArea);
    }

    /** Convenience overload using default parameters. */
    public static List<Rect> proposeMulti(Mat scene, ReferenceId refId, double hueTolerance) {
        return proposeMulti(scene, refId, hueTolerance, DEFAULT_MIN_AREA,
                DEFAULT_MIN_HUE_SEP, DEFAULT_MIN_SATURATION);
    }

    /**
     * Extracts a list of perceptually distinct foreground BGR colours from a reference
     * image by analysing its hue histogram.
     *
     * <p>This is a histogram-based fallback for references where explicit colour
     * registration is unavailable.  It finds hue peaks in the non-black foreground
     * pixels separated by at least {@code minHueSep} degrees, ignoring near-grey
     * pixels below {@code minSaturation}.
     *
     * @param refMat        128×128 BGR reference image
     * @param minHueSep     minimum hue separation (0–179 OpenCV scale) between distinct colours
     * @param minSaturation minimum saturation (0–255) for a pixel to be included
     * @return list of distinct BGR colours found (may be empty if the image is all-grey)
     */
    public static List<Scalar> extractDistinctColours(Mat refMat,
                                                       int minHueSep,
                                                       int minSaturation) {
        // Convert to HSV
        Mat hsv = new Mat();
        Imgproc.cvtColor(refMat, hsv, Imgproc.COLOR_BGR2HSV);

        // Build a hue histogram (180 bins) counting only sufficiently saturated pixels
        int[] hueHist = new int[180];
        for (int y = 0; y < hsv.rows(); y++) {
            for (int x = 0; x < hsv.cols(); x++) {
                double[] pixel = hsv.get(y, x);
                int h = (int) pixel[0]; // 0–179
                int s = (int) pixel[1]; // 0–255
                int v = (int) pixel[2]; // 0–255
                if (s >= minSaturation && v > 10) {
                    hueHist[h]++;
                }
            }
        }
        hsv.release();

        // Find peaks: a hue bin is a peak if it is the local maximum within minHueSep
        // on both sides (circular wrap around 0/179)
        List<Integer> peakHues = new ArrayList<>();
        int totalPixels = refMat.rows() * refMat.cols();
        int minCount = Math.max(5, totalPixels / 200); // ignore tiny smudges

        for (int h = 0; h < 180; h++) {
            if (hueHist[h] < minCount) continue;
            boolean isPeak = true;
            for (int d = 1; d <= minHueSep && isPeak; d++) {
                int hPrev = (h - d + 180) % 180;
                int hNext = (h + d) % 180;
                if (hueHist[hPrev] > hueHist[h] || hueHist[hNext] > hueHist[h]) {
                    isPeak = false;
                }
            }
            if (isPeak) peakHues.add(h);
        }

        // Remove peaks that are too close to each other (keep the higher one)
        List<Integer> filteredPeaks = new ArrayList<>(peakHues);
        boolean changed = true;
        while (changed) {
            changed = false;
            for (int i = 0; i < filteredPeaks.size() && !changed; i++) {
                for (int j = i + 1; j < filteredPeaks.size() && !changed; j++) {
                    int ha = filteredPeaks.get(i), hb = filteredPeaks.get(j);
                    int dist = Math.min(Math.abs(ha - hb), 180 - Math.abs(ha - hb));
                    if (dist < minHueSep) {
                        // Remove the weaker peak
                        if (hueHist[ha] >= hueHist[hb]) {
                            filteredPeaks.remove(j);
                        } else {
                            filteredPeaks.remove(i);
                        }
                        changed = true;
                    }
                }
            }
        }

        // Convert hue peaks back to BGR Scalars
        List<Scalar> colours = new ArrayList<>();
        for (int h : filteredPeaks) {
            // HSV(h, 200, 200) → BGR
            Mat hsvPx = new Mat(1, 1, CvType.CV_8UC3);
            hsvPx.put(0, 0, h, 200, 200);
            Mat bgrPx = new Mat();
            Imgproc.cvtColor(hsvPx, bgrPx, Imgproc.COLOR_HSV2BGR);
            double[] px = bgrPx.get(0, 0);
            colours.add(new Scalar(px[0], px[1], px[2]));
            hsvPx.release();
            bgrPx.release();
        }
        return colours;
    }

    /** Converts a BGR Scalar to a normalised HSV double[3] (H:0–179, S:0–1, V:0–1). */
    private static double[] bgrToHsv(Scalar bgr) {
        Mat bgrPx  = new Mat(1, 1, CvType.CV_8UC3);
        bgrPx.put(0, 0, bgr.val[0], bgr.val[1], bgr.val[2]);
        Mat hsvPx = new Mat();
        Imgproc.cvtColor(bgrPx, hsvPx, Imgproc.COLOR_BGR2HSV);
        double[] px = hsvPx.get(0, 0);
        bgrPx.release();
        hsvPx.release();
        return new double[]{ px[0], px[1] / 255.0, px[2] / 255.0 };
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

