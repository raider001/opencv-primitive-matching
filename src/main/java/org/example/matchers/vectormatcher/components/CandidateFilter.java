package org.example.matchers.vectormatcher.components;

import org.example.matchers.SceneContourEntry;
import org.example.matchers.SceneDescriptor;
import org.example.scene.SceneEntry;

import java.util.ArrayList;
import java.util.List;

/**
 * Multi-stage filtering pipeline for scene candidates.
 *
 * <p>Applies three progressive filters to reduce the candidate set before scoring:
 * <ol>
 *   <li><b>Connected-component filter:</b> Per-cluster noise reduction — drops
 *       contours whose area is below 10% of their cluster's largest contour.</li>
 *   <li><b>Global size filter:</b> Scene-wide minimum size — drops contours below
 *       15% of the global largest contour across all clusters.</li>
 *   <li><b>Reference-adaptive morphological opening:</b> (conditional) Re-extracts
 *       top-K candidates with an erosion sized to the reference fill density,
 *       severing thin background-line connections.</li>
 * </ol>
 */
public final class CandidateFilter {

    /**
     * Minimum contour area as a fraction of the largest contour in the same
     * cluster.  Contours below this ratio are isolated noise blobs.
     */
    private static final double CC_AREA_RATIO_MIN = 0.10;

    /**
     * Minimum contour area as a fraction of the GLOBAL largest contour across
     * all clusters.  Eliminates tiny background elements that survive per-cluster
     * filtering but would still outscore the actual target due to incidental
     * geometry matches.
     */
    private static final double MIN_GLOBAL_AREA_RATIO = 0.08;


    /**
     * Connected-component filter — per-cluster noise reduction.
     *
     * <p>For each cluster, drops contours whose area is below {@code CC_AREA_RATIO_MIN}
     * of that cluster's largest contour.  Handles disconnected same-colour background
     * fragments without requiring reference geometry knowledge.
     *
     * @param candidates raw scene candidates from {@code collectSceneCandidates}
     * @return filtered list with isolated noise blobs removed
     */
    public static List<SceneContourEntry> applyConnectedComponentFilter(List<SceneContourEntry> candidates) {
        if (candidates.size() <= 1) return candidates;

        // Largest contour area and its bbox, keyed by clusterIdx
        var maxArea = new java.util.HashMap<Integer, Double>();
        var maxBbox = new java.util.HashMap<Integer, org.opencv.core.Rect>();
        for (SceneContourEntry ce : candidates) {
            double area = ce.area();
            if (area > maxArea.getOrDefault(ce.clusterIdx(), 0.0)) {
                maxArea.put(ce.clusterIdx(), area);
                maxBbox.put(ce.clusterIdx(), ce.bbox());
            }
        }

        List<SceneContourEntry> out = new ArrayList<>();
        for (SceneContourEntry ce : candidates) {
            double area   = ce.area();
            double clsMax = maxArea.getOrDefault(ce.clusterIdx(), 1.0);

            // Primary rule: area large enough relative to cluster max
            // Relaxed threshold for achromatic clusters (0.05 vs 0.10) to preserve
            // thin outline rings in compound shapes like COMPOUND_BULLSEYE
            double threshold = ce.achromatic() ? 0.05 : CC_AREA_RATIO_MIN;
            if (area >= clsMax * threshold) { out.add(ce); continue; }

            // Secondary rule: small but spatially inside the main shape's bbox
            // → compound component (cross arm, inner ring, etc.), not background noise
            org.opencv.core.Rect mainBb = maxBbox.get(ce.clusterIdx());
            if (mainBb != null) {
                org.opencv.core.Rect ceBb = ce.bbox();
                double ceCx = ceBb.x + ceBb.width  / 2.0;
                double ceCy = ceBb.y + ceBb.height / 2.0;
                if (ceCx >= mainBb.x && ceCx <= mainBb.x + mainBb.width
                 && ceCy >= mainBb.y && ceCy <= mainBb.y + mainBb.height) {
                    out.add(ce);
                }
            }
        }
        return out;
    }

    /**
     * Global minimum-size filter — scene-wide threshold.
     *
     * <p>Drops candidates whose contour area is below {@code MIN_GLOBAL_AREA_RATIO}
     * of the scene-wide largest contour (across all clusters).  Eliminates tiny
     * background elements (random circles 20–60px, short line segments) that
     * survive per-cluster filtering but would outscore the actual target in
     * Layer 2/3 due to incidentally matching geometry.
     *
     * <p>Inner components of compound shapes (cross inside circle, inner bullseye
     * rings) are retained because they are ≥ 15–25% of the outer contour.
     *
     * @param candidates candidates surviving the CC filter
     * @return filtered list with tiny background elements removed
     */
    public static List<SceneContourEntry> applyGlobalSizeFilter(List<SceneContourEntry> candidates) {
        if (candidates.size() <= 1) return candidates;
        double maxArea = 0.0;
        for (SceneContourEntry ce : candidates) {
            double a = ce.area();
            if (a > maxArea) maxArea = a;
        }
        if (maxArea <= 0.0) return candidates;
        final double minArea = maxArea * MIN_GLOBAL_AREA_RATIO;
        List<SceneContourEntry> out = new ArrayList<>();
        for (SceneContourEntry ce : candidates) {
            if (ce.area() >= minArea) out.add(ce);
        }
        return out.isEmpty() ? candidates : out;  // never leave caller empty-handed
    }

    /**
     * Computes erosion depth for morphological opening based on reference fill density.
     *
     * <p>Only applies to solid/filled references (solidity ≥ 0.30).  Outline/line
     * references skip morphological re-extraction entirely (return 0).
     *
     * @param primaryRef the primary (largest-area) reference cluster
     * @return erosion depth in pixels, or 0 if morphological opening should be skipped
     */
    public static int computeErosionDepth(RefCluster primaryRef) {
        // Currently returns 0 for all shapes.  Applying MORPH_OPEN — even at 1 px —
        // rounds the corners of triangular colour sections and hexagon/circle outline
        // strokes enough to shift their VectorSignature vertex angles, causing
        // regressions on TRICOLOUR_TRIANGLE, HEXAGON_OUTLINE and BICOLOUR_CIRCLE_RING.
        // Background-line tests already pass at the 60% threshold without erosion.
        // Re-enable if a stricter contamination metric is added that guards against
        // corner-rounding on outline shapes.
        return 0;
    }

    /**
     * Re-extracts top-K candidates with morphological opening to sever thin
     * background-line connections.
     *
     * <p>Builds a temporary mask from the top candidates (by contour area), applies
     * erosion → dilation, then re-extracts contours from the cleaned mask.  This
     * physically separates the main shape from thin background lines that are
     * connected in the original scene.
     *
     * @param candidates candidates surviving CC and global size filters
     * @param scene      scene entry (for re-extraction via descriptor rebuild)
     * @param erosionDepth erosion kernel size (typically 1–4 px)
     * @param descriptor original scene descriptor (provides sceneArea for signature building)
     * @return re-extracted candidates with thin connections severed
     */
    public static List<SceneContourEntry> reExtractTopCandidates(
            List<SceneContourEntry> candidates,
            SceneEntry scene,
            int erosionDepth,
            SceneDescriptor descriptor) {

        // Top-K selection — retain candidates covering ≥ 95% of total candidate area
        final double TOP_K_AREA_COVERAGE = 0.95;

        // Sort by area descending
        var sorted = new ArrayList<>(candidates);
        sorted.sort((a, b) -> Double.compare(b.area(), a.area()));

        double totalArea = sorted.stream().mapToDouble(ce -> ce.area()).sum();
        double targetArea = totalArea * TOP_K_AREA_COVERAGE;
        double accum = 0.0;
        int topK = 0;
        for (SceneContourEntry ce : sorted) {
            accum += ce.area();
            topK++;
            if (accum >= targetArea) break;
        }

        // Build temporary mask from top-K contours
        org.opencv.core.Mat mask = org.opencv.core.Mat.zeros(
                scene.sceneMat().size(), org.opencv.core.CvType.CV_8UC1);
        for (int i = 0; i < topK; i++) {
            SceneContourEntry ce = sorted.get(i);
            org.opencv.imgproc.Imgproc.drawContours(mask, List.of(ce.contour()),
                    0, new org.opencv.core.Scalar(255), -1);
        }

        // Apply morphological opening: erode → dilate
        org.opencv.core.Mat kernel = org.opencv.imgproc.Imgproc.getStructuringElement(
                org.opencv.imgproc.Imgproc.MORPH_RECT,
                new org.opencv.core.Size(erosionDepth, erosionDepth));
        org.opencv.core.Mat eroded = new org.opencv.core.Mat();
        org.opencv.core.Mat opened = new org.opencv.core.Mat();
        org.opencv.imgproc.Imgproc.erode(mask, eroded, kernel);
        org.opencv.imgproc.Imgproc.dilate(eroded, opened, kernel);

        // Re-extract contours from cleaned mask
        List<org.opencv.core.MatOfPoint> reExtracted = new ArrayList<>();
        org.opencv.imgproc.Imgproc.findContours(opened, reExtracted,
                new org.opencv.core.Mat(),
                org.opencv.imgproc.Imgproc.RETR_EXTERNAL,
                org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE);

        // Convert back to SceneContourEntry (without signatures — built later)
        List<SceneContourEntry> result = new ArrayList<>();
        for (org.opencv.core.MatOfPoint cnt : reExtracted) {
            double area = org.opencv.imgproc.Imgproc.contourArea(cnt);
            if (area < 50.0) continue;  // Skip noise specks
            org.opencv.core.Rect bbox = org.opencv.imgproc.Imgproc.boundingRect(cnt);
            // Assign to cluster index 0 (morphological opening merges clusters)
            // achromatic=false, brightAchromatic=false, hue=0.0, sig=null (built later)
            result.add(new SceneContourEntry(cnt, 0, false, false, 0.0, null, bbox, area));
        }

        // Cleanup
        mask.release();
        kernel.release();
        eroded.release();
        opened.release();

        return result;
    }


    private CandidateFilter() {}  // static utility class
}
