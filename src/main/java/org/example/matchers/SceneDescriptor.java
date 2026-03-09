package org.example.matchers;

import org.example.colour.SceneColourClusters;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Pre-computed scene description: contours grouped by colour cluster.
 *
 * <p>Build once per scene via {@link #build(Mat)}, then pass to
 * {@link VectorMatcher#match} for every reference you want to search for
 * in that scene.  This means the expensive work — BGR→HSV conversion,
 * hue histogram, per-cluster masking, binarisation and {@code findContours}
 * — is paid exactly once regardless of how many references are matched.
 *
 * <h2>Cost model</h2>
 * <pre>
 *   Without SceneDescriptor:  N references × ~31 image passes = 31N passes
 *   With    SceneDescriptor:  31 passes (build) + N × scoring only
 * </pre>
 *
 * <h2>Lifecycle</h2>
 * <p>The underlying {@link MatOfPoint} contours are owned by this object.
 * Call {@link #release()} when the descriptor is no longer needed to free
 * native OpenCV memory.  The source scene {@link Mat} is not retained —
 * only the extracted contour data is stored.
 */
public final class SceneDescriptor {

    /**
     * Flattened list of all contours across all colour clusters.
     * Grouped by cluster — contours[0..k0] belong to cluster 0,
     * contours[k0+1..k1] to cluster 1, etc. — but for scoring purposes
     * the flat list is all that is needed.
     */
    private final List<List<MatOfPoint>> contoursPerCluster;

    /** Total pixel area of the scene (rows × cols), used for normalised-area scoring. */
    public final double sceneArea;

    /** Wall-clock time in ms taken to build this descriptor (colour extraction + findContours). */
    public final long buildMs;

    private SceneDescriptor(List<List<MatOfPoint>> contoursPerCluster, double sceneArea, long buildMs) {
        this.contoursPerCluster = contoursPerCluster;
        this.sceneArea          = sceneArea;
        this.buildMs            = buildMs;
    }

    // -------------------------------------------------------------------------
    // Factory
    // -------------------------------------------------------------------------

    /**
     * Builds a {@code SceneDescriptor} from a BGR scene image.
     *
     * <p>This is the only expensive operation — all subsequent calls to
     * {@link VectorMatcher#match} reuse the result without re-scanning the image.
     *
     * @param bgrScene  source scene (CV_8UC3 BGR, not retained after this call)
     * @return pre-computed descriptor ready for matching
     */
    public static SceneDescriptor build(Mat bgrScene) {
        long t0   = System.currentTimeMillis();
        double area = (double) bgrScene.rows() * bgrScene.cols();

        List<SceneColourClusters.Cluster> clusters = SceneColourClusters.extract(bgrScene);
        List<List<MatOfPoint>> contoursPerCluster  = new ArrayList<>(clusters.size());

        for (SceneColourClusters.Cluster cluster : clusters) {
            Mat masked = SceneColourClusters.applyMask(bgrScene, cluster);
            contoursPerCluster.add(VectorMatcher.extractContoursFromBinary(masked));
            masked.release();
            cluster.release();
        }

        return new SceneDescriptor(contoursPerCluster, area, System.currentTimeMillis() - t0);
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    /**
     * Returns contours grouped by colour cluster.
     * The outer list is unmodifiable; each inner list is the contours for one cluster.
     */
    public List<List<MatOfPoint>> contoursPerCluster() {
        return Collections.unmodifiableList(contoursPerCluster);
    }

    /**
     * Releases all native OpenCV memory held by the contours in this descriptor.
     * After calling this method the descriptor must not be used.
     */
    public void release() {
        for (List<MatOfPoint> cluster : contoursPerCluster) {
            for (MatOfPoint c : cluster) c.release();
        }
        contoursPerCluster.clear();
    }
}

