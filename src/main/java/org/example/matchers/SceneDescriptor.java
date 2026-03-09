package org.example.matchers;

import org.example.colour.SceneColourClusters;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Pre-computed scene description: contours grouped by colour cluster.
 *
 * <p>Build once per scene via {@link #build(Mat)}, then pass to
 * {@link VectorMatcher} for every reference you want to search for in that scene.
 *
 * <h2>Cost model</h2>
 * <pre>
 *   Without SceneDescriptor:  N references × ~31 image passes = 31N passes
 *   With    SceneDescriptor:  31 passes (build) + N × scoring only
 * </pre>
 *
 * <h2>Lifecycle</h2>
 * <p>Call {@link #release()} when done to free native OpenCV memory.
 */
public final class SceneDescriptor {

    private static final int MIN_AREA = 64;

    /** One colour cluster's contours plus its hue metadata. */
    public static final class ClusterContours {
        public final List<MatOfPoint> contours;
        /** Centre hue (OpenCV half-degrees 0–179). NaN = achromatic cluster. */
        public final double hue;
        public final boolean achromatic;

        ClusterContours(List<MatOfPoint> contours, double hue, boolean achromatic) {
            this.contours   = contours;
            this.hue        = hue;
            this.achromatic = achromatic;
        }
    }

    private final List<ClusterContours> clusters;

    /** Total pixel area of the scene (rows × cols). */
    public final double sceneArea;

    /** Wall-clock time in ms taken to build this descriptor. */
    public final long buildMs;

    private SceneDescriptor(List<ClusterContours> clusters, double sceneArea, long buildMs) {
        this.clusters  = clusters;
        this.sceneArea = sceneArea;
        this.buildMs   = buildMs;
    }

    public static SceneDescriptor build(Mat bgrScene) {
        long t0     = System.currentTimeMillis();
        double area = (double) bgrScene.rows() * bgrScene.cols();

        List<SceneColourClusters.Cluster> rawClusters = SceneColourClusters.extract(bgrScene);
        List<ClusterContours> clusters = new ArrayList<>(rawClusters.size());

        for (SceneColourClusters.Cluster cluster : rawClusters) {
            Mat masked = SceneColourClusters.applyMask(bgrScene, cluster);
            List<MatOfPoint> contours = extractContours(masked);
            masked.release();
            clusters.add(new ClusterContours(contours, cluster.hue, cluster.achromatic));
            cluster.release();
        }

        return new SceneDescriptor(clusters, area, System.currentTimeMillis() - t0);
    }

    /**
     * Extracts filtered contours from a masked BGR image.
     * Kept here (not in VectorMatcher) to avoid a circular class dependency.
     */
    static List<MatOfPoint> extractContours(Mat maskedBgr) {
        Mat grey = new Mat();
        Mat bin  = new Mat();
        Imgproc.cvtColor(maskedBgr, grey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(grey, bin, 20, 255, Imgproc.THRESH_BINARY);
        grey.release();

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(bin, contours, hierarchy,
                Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        hierarchy.release();
        bin.release();

        contours.removeIf(c -> Imgproc.contourArea(c) < MIN_AREA);
        return contours;
    }

    /** All clusters with their contours and hue metadata. */
    public List<ClusterContours> clusters() {
        return Collections.unmodifiableList(clusters);
    }

    /** Flat backwards-compatible accessor. */
    public List<List<MatOfPoint>> contoursPerCluster() {
        List<List<MatOfPoint>> out = new ArrayList<>(clusters.size());
        for (ClusterContours cc : clusters) out.add(cc.contours);
        return out;
    }

    public void release() {
        for (ClusterContours cc : clusters)
            for (MatOfPoint c : cc.contours) c.release();
        clusters.clear();
    }
}
