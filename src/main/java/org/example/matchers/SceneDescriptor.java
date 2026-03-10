package org.example.matchers;
import org.example.colour.SceneColourClusters;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
/**
 * Pre-computed scene description: contours grouped by colour cluster.
 *
 * Each colour cluster is a binary mask of pixels sharing the same hue group.
 * Running findContours on that mask directly gives the outlines of every
 * distinct connected region of that colour — i.e. the geometric shapes.
 *
 * Overlapping shapes of the same colour merge into one connected region and
 * produce one combined contour, which is correct: they share a colour so they
 * belong to the same cluster.
 *
 * This is colour-agnostic: a red circle, a black rectangle, a white triangle
 * each end up in their own cluster, and their contours come from the real
 * scene geometry via the cluster mask boundary — no greyscale tricks needed.
 */
public final class SceneDescriptor {
    private static final int MIN_AREA = SceneColourClusters.MIN_CONTOUR_AREA;
    public static final class ClusterContours {
        public final List<MatOfPoint> contours;
        /** Centre hue (OpenCV half-degrees 0-179). NaN = achromatic cluster. */
        public final double hue;
        public final boolean achromatic;
        ClusterContours(List<MatOfPoint> contours, double hue, boolean achromatic) {
            this.contours   = contours;
            this.hue        = hue;
            this.achromatic = achromatic;
        }
    }
    private final List<ClusterContours> clusters;
    public final double sceneArea;
    public final long buildMs;
    private SceneDescriptor(List<ClusterContours> clusters, double sceneArea, long buildMs) {
        this.clusters  = clusters;
        this.sceneArea = sceneArea;
        this.buildMs   = buildMs;
    }
    /**
     * Builds a SceneDescriptor from a BGR scene.
     *
     * For each colour cluster extracted by SceneColourClusters:
     *   1. The cluster mask is already a binary image (255 = this colour, 0 = not).
     *   2. findContours on the mask gives the outline of every distinct connected
     *      region of that colour — those are the geometric shapes.
     *   3. Contours below MIN_CONTOUR_AREA are discarded as noise.
     */
    public static SceneDescriptor build(Mat bgrScene) {
        long t0    = System.currentTimeMillis();
        double area = (double) bgrScene.rows() * bgrScene.cols();
        List<SceneColourClusters.Cluster> rawClusters = SceneColourClusters.extract(bgrScene);
        List<ClusterContours> result = new ArrayList<>(rawClusters.size());
        for (SceneColourClusters.Cluster cluster : rawClusters) {
            List<MatOfPoint> contours = contoursFromMask(cluster.mask);
            result.add(new ClusterContours(contours, cluster.hue, cluster.achromatic));
            cluster.release();
        }
        return new SceneDescriptor(result, area, System.currentTimeMillis() - t0);
    }
    /**
     * Finds the outlines of all connected colour regions in a binary cluster mask.
     * The mask is 255 where the pixel belongs to this cluster, 0 elsewhere.
     * Each connected white blob is one distinct shape (or group of touching
     * same-colour shapes). Contours smaller than MIN_AREA are dropped as noise.
     */
    public static List<MatOfPoint> contoursFromMask(Mat mask) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        // Erode by 1px so any blob touching the image border is clipped away.
        // This prevents findContours from tracing the image edge as a contour
        // (e.g. the dark-achromatic background cluster on a black-canvas ref image).
        Mat eroded = new Mat();
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Imgproc.erode(mask, eroded, kernel);
        kernel.release();
        Imgproc.findContours(eroded, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        hierarchy.release();
        eroded.release();
        contours.removeIf(c -> Imgproc.contourArea(c) < MIN_AREA);
        return contours;
    }
    /** Legacy helper kept for visualisation callers that pass a masked BGR image. */
    static List<MatOfPoint> extractContours(Mat maskedBgr) {
        Mat grey = new Mat();
        Mat bin  = new Mat();
        Imgproc.cvtColor(maskedBgr, grey, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(grey, bin, 20, 255, Imgproc.THRESH_BINARY);
        grey.release();
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(bin, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        hierarchy.release();
        bin.release();
        contours.removeIf(c -> Imgproc.contourArea(c) < MIN_AREA);
        return contours;
    }
    public List<ClusterContours> clusters() {
        return Collections.unmodifiableList(clusters);
    }
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