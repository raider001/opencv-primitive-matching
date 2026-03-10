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
        // Zero the 1px border on a clone so any contour that literally touches
        // the image edge is disconnected — but thin interior strokes are untouched.
        // (Morphological erosion would destroy 2px-thick lines such as crosshairs.)
        Mat bordered = mask.clone();
        bordered.row(0).setTo(Scalar.all(0));
        bordered.row(bordered.rows() - 1).setTo(Scalar.all(0));
        bordered.col(0).setTo(Scalar.all(0));
        bordered.col(bordered.cols() - 1).setTo(Scalar.all(0));
        // RETR_LIST finds ALL contours including inner ones (COMPOUND shapes).
        Imgproc.findContours(bordered, contours, hierarchy,
                Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        hierarchy.release();
        bordered.release();
        contours.removeIf(c -> Imgproc.contourArea(c) < MIN_AREA);

        // Deduplicate: RETR_LIST returns both inner and outer traces of the same
        // stroked shape. Drop any contour whose centre and area are within
        // tolerance of an already-kept contour.
        List<MatOfPoint> deduped = new ArrayList<>();
        for (MatOfPoint c : contours) {
            Rect   bb    = Imgproc.boundingRect(c);
            double cx    = bb.x + bb.width  / 2.0;
            double cy    = bb.y + bb.height / 2.0;
            double area  = Imgproc.contourArea(c);
            boolean dup  = false;
            for (MatOfPoint kept : deduped) {
                Rect   kb    = Imgproc.boundingRect(kept);
                double kcx   = kb.x + kb.width  / 2.0;
                double kcy   = kb.y + kb.height / 2.0;
                double kArea = Imgproc.contourArea(kept);
                double distFrac = Math.hypot(cx - kcx, cy - kcy)
                        / Math.max(1, Math.max(kb.width, kb.height));
                double areaFrac = Math.abs(area - kArea) / Math.max(1, kArea);
                if (distFrac < 0.05 && areaFrac < 0.10) { dup = true; break; }
            }
            if (!dup) deduped.add(c);
        }
        return deduped;
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