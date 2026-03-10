package org.example.colour;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.List;
/**
 * Decomposes a BGR scene into colour-isolated binary masks.
 *
 * Chromatic pixels (sufficient saturation) are grouped by hue peak via inRange
 * windows. Achromatic pixels are split into two brightness bands:
 *   BRIGHT_ACHROMATIC — white, light grey (val >= BRIGHT_VAL_THRESHOLD)
 *   DARK_ACHROMATIC   — black, dark grey, near-black (val < BRIGHT_VAL_THRESHOLD)
 *
 * Keeping these separate means a black shape on a white background produces two
 * distinct masks whose boundaries give the shape contour, regardless of whether
 * the shape is dark-on-light or light-on-dark.
 */
public final class SceneColourClusters {
    public static final int    MAX_CLUSTERS        = 12;
    public static final double HUE_TOLERANCE       = 14.0;
    private static final int   PEAK_MIN_SEPARATION = 18;
    public static final int    MIN_CONTOUR_AREA     = 64;
    private static final double MIN_SAT          = 35.0;
    private static final double MIN_VAL          = 25.0;   // below = near-black
    private static final double BRIGHT_VAL_THRESHOLD = 100.0; // split achromatic here
    private static final int    MIN_PIXEL_COUNT  = 64;
    // -------------------------------------------------------------------------
    public static final class Cluster {
        public final Mat     mask;
        public final double  hue;
        public final boolean achromatic;
        Cluster(Mat mask, double hue, boolean achromatic) {
            this.mask       = mask;
            this.hue        = hue;
            this.achromatic = achromatic;
        }
        public void release() { mask.release(); }
    }
    private SceneColourClusters() {}
    // =========================================================================
    public static List<Cluster> extract(Mat bgrScene) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(bgrScene, hsv, Imgproc.COLOR_BGR2HSV);
        // Bright achromatic: white / light grey (high value, low saturation)
        Mat brightAchromatic = new Mat();
        Core.inRange(hsv,
                new Scalar(0,   0,       BRIGHT_VAL_THRESHOLD),
                new Scalar(179, MIN_SAT, 255),
                brightAchromatic);
        // Dark achromatic: near-black / dark grey (low value regardless of sat)
        Mat darkAchromatic = new Mat();
        Core.inRange(hsv,
                new Scalar(0,   0, 0),
                new Scalar(179, 255, BRIGHT_VAL_THRESHOLD),
                darkAchromatic);
        // Chromatic: sufficient saturation + brightness
        Mat chromaticMask = new Mat();
        Core.inRange(hsv,
                new Scalar(0,   MIN_SAT, MIN_VAL),
                new Scalar(179, 255,     255),
                chromaticMask);
        float[] hueHist = buildHueHistogram(hsv, chromaticMask);
        chromaticMask.release();
        List<Integer> peaks = findPeaks(hueHist, MIN_PIXEL_COUNT);
        List<Cluster> clusters = new ArrayList<>();
        // Chromatic clusters
        for (int peakHue : peaks) {
            if (clusters.size() >= MAX_CLUSTERS) break;
            Mat clusterMask = hueRangeMask(hsv, peakHue, HUE_TOLERANCE);
            if (Core.countNonZero(clusterMask) < MIN_PIXEL_COUNT) {
                clusterMask.release();
                continue;
            }
            clusters.add(new Cluster(clusterMask, peakHue, false));
        }
        // Bright achromatic cluster
        if (Core.countNonZero(brightAchromatic) >= MIN_PIXEL_COUNT) {
            clusters.add(new Cluster(brightAchromatic, Double.NaN, true));
        } else {
            brightAchromatic.release();
        }
        // Dark achromatic cluster
        if (Core.countNonZero(darkAchromatic) >= MIN_PIXEL_COUNT) {
            clusters.add(new Cluster(darkAchromatic, Double.NaN, true));
        } else {
            darkAchromatic.release();
        }
        hsv.release();
        return clusters;
    }
    public static Mat applyMask(Mat bgrScene, Cluster cluster) {
        Mat result = Mat.zeros(bgrScene.size(), bgrScene.type());
        bgrScene.copyTo(result, cluster.mask);
        return result;
    }
    // =========================================================================
    private static float[] buildHueHistogram(Mat hsv, Mat chromaticMask) {
        List<Mat> hsvChannels = new ArrayList<>();
        Core.split(hsv, hsvChannels);
        Mat hueChannel = hsvChannels.get(0);
        Mat hist = new Mat();
        Imgproc.calcHist(List.of(hueChannel), new MatOfInt(0), chromaticMask,
                hist, new MatOfInt(180), new MatOfFloat(0f, 180f));
        float[] result = new float[180];
        for (int i = 0; i < 180; i++) result[i] = (float) hist.get(i, 0)[0];
        hist.release();
        hueChannel.release();
        for (int i = 1; i < hsvChannels.size(); i++) hsvChannels.get(i).release();
        return result;
    }
    private static List<Integer> findPeaks(float[] hist, int minCount) {
        float[] smoothed = new float[180];
        int smoothR = 2;
        for (int i = 0; i < 180; i++) {
            float sum = 0;
            for (int d = -smoothR; d <= smoothR; d++)
                sum += hist[(i + d + 180) % 180];
            smoothed[i] = sum / (2 * smoothR + 1);
        }
        List<int[]> peaks = new ArrayList<>();
        for (int i = 0; i < 180; i++) {
            float prev = smoothed[(i - 1 + 180) % 180];
            float curr = smoothed[i];
            float next = smoothed[(i + 1) % 180];
            if (curr > prev && curr >= next && curr >= minCount)
                peaks.add(new int[]{i, (int) curr});
        }
        peaks.sort((a, b) -> b[1] - a[1]);
        List<Integer> result = new ArrayList<>();
        boolean[] suppressed = new boolean[peaks.size()];
        for (int i = 0; i < peaks.size() && result.size() < MAX_CLUSTERS; i++) {
            if (suppressed[i]) continue;
            result.add(peaks.get(i)[0]);
            for (int j = i + 1; j < peaks.size(); j++) {
                int d = Math.abs(peaks.get(i)[0] - peaks.get(j)[0]);
                d = Math.min(d, 180 - d);
                if (d < PEAK_MIN_SEPARATION) suppressed[j] = true;
            }
        }
        return result;
    }
    private static Mat hueRangeMask(Mat hsv, int peakHue, double tolerance) {
        double lo = peakHue - tolerance;
        double hi = peakHue + tolerance;
        Mat mask = new Mat();
        if (lo < 0) {
            Mat m1 = new Mat(), m2 = new Mat();
            Core.inRange(hsv, new Scalar(0, MIN_SAT, MIN_VAL), new Scalar(hi, 255, 255), m1);
            Core.inRange(hsv, new Scalar(180 + lo, MIN_SAT, MIN_VAL), new Scalar(179, 255, 255), m2);
            Core.bitwise_or(m1, m2, mask); m1.release(); m2.release();
        } else if (hi > 179) {
            Mat m1 = new Mat(), m2 = new Mat();
            Core.inRange(hsv, new Scalar(lo, MIN_SAT, MIN_VAL), new Scalar(179, 255, 255), m1);
            Core.inRange(hsv, new Scalar(0, MIN_SAT, MIN_VAL), new Scalar(hi - 180, 255, 255), m2);
            Core.bitwise_or(m1, m2, mask); m1.release(); m2.release();
        } else {
            Core.inRange(hsv, new Scalar(lo, MIN_SAT, MIN_VAL), new Scalar(hi, 255, 255), mask);
        }
        return mask;
    }
}