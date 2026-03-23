package org.example.matchers.vectormatcher.components;

import org.example.matchers.VectorSignature;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;

import java.util.List;

/**
 * A single colour-edge cluster extracted from the reference image.
 *
 * <p>Holds all contours, bounding boxes, and metadata for one cluster.
 * The "primary" contour is the one with the largest area (used for
 * geometry scoring in Layer 3).
 */
public final class RefCluster {
    public final double hue;
    public final boolean achromatic;
    public final boolean brightAchromatic;
    public final List<MatOfPoint> contours;
    public final double imageArea;
    /** Largest contour area in this cluster — the primary shape boundary. */
    public final double maxContourArea;
    /** Bounding rect of the primary (largest-area) contour — cached. */
    public final Rect primaryBbox;
    /** Bounding rects for ALL contours — cached (OPT-E). */
    public final Rect[] contourBboxes;
    /** Cached maxContourArea / imageArea — avoids redundant division (OPT-S). */
    public final double cachedRefFraction;
    /** Solidity of the primary contour (area / convex hull area). */
    public final double solidity;
    /** Best (highest solidity) VectorSignature for this cluster — set by caller after construction. */
    public VectorSignature bestSig = null;

    private VectorSignature cachedSig = null;

    public RefCluster(double hue, boolean achromatic, boolean brightAchromatic,
                      List<MatOfPoint> contours, double imageArea, double solidity) {
        this.hue = hue;
        this.achromatic = achromatic;
        this.brightAchromatic = brightAchromatic;
        this.contours = contours;
        this.imageArea = imageArea;
        this.solidity = solidity;

        // Pre-compute all contour areas and bounding rects in one pass (OPT-E)
        int size = contours.size();
        this.contourBboxes = new Rect[size];
        double bestArea = 0.0;
        int primaryIdx = 0;
        for (int i = 0; i < size; i++) {
            MatOfPoint c = contours.get(i);
            double a = Imgproc.contourArea(c);
            contourBboxes[i] = Imgproc.boundingRect(c);
            if (a > bestArea) {
                bestArea = a;
                primaryIdx = i;
            }
        }
        this.maxContourArea = bestArea;
        this.primaryBbox = contourBboxes[primaryIdx];
        this.cachedRefFraction = (imageArea > 0) ? maxContourArea / imageArea : 0.0;
    }

    /**
     * Returns the best (highest solidity) VectorSignature at fixed epsilon.
     *
     * @param eps epsilon factor for signature building (fraction of perimeter)
     * @return primary contour's vector signature, cached after first call
     */
    public VectorSignature bestSig(double eps) {
        if (cachedSig != null) return cachedSig;

        VectorSignature best = null;
        double bestSol = -1;
        for (MatOfPoint c : contours) {
            // Keep ref-side normalisedArea as NaN. The normalised-area guards in
            // VectorSignature are scene-side gates; filling both sides triggers an
            // unintended 0.25 cap for valid matches because ref and scene canvases
            // are different sizes (128x128 vs 640x480 in this suite).
            VectorSignature s = VectorSignature.buildFromContour(c, eps, Double.NaN);
            if (s.solidity > bestSol) {
                bestSol = s.solidity;
                best = s;
            }
        }
        cachedSig = (best != null) ? best
                : VectorSignature.build(Mat.zeros(1, 1, CvType.CV_8UC1), eps, Double.NaN);
        return cachedSig;
    }

    /**
     * Fraction of ref image area occupied by this cluster's primary contour.
     *
     * @return cached refFraction value
     */
    public double refFraction() {
        return cachedRefFraction;
    }

    /**
     * Releases native OpenCV memory for all contours.
     */
    public void release() {
        for (MatOfPoint c : contours) {
            c.release();
        }
    }

    /**
     * Returns the bounding rect of the primary (largest-area) contour.
     *
     * @return primary bounding rectangle (cached)
     */
    public Rect primaryBbox() {
        return primaryBbox;
    }
}

