package org.example.matchers;

import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;

/**
 * One contour from the scene with its cluster metadata and pre-built signature.
 *
 * @param contour            the OpenCV contour
 * @param clusterIdx        index of the colour cluster this contour belongs to
 * @param achromatic        true if the cluster is achromatic (saturation < 30)
 * @param brightAchromatic  true if the cluster is bright achromatic (V > 200)
 * @param clusterHue        mean hue of the cluster (0–180 OpenCV scale)
 * @param sig               pre-built VectorSignature (null until buildSignatures() is called)
 * @param bbox              bounding rectangle (cached for performance)
 * @param area              contour area (cached for performance)
 */
public record SceneContourEntry(
        MatOfPoint contour,
        int clusterIdx,
        boolean achromatic,
        boolean brightAchromatic,
        double clusterHue,
        VectorSignature sig,
        Rect bbox,
        double area) {
}

