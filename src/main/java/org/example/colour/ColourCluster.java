package org.example.colour;

import org.opencv.core.Mat;

/**
 * A colour-isolated binary mask extracted from a scene or reference image.
 *
 * <p>Chromatic clusters carry a peak {@link #hue} value (OpenCV half-degrees
 * 0–179) and the exclusive valley-based hue window {@link #loBound}/
 * {@link #hiBound} that was used to build the mask.  Achromatic clusters
 * ({@link #achromatic} == {@code true}) carry {@code Double.NaN} for hue and
 * use the sentinel bounds [0, 179].
 *
 * <p>The {@link #mask} is a native OpenCV {@code Mat} — always call
 * {@link #release()} when finished to avoid native memory leaks.
 *
 * <p>This class was extracted from the {@code SceneColourClusters.Cluster}
 * inner class to serve as the shared data type for all
 * {@link SceneColourExtractor} implementations.
 */
public class ColourCluster {

    /** Binary mask (CV_8UC1): 255 where this cluster's pixels are, 0 elsewhere. */
    public final Mat     mask;

    /**
     * Peak hue in OpenCV half-degree units (0–179).
     * {@code Double.NaN} for achromatic clusters.
     */
    public final double  hue;

    /** {@code true} if this is an achromatic (grey/white/black) cluster. */
    public final boolean achromatic;

    /**
     * {@code true} for BRIGHT achromatic (white / light-grey, val ≥ threshold).
     * {@code false} for chromatic clusters or DARK achromatic (black / dark-grey).
     */
    public final boolean brightAchromatic;

    /**
     * Exclusive valley-based hue lower bound for this cluster (0–179).
     * For achromatic clusters this is always 0.
     */
    public final int loBound;

    /**
     * Exclusive valley-based hue upper bound for this cluster (0–179).
     * For achromatic clusters this is always 179.
     */
    public final int hiBound;

    /**
     * Saturation lower bound (inclusive, 0–255) of the S sub-cluster window.
     * {@code 0} when no saturation sub-clustering was applied (full range).
     */
    public final int sLo;

    /**
     * Saturation upper bound (inclusive, 0–255) of the S sub-cluster window.
     * {@code 255} when no saturation sub-clustering was applied (full range).
     */
    public final int sHi;

    /** Full constructor — used by saturation-sub-clustering extractors. */
    public ColourCluster(Mat mask, double hue, boolean achromatic, boolean brightAchromatic,
                         int loBound, int hiBound, int sLo, int sHi) {
        this.mask             = mask;
        this.hue              = hue;
        this.achromatic       = achromatic;
        this.brightAchromatic = brightAchromatic;
        this.loBound          = loBound;
        this.hiBound          = hiBound;
        this.sLo              = sLo;
        this.sHi              = sHi;
    }

    /** Backward-compatible constructor — S range defaults to full span [0, 255]. */
    public ColourCluster(Mat mask, double hue, boolean achromatic, boolean brightAchromatic,
                         int loBound, int hiBound) {
        this(mask, hue, achromatic, brightAchromatic, loBound, hiBound, 0, 255);
    }

    /** Convenience constructor for achromatic clusters (no meaningful hue bounds). */
    public ColourCluster(Mat mask, double hue, boolean achromatic, boolean brightAchromatic) {
        this(mask, hue, achromatic, brightAchromatic, 0, 179, 0, 255);
    }

    /** Release the underlying native {@link Mat}. Must be called exactly once. */
    public void release() { mask.release(); }
}
