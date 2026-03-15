package org.example.colour;

import org.opencv.core.Mat;

import java.util.List;

/**
 * Strategy interface for decomposing a BGR scene (or reference image) into a
 * list of colour-isolated binary masks ({@link ColourCluster}).
 *
 * <p>Two extraction modes are required:
 * <ul>
 *   <li>{@link #extractFromBorderPixels(Mat)} — restricts the hue histogram to
 *       morphological-gradient border pixels before clustering.  This is the
 *       primary production path used by {@code SceneDescriptor.build()} because
 *       it avoids large solid interiors dominating the histogram.</li>
 *   <li>{@link #extract(Mat)} — performs full-pixel chromatic clustering.  Used
 *       as a fallback (e.g. when border extraction yields no clusters) and for
 *       building combined chromatic masks.</li>
 * </ul>
 *
 * <p>Callers are responsible for releasing every returned {@link ColourCluster}
 * (via {@link ColourCluster#release()}) once they are done with the native Mats.
 *
 * <p>Implementations must be <em>thread-safe</em>: the benchmark runner calls
 * matchers in a {@code ForkJoinPool} and a shared extractor instance may be
 * invoked concurrently from multiple threads.
 *
 * @see SceneColourClusters#INSTANCE
 * @see ExperimentalSceneColourClusters#INSTANCE
 */
public interface SceneColourExtractor {

    /**
     * Extracts colour clusters by analysing only the morphological-gradient
     * border pixels of each chromatic region.
     *
     * <p>This prevents large filled-colour areas from swamping the hue histogram
     * and keeps cluster discovery edge-aligned — consistent with how reference
     * clusters are identified.
     *
     * @param bgrScene BGR image to decompose (not modified)
     * @return ordered list of {@link ColourCluster}s; never {@code null}, may be empty
     */
    List<ColourCluster> extractFromBorderPixels(Mat bgrScene);

    /**
     * Extracts colour clusters from all pixels in the image (no border filter).
     *
     * <p>Used as a fallback when border extraction yields no clusters, and for
     * building filled chromatic masks (e.g. the {@code combinedChromaticMask} in
     * {@code SceneDescriptor}).
     *
     * @param bgrScene BGR image to decompose (not modified)
     * @return ordered list of {@link ColourCluster}s; never {@code null}, may be empty
     */
    List<ColourCluster> extract(Mat bgrScene);
}


