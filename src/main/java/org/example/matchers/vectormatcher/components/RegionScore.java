package org.example.matchers.vectormatcher.components;

/**
 * Result of the three-layer scoring in {@link RegionScorer}.
 *
 * <p>All values are in [0, 1].
 *
 * @param combined   weighted combination of the three layer scores
 * @param countScore Layer 1 — boundary count match (before weighting)
 * @param matchScore Layer 2 — structural coherence (before weighting)
 * @param geomScore  Layer 3 — primary geometry via VectorSignature (before weighting)
 */
public record RegionScore(double combined, double countScore, double matchScore, double geomScore) {

    /** Convenience for transition code that still expects {@code double[4]}. */
    public double[] toArray() {
        return new double[]{combined, countScore, matchScore, geomScore};
    }
}

