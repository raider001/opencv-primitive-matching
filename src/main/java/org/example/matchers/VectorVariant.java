package org.example.matchers;

import org.example.MatcherVariant;

/**
 * The 3 Vector Matcher variants — one per approximation epsilon level.
 * Colour isolation is handled automatically inside the matcher via
 * {@link org.example.colour.SceneColourClusters}; no CF variants are needed.
 */
public enum VectorVariant implements MatcherVariant {

    /** epsilon = 2% of perimeter — preserves fine detail (star points, arrow tips) */
    VECTOR_STRICT ("VECTOR_STRICT", 0.02),
    /** epsilon = 4% of perimeter — general-purpose baseline */
    VECTOR_NORMAL ("VECTOR_NORMAL", 0.04),
    /** epsilon = 8% of perimeter — tolerates noise and low-resolution scaled scenes */
    VECTOR_LOOSE  ("VECTOR_LOOSE",  0.08);

    private final String variantName;
    private final double epsilonFactor;

    VectorVariant(String variantName, double epsilonFactor) {
        this.variantName   = variantName;
        this.epsilonFactor = epsilonFactor;
    }

    public double epsilonFactor() { return epsilonFactor; }

    @Override public String variantName() { return variantName; }
    @Override public String toString()    { return variantName; }
}
