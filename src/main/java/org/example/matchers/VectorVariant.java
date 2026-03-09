package org.example.matchers;

import org.example.colour.CfMode;
import org.example.MatcherVariant;

/**
 * All 9 Vector Matcher variants: 3 approximation epsilon levels x 3 CF modes.
 *
 * <ul>
 *   <li><b>STRICT</b>  — epsilon = 2% of perimeter: preserves fine detail
 *       (star points, chevron notches, arrow tips)</li>
 *   <li><b>NORMAL</b>  — epsilon = 4% of perimeter: general-purpose baseline</li>
 *   <li><b>LOOSE</b>   — epsilon = 8% of perimeter: tolerates noise and
 *       low-resolution scaled scenes</li>
 * </ul>
 */
public enum VectorVariant implements MatcherVariant {

    VECTOR_STRICT           ("VECTOR_STRICT",            0.02, CfMode.NONE),
    VECTOR_STRICT_CF_LOOSE  ("VECTOR_STRICT_CF_LOOSE",   0.02, CfMode.LOOSE),
    VECTOR_STRICT_CF_TIGHT  ("VECTOR_STRICT_CF_TIGHT",   0.02, CfMode.TIGHT),

    VECTOR_NORMAL           ("VECTOR_NORMAL",             0.04, CfMode.NONE),
    VECTOR_NORMAL_CF_LOOSE  ("VECTOR_NORMAL_CF_LOOSE",    0.04, CfMode.LOOSE),
    VECTOR_NORMAL_CF_TIGHT  ("VECTOR_NORMAL_CF_TIGHT",    0.04, CfMode.TIGHT),

    VECTOR_LOOSE            ("VECTOR_LOOSE",              0.08, CfMode.NONE),
    VECTOR_LOOSE_CF_LOOSE   ("VECTOR_LOOSE_CF_LOOSE",     0.08, CfMode.LOOSE),
    VECTOR_LOOSE_CF_TIGHT   ("VECTOR_LOOSE_CF_TIGHT",     0.08, CfMode.TIGHT);

    private final String variantName;
    private final double epsilonFactor;
    private final CfMode cfMode;

    VectorVariant(String variantName, double epsilonFactor, CfMode cfMode) {
        this.variantName   = variantName;
        this.epsilonFactor = epsilonFactor;
        this.cfMode        = cfMode;
    }

    /** The fraction of a contour's perimeter used as the approxPolyDP epsilon. */
    public double epsilonFactor() { return epsilonFactor; }

    /** The colour-filter mode for this variant. */
    public CfMode cfMode() { return cfMode; }

    @Override public String variantName() { return variantName; }
    @Override public String toString()    { return variantName; }
}

