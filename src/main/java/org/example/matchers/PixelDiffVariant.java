package org.example.matchers;

import org.example.colour.CfMode;
import org.example.MatcherVariant;

/**
 * All 5 Pixel Difference variants (1 base + 2 CF modes + 2 CF1 window variants).
 */
public enum PixelDiffVariant implements MatcherVariant {

    PIXEL_DIFF           ("PIXEL_DIFF",            CfMode.NONE),
    PIXEL_DIFF_CF_LOOSE  ("PIXEL_DIFF_CF_LOOSE",   CfMode.LOOSE),
    PIXEL_DIFF_CF_TIGHT  ("PIXEL_DIFF_CF_TIGHT",   CfMode.TIGHT),

    /** Colour-First window search using PIXEL_DIFF, loose tolerance. */
    PIXEL_DIFF_CF1_LOOSE ("PIXEL_DIFF_CF1_LOOSE",  CfMode.LOOSE),
    /** Colour-First window search using PIXEL_DIFF, tight tolerance. */
    PIXEL_DIFF_CF1_TIGHT ("PIXEL_DIFF_CF1_TIGHT",  CfMode.TIGHT);

    private final String variantName;
    private final CfMode cfMode;

    PixelDiffVariant(String variantName, CfMode cfMode) {
        this.variantName = variantName;
        this.cfMode      = cfMode;
    }

    @Override public String variantName() { return variantName; }
    public        CfMode    cfMode()      { return cfMode; }

    /** Returns {@code true} if this is a CF1 (Colour-First window) variant. */
    public boolean isCf1() {
        return this == PIXEL_DIFF_CF1_LOOSE || this == PIXEL_DIFF_CF1_TIGHT;
    }

    @Override public String toString() { return variantName; }
}

