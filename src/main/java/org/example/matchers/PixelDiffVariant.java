package org.example.matchers;

import org.example.CfMode;
import org.example.MatcherVariant;

/**
 * All 3 Pixel Difference variants (1 base + 2 CF modes).
 */
public enum PixelDiffVariant implements MatcherVariant {

    PIXEL_DIFF           ("PIXEL_DIFF",            CfMode.NONE),
    PIXEL_DIFF_CF_LOOSE  ("PIXEL_DIFF_CF_LOOSE",   CfMode.LOOSE),
    PIXEL_DIFF_CF_TIGHT  ("PIXEL_DIFF_CF_TIGHT",   CfMode.TIGHT);

    private final String variantName;
    private final CfMode cfMode;

    PixelDiffVariant(String variantName, CfMode cfMode) {
        this.variantName = variantName;
        this.cfMode      = cfMode;
    }

    @Override public String variantName() { return variantName; }
    public        CfMode    cfMode()      { return cfMode; }

    @Override public String toString() { return variantName; }
}

