package org.example.matchers;

import org.example.CfMode;
import org.example.MatcherVariant;

/**
 * All 3 SSIM variants (1 base + 2 CF modes).
 */
public enum SsimVariant implements MatcherVariant {

    SSIM           ("SSIM",            CfMode.NONE),
    SSIM_CF_LOOSE  ("SSIM_CF_LOOSE",   CfMode.LOOSE),
    SSIM_CF_TIGHT  ("SSIM_CF_TIGHT",   CfMode.TIGHT);

    private final String variantName;
    private final CfMode cfMode;

    SsimVariant(String variantName, CfMode cfMode) {
        this.variantName = variantName;
        this.cfMode      = cfMode;
    }

    @Override public String variantName() { return variantName; }
    public        CfMode    cfMode()      { return cfMode; }

    @Override public String toString() { return variantName; }
}

