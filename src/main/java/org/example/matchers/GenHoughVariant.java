package org.example.matchers;

import org.example.CfMode;
import org.example.MatcherVariant;

/**
 * All 6 Generalized Hough Transform variants (2 detectors × 3 CF modes).
 */
public enum GenHoughVariant implements MatcherVariant {

    BALLARD           ("GeneralizedHoughBallard",            CfMode.NONE),
    BALLARD_CF_LOOSE  ("GeneralizedHoughBallard_CF_LOOSE",   CfMode.LOOSE),
    BALLARD_CF_TIGHT  ("GeneralizedHoughBallard_CF_TIGHT",   CfMode.TIGHT),

    GUIL              ("GeneralizedHoughGuil",               CfMode.NONE),
    GUIL_CF_LOOSE     ("GeneralizedHoughGuil_CF_LOOSE",      CfMode.LOOSE),
    GUIL_CF_TIGHT     ("GeneralizedHoughGuil_CF_TIGHT",      CfMode.TIGHT);

    private final String variantName;
    private final CfMode cfMode;

    GenHoughVariant(String variantName, CfMode cfMode) {
        this.variantName = variantName;
        this.cfMode      = cfMode;
    }

    @Override public String variantName() { return variantName; }
    public        CfMode    cfMode()      { return cfMode; }

    /** Returns {@code true} if this is a Guil (fine-resolution) variant. */
    public boolean isGuil() {
        return this == GUIL || this == GUIL_CF_LOOSE || this == GUIL_CF_TIGHT;
    }

    @Override public String toString() { return variantName; }
}

