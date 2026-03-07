package org.example.matchers;

import org.example.CfMode;
import org.example.MatcherVariant;

/**
 * All 8 Generalized Hough Transform variants (2 detectors × 3 CF modes + 2 CF1 window variants for Ballard).
 */
public enum GenHoughVariant implements MatcherVariant {

    BALLARD           ("GeneralizedHoughBallard",              CfMode.NONE),
    BALLARD_CF_LOOSE  ("GeneralizedHoughBallard_CF_LOOSE",     CfMode.LOOSE),
    BALLARD_CF_TIGHT  ("GeneralizedHoughBallard_CF_TIGHT",     CfMode.TIGHT),

    GUIL              ("GeneralizedHoughGuil",                 CfMode.NONE),
    GUIL_CF_LOOSE     ("GeneralizedHoughGuil_CF_LOOSE",        CfMode.LOOSE),
    GUIL_CF_TIGHT     ("GeneralizedHoughGuil_CF_TIGHT",        CfMode.TIGHT),

    /** Colour-First window search using Ballard, loose tolerance. */
    BALLARD_CF1_LOOSE ("GeneralizedHoughBallard_CF1_LOOSE",    CfMode.LOOSE),
    /** Colour-First window search using Ballard, tight tolerance. */
    BALLARD_CF1_TIGHT ("GeneralizedHoughBallard_CF1_TIGHT",    CfMode.TIGHT);

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

    /** Returns {@code true} if this is a CF1 (Colour-First window) variant. */
    public boolean isCf1() {
        return this == BALLARD_CF1_LOOSE || this == BALLARD_CF1_TIGHT;
    }

    @Override public String toString() { return variantName; }
}

