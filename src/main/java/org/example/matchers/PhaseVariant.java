package org.example.matchers;

import org.example.CfMode;
import org.example.MatcherVariant;

/**
 * All 8 Phase Correlation variants (2 methods × 3 CF modes + 2 CF1 window variants for PHASE_CORRELATE).
 */
public enum PhaseVariant implements MatcherVariant {

    PHASE_CORRELATE             ("PHASE_CORRELATE",              false, CfMode.NONE),
    PHASE_CORRELATE_CF_LOOSE    ("PHASE_CORRELATE_CF_LOOSE",     false, CfMode.LOOSE),
    PHASE_CORRELATE_CF_TIGHT    ("PHASE_CORRELATE_CF_TIGHT",     false, CfMode.TIGHT),

    PHASE_CORRELATE_HANNING          ("PHASE_CORRELATE_HANNING",           true, CfMode.NONE),
    PHASE_CORRELATE_HANNING_CF_LOOSE ("PHASE_CORRELATE_HANNING_CF_LOOSE",  true, CfMode.LOOSE),
    PHASE_CORRELATE_HANNING_CF_TIGHT ("PHASE_CORRELATE_HANNING_CF_TIGHT",  true, CfMode.TIGHT),

    /** Colour-First window search using PHASE_CORRELATE (plain), loose tolerance. */
    PHASE_CORRELATE_CF1_LOOSE   ("PHASE_CORRELATE_CF1_LOOSE",   false, CfMode.LOOSE),
    /** Colour-First window search using PHASE_CORRELATE (plain), tight tolerance. */
    PHASE_CORRELATE_CF1_TIGHT   ("PHASE_CORRELATE_CF1_TIGHT",   false, CfMode.TIGHT);

    private final String  variantName;
    private final boolean hanning;
    private final CfMode  cfMode;

    PhaseVariant(String variantName, boolean hanning, CfMode cfMode) {
        this.variantName = variantName;
        this.hanning     = hanning;
        this.cfMode      = cfMode;
    }

    @Override public String  variantName() { return variantName; }
    public        boolean    hanning()     { return hanning; }
    public        CfMode     cfMode()      { return cfMode; }

    /** Returns {@code true} if this is a CF1 (Colour-First window) variant. */
    public boolean isCf1() {
        return this == PHASE_CORRELATE_CF1_LOOSE || this == PHASE_CORRELATE_CF1_TIGHT;
    }

    @Override public String toString() { return variantName; }
}

