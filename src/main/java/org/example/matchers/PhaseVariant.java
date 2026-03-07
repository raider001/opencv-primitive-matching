package org.example.matchers;

import org.example.CfMode;
import org.example.MatcherVariant;

/**
 * All 6 Phase Correlation variants (2 methods × 3 CF modes).
 */
public enum PhaseVariant implements MatcherVariant {

    PHASE_CORRELATE             ("PHASE_CORRELATE",              false, CfMode.NONE),
    PHASE_CORRELATE_CF_LOOSE    ("PHASE_CORRELATE_CF_LOOSE",     false, CfMode.LOOSE),
    PHASE_CORRELATE_CF_TIGHT    ("PHASE_CORRELATE_CF_TIGHT",     false, CfMode.TIGHT),

    PHASE_CORRELATE_HANNING          ("PHASE_CORRELATE_HANNING",           true, CfMode.NONE),
    PHASE_CORRELATE_HANNING_CF_LOOSE ("PHASE_CORRELATE_HANNING_CF_LOOSE",  true, CfMode.LOOSE),
    PHASE_CORRELATE_HANNING_CF_TIGHT ("PHASE_CORRELATE_HANNING_CF_TIGHT",  true, CfMode.TIGHT);

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

    @Override public String toString() { return variantName; }
}

