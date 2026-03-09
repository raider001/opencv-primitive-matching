package org.example.matchers;

import org.example.colour.CfMode;
import org.example.MatcherVariant;

/** All 3 Fourier Shape Descriptor variants (1 base + 2 CF modes). */
public enum FourierShapeVariant implements MatcherVariant {

    FOURIER_SHAPE           ("FOURIER_SHAPE",            CfMode.NONE),
    FOURIER_SHAPE_CF_LOOSE  ("FOURIER_SHAPE_CF_LOOSE",   CfMode.LOOSE),
    FOURIER_SHAPE_CF_TIGHT  ("FOURIER_SHAPE_CF_TIGHT",   CfMode.TIGHT);

    private final String variantName;
    private final CfMode cfMode;

    FourierShapeVariant(String variantName, CfMode cfMode) {
        this.variantName = variantName;
        this.cfMode      = cfMode;
    }

    @Override public String variantName() { return variantName; }
    public        CfMode    cfMode()      { return cfMode; }

    @Override public String toString() { return variantName; }
}

