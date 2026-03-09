package org.example.matchers;

import org.example.colour.CfMode;
import org.example.MatcherVariant;
import org.opencv.imgproc.Imgproc;

/**
 * All 14 Histogram Comparison variants (4 methods × 3 CF modes + 2 CF1 window variants for CORREL).
 */
public enum HistVariant implements MatcherVariant {

    HISTCMP_CORREL             ("HISTCMP_CORREL",              Imgproc.HISTCMP_CORREL,        false, CfMode.NONE),
    HISTCMP_CORREL_CF_LOOSE    ("HISTCMP_CORREL_CF_LOOSE",     Imgproc.HISTCMP_CORREL,        false, CfMode.LOOSE),
    HISTCMP_CORREL_CF_TIGHT    ("HISTCMP_CORREL_CF_TIGHT",     Imgproc.HISTCMP_CORREL,        false, CfMode.TIGHT),

    HISTCMP_CHISQR             ("HISTCMP_CHISQR",              Imgproc.HISTCMP_CHISQR,        true,  CfMode.NONE),
    HISTCMP_CHISQR_CF_LOOSE    ("HISTCMP_CHISQR_CF_LOOSE",     Imgproc.HISTCMP_CHISQR,        true,  CfMode.LOOSE),
    HISTCMP_CHISQR_CF_TIGHT    ("HISTCMP_CHISQR_CF_TIGHT",     Imgproc.HISTCMP_CHISQR,        true,  CfMode.TIGHT),

    HISTCMP_INTERSECT          ("HISTCMP_INTERSECT",           Imgproc.HISTCMP_INTERSECT,     false, CfMode.NONE),
    HISTCMP_INTERSECT_CF_LOOSE ("HISTCMP_INTERSECT_CF_LOOSE",  Imgproc.HISTCMP_INTERSECT,     false, CfMode.LOOSE),
    HISTCMP_INTERSECT_CF_TIGHT ("HISTCMP_INTERSECT_CF_TIGHT",  Imgproc.HISTCMP_INTERSECT,     false, CfMode.TIGHT),

    HISTCMP_BHATTACHARYYA           ("HISTCMP_BHATTACHARYYA",            Imgproc.HISTCMP_BHATTACHARYYA, true,  CfMode.NONE),
    HISTCMP_BHATTACHARYYA_CF_LOOSE  ("HISTCMP_BHATTACHARYYA_CF_LOOSE",   Imgproc.HISTCMP_BHATTACHARYYA, true,  CfMode.LOOSE),
    HISTCMP_BHATTACHARYYA_CF_TIGHT  ("HISTCMP_BHATTACHARYYA_CF_TIGHT",   Imgproc.HISTCMP_BHATTACHARYYA, true,  CfMode.TIGHT),

    /** Colour-First window search using HISTCMP_CORREL, loose tolerance. */
    HISTCMP_CORREL_CF1_LOOSE   ("HISTCMP_CORREL_CF1_LOOSE",   Imgproc.HISTCMP_CORREL,        false, CfMode.LOOSE),
    /** Colour-First window search using HISTCMP_CORREL, tight tolerance. */
    HISTCMP_CORREL_CF1_TIGHT   ("HISTCMP_CORREL_CF1_TIGHT",   Imgproc.HISTCMP_CORREL,        false, CfMode.TIGHT);

    // -------------------------------------------------------------------------

    private final String  variantName;
    private final int     cvMethod;
    private final boolean lowerIsBetter;
    private final CfMode  cfMode;

    HistVariant(String variantName, int cvMethod, boolean lowerIsBetter, CfMode cfMode) {
        this.variantName   = variantName;
        this.cvMethod      = cvMethod;
        this.lowerIsBetter = lowerIsBetter;
        this.cfMode        = cfMode;
    }

    @Override public String  variantName()   { return variantName; }
    public        int        cvMethod()      { return cvMethod; }
    public        boolean    lowerIsBetter() { return lowerIsBetter; }
    public        CfMode     cfMode()        { return cfMode; }

    /** Returns {@code true} if this is a CF1 (Colour-First window) variant. */
    public boolean isCf1() {
        return this == HISTCMP_CORREL_CF1_LOOSE || this == HISTCMP_CORREL_CF1_TIGHT;
    }

    @Override public String toString() { return variantName; }
}

