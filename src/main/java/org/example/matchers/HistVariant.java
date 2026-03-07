package org.example.matchers;

import org.example.CfMode;
import org.example.MatcherVariant;
import org.opencv.imgproc.Imgproc;

/**
 * All 12 Histogram Comparison variants (4 methods × 3 CF modes).
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
    HISTCMP_BHATTACHARYYA_CF_TIGHT  ("HISTCMP_BHATTACHARYYA_CF_TIGHT",   Imgproc.HISTCMP_BHATTACHARYYA, true,  CfMode.TIGHT);

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

    @Override public String toString() { return variantName; }
}

