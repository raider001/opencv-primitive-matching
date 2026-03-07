package org.example.matchers;

import org.example.CfMode;
import org.example.MatcherVariant;
import org.opencv.imgproc.Imgproc;

/**
 * All 11 Contour Shape Matching variants
 * (3 Hu-moment methods × 3 CF modes + 2 CF1 window variants for I1).
 */
public enum ContourVariant implements MatcherVariant {

    CONTOURS_MATCH_I1           ("CONTOURS_MATCH_I1",            Imgproc.CONTOURS_MATCH_I1, CfMode.NONE),
    CONTOURS_MATCH_I1_CF_LOOSE  ("CONTOURS_MATCH_I1_CF_LOOSE",   Imgproc.CONTOURS_MATCH_I1, CfMode.LOOSE),
    CONTOURS_MATCH_I1_CF_TIGHT  ("CONTOURS_MATCH_I1_CF_TIGHT",   Imgproc.CONTOURS_MATCH_I1, CfMode.TIGHT),

    CONTOURS_MATCH_I2           ("CONTOURS_MATCH_I2",            Imgproc.CONTOURS_MATCH_I2, CfMode.NONE),
    CONTOURS_MATCH_I2_CF_LOOSE  ("CONTOURS_MATCH_I2_CF_LOOSE",   Imgproc.CONTOURS_MATCH_I2, CfMode.LOOSE),
    CONTOURS_MATCH_I2_CF_TIGHT  ("CONTOURS_MATCH_I2_CF_TIGHT",   Imgproc.CONTOURS_MATCH_I2, CfMode.TIGHT),

    CONTOURS_MATCH_I3           ("CONTOURS_MATCH_I3",            Imgproc.CONTOURS_MATCH_I3, CfMode.NONE),
    CONTOURS_MATCH_I3_CF_LOOSE  ("CONTOURS_MATCH_I3_CF_LOOSE",   Imgproc.CONTOURS_MATCH_I3, CfMode.LOOSE),
    CONTOURS_MATCH_I3_CF_TIGHT  ("CONTOURS_MATCH_I3_CF_TIGHT",   Imgproc.CONTOURS_MATCH_I3, CfMode.TIGHT),

    /** Colour-First window search using I1, loose tolerance. */
    CONTOURS_MATCH_I1_CF1_LOOSE ("CONTOURS_MATCH_I1_CF1_LOOSE",  Imgproc.CONTOURS_MATCH_I1, CfMode.LOOSE),
    /** Colour-First window search using I1, tight tolerance. */
    CONTOURS_MATCH_I1_CF1_TIGHT ("CONTOURS_MATCH_I1_CF1_TIGHT",  Imgproc.CONTOURS_MATCH_I1, CfMode.TIGHT);

    // -------------------------------------------------------------------------

    private final String variantName;
    private final int    cvMethod;
    private final CfMode cfMode;

    ContourVariant(String variantName, int cvMethod, CfMode cfMode) {
        this.variantName = variantName;
        this.cvMethod    = cvMethod;
        this.cfMode      = cfMode;
    }

    @Override public String variantName() { return variantName; }
    public        int       cvMethod()    { return cvMethod; }
    public        CfMode    cfMode()      { return cfMode; }

    /** Returns {@code true} if this is a CF1 (Colour-First window) variant. */
    public boolean isCf1() {
        return this == CONTOURS_MATCH_I1_CF1_LOOSE || this == CONTOURS_MATCH_I1_CF1_TIGHT;
    }

    @Override public String toString() { return variantName; }
}

