package org.example.matchers;

import org.example.CfMode;
import org.example.MatcherVariant;
import org.opencv.imgproc.Imgproc;

/**
 * All 6 Chamfer Distance Matching variants (2 distance types x 3 CF modes).
 */
public enum ChamferVariant implements MatcherVariant {

    CHAMFER_L1           ("CHAMFER_L1",          Imgproc.DIST_L1, CfMode.NONE),
    CHAMFER_L1_CF_LOOSE  ("CHAMFER_L1_CF_LOOSE", Imgproc.DIST_L1, CfMode.LOOSE),
    CHAMFER_L1_CF_TIGHT  ("CHAMFER_L1_CF_TIGHT", Imgproc.DIST_L1, CfMode.TIGHT),

    CHAMFER_L2           ("CHAMFER_L2",          Imgproc.DIST_L2, CfMode.NONE),
    CHAMFER_L2_CF_LOOSE  ("CHAMFER_L2_CF_LOOSE", Imgproc.DIST_L2, CfMode.LOOSE),
    CHAMFER_L2_CF_TIGHT  ("CHAMFER_L2_CF_TIGHT", Imgproc.DIST_L2, CfMode.TIGHT);

    private final String variantName;
    private final int    distType;
    private final CfMode cfMode;

    ChamferVariant(String variantName, int distType, CfMode cfMode) {
        this.variantName = variantName;
        this.distType    = distType;
        this.cfMode      = cfMode;
    }

    @Override public String variantName() { return variantName; }
    public        int       distType()    { return distType; }
    public        CfMode    cfMode()      { return cfMode; }

    @Override public String toString() { return variantName; }
}


