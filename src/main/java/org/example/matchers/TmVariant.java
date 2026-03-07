package org.example.matchers;

import org.example.CfMode;
import org.example.MatcherVariant;
import org.opencv.imgproc.Imgproc;

/**
 * All 20 Template Matching variants (6 base × 3 CF modes + 2 CF1 window variants).
 *
 * <p>Naming: {@code <BASE>} for base, {@code <BASE>_CF_LOOSE} / {@code <BASE>_CF_TIGHT}
 * for colour-pre-filtered variants, and the two special {@code CF1} window-search variants.
 */
public enum TmVariant implements MatcherVariant {

    TM_SQDIFF             ("TM_SQDIFF",              Imgproc.TM_SQDIFF,         true,  CfMode.NONE),
    TM_SQDIFF_CF_LOOSE    ("TM_SQDIFF_CF_LOOSE",     Imgproc.TM_SQDIFF,         true,  CfMode.LOOSE),
    TM_SQDIFF_CF_TIGHT    ("TM_SQDIFF_CF_TIGHT",     Imgproc.TM_SQDIFF,         true,  CfMode.TIGHT),

    TM_SQDIFF_NORMED      ("TM_SQDIFF_NORMED",       Imgproc.TM_SQDIFF_NORMED,  true,  CfMode.NONE),
    TM_SQDIFF_NORMED_CF_LOOSE("TM_SQDIFF_NORMED_CF_LOOSE", Imgproc.TM_SQDIFF_NORMED, true, CfMode.LOOSE),
    TM_SQDIFF_NORMED_CF_TIGHT("TM_SQDIFF_NORMED_CF_TIGHT", Imgproc.TM_SQDIFF_NORMED, true, CfMode.TIGHT),

    TM_CCORR              ("TM_CCORR",               Imgproc.TM_CCORR,          false, CfMode.NONE),
    TM_CCORR_CF_LOOSE     ("TM_CCORR_CF_LOOSE",      Imgproc.TM_CCORR,          false, CfMode.LOOSE),
    TM_CCORR_CF_TIGHT     ("TM_CCORR_CF_TIGHT",      Imgproc.TM_CCORR,          false, CfMode.TIGHT),

    TM_CCORR_NORMED       ("TM_CCORR_NORMED",        Imgproc.TM_CCORR_NORMED,   false, CfMode.NONE),
    TM_CCORR_NORMED_CF_LOOSE("TM_CCORR_NORMED_CF_LOOSE", Imgproc.TM_CCORR_NORMED, false, CfMode.LOOSE),
    TM_CCORR_NORMED_CF_TIGHT("TM_CCORR_NORMED_CF_TIGHT", Imgproc.TM_CCORR_NORMED, false, CfMode.TIGHT),

    TM_CCOEFF             ("TM_CCOEFF",              Imgproc.TM_CCOEFF,         false, CfMode.NONE),
    TM_CCOEFF_CF_LOOSE    ("TM_CCOEFF_CF_LOOSE",     Imgproc.TM_CCOEFF,         false, CfMode.LOOSE),
    TM_CCOEFF_CF_TIGHT    ("TM_CCOEFF_CF_TIGHT",     Imgproc.TM_CCOEFF,         false, CfMode.TIGHT),

    TM_CCOEFF_NORMED      ("TM_CCOEFF_NORMED",       Imgproc.TM_CCOEFF_NORMED,  false, CfMode.NONE),
    TM_CCOEFF_NORMED_CF_LOOSE("TM_CCOEFF_NORMED_CF_LOOSE", Imgproc.TM_CCOEFF_NORMED, false, CfMode.LOOSE),
    TM_CCOEFF_NORMED_CF_TIGHT("TM_CCOEFF_NORMED_CF_TIGHT", Imgproc.TM_CCOEFF_NORMED, false, CfMode.TIGHT),

    /** Colour-First window search using TM_CCOEFF_NORMED, loose tolerance. */
    TM_CCOEFF_NORMED_CF1_LOOSE("TM_CCOEFF_NORMED_CF1_LOOSE", Imgproc.TM_CCOEFF_NORMED, false, CfMode.LOOSE),

    /** Colour-First window search using TM_CCOEFF_NORMED, tight tolerance. */
    TM_CCOEFF_NORMED_CF1_TIGHT("TM_CCOEFF_NORMED_CF1_TIGHT", Imgproc.TM_CCOEFF_NORMED, false, CfMode.TIGHT);

    // -------------------------------------------------------------------------

    private final String  variantName;
    private final int     cvFlag;
    private final boolean lowerIsBetter;
    private final CfMode  cfMode;

    TmVariant(String variantName, int cvFlag, boolean lowerIsBetter, CfMode cfMode) {
        this.variantName   = variantName;
        this.cvFlag        = cvFlag;
        this.lowerIsBetter = lowerIsBetter;
        this.cfMode        = cfMode;
    }

    @Override public String  variantName()    { return variantName; }
    public        int        cvFlag()         { return cvFlag; }
    public        boolean    lowerIsBetter()  { return lowerIsBetter; }
    public        CfMode     cfMode()         { return cfMode; }

    /** Returns {@code true} if this is a CF1 window-search variant. */
    public boolean isCf1() {
        return this == TM_CCOEFF_NORMED_CF1_LOOSE || this == TM_CCOEFF_NORMED_CF1_TIGHT;
    }

    /** Returns {@code true} if OpenCV's native masked matchTemplate supports this flag. */
    public boolean supportsNativeMask() {
        return cvFlag == Imgproc.TM_SQDIFF || cvFlag == Imgproc.TM_CCORR_NORMED;
    }

    /**
     * Returns the base variant (CF mode = NONE) for this variant,
     * e.g. {@code TM_CCOEFF_NORMED_CF_LOOSE} → {@code TM_CCOEFF_NORMED}.
     */
    public TmVariant base() {
        if (cfMode == CfMode.NONE) return this;
        for (TmVariant v : values()) {
            if (v.cvFlag == this.cvFlag && v.cfMode == CfMode.NONE && !v.isCf1()) return v;
        }
        return this;
    }

    @Override public String toString() { return variantName; }
}

