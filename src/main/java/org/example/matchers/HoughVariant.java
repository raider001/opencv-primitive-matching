package org.example.matchers;

import org.example.CfMode;
import org.example.MatcherVariant;

/**
 * All 8 Hough Transform variants (2 detectors × 3 CF modes + 2 CF1 window variants for HoughLinesP).
 */
public enum HoughVariant implements MatcherVariant {

    HOUGH_LINES_P           ("HoughLinesP",              CfMode.NONE),
    HOUGH_LINES_P_CF_LOOSE  ("HoughLinesP_CF_LOOSE",     CfMode.LOOSE),
    HOUGH_LINES_P_CF_TIGHT  ("HoughLinesP_CF_TIGHT",     CfMode.TIGHT),

    HOUGH_CIRCLES           ("HoughCircles",              CfMode.NONE),
    HOUGH_CIRCLES_CF_LOOSE  ("HoughCircles_CF_LOOSE",     CfMode.LOOSE),
    HOUGH_CIRCLES_CF_TIGHT  ("HoughCircles_CF_TIGHT",     CfMode.TIGHT),

    /** Colour-First window search using HoughLinesP, loose tolerance. */
    HOUGH_LINES_P_CF1_LOOSE ("HoughLinesP_CF1_LOOSE",    CfMode.LOOSE),
    /** Colour-First window search using HoughLinesP, tight tolerance. */
    HOUGH_LINES_P_CF1_TIGHT ("HoughLinesP_CF1_TIGHT",    CfMode.TIGHT);

    private final String variantName;
    private final CfMode cfMode;

    HoughVariant(String variantName, CfMode cfMode) {
        this.variantName = variantName;
        this.cfMode      = cfMode;
    }

    @Override public String variantName() { return variantName; }
    public        CfMode    cfMode()      { return cfMode; }

    /** Returns {@code true} if this is a circles-based variant. */
    public boolean isCircles() {
        return this == HOUGH_CIRCLES || this == HOUGH_CIRCLES_CF_LOOSE || this == HOUGH_CIRCLES_CF_TIGHT;
    }

    /** Returns {@code true} if this is a CF1 (Colour-First window) variant. */
    public boolean isCf1() {
        return this == HOUGH_LINES_P_CF1_LOOSE || this == HOUGH_LINES_P_CF1_TIGHT;
    }

    @Override public String toString() { return variantName; }
}

