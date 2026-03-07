package org.example.matchers;

import org.example.CfMode;
import org.example.MatcherVariant;

/**
 * All feature detector variants (up to 5 detectors × 3 CF modes = up to 15 variants,
 * plus 2 CF1 window variants for SIFT).
 *
 * <p>SIFT may be absent in some OpenCV builds — {@link FeatureMatcher} checks at
 * runtime and skips it if unavailable.  All other variants are always attempted.
 */
public enum FeatureVariant implements MatcherVariant {

    SIFT              ("SIFT",              false, CfMode.NONE),
    SIFT_CF_LOOSE     ("SIFT_CF_LOOSE",     false, CfMode.LOOSE),
    SIFT_CF_TIGHT     ("SIFT_CF_TIGHT",     false, CfMode.TIGHT),

    ORB               ("ORB",               true,  CfMode.NONE),
    ORB_CF_LOOSE      ("ORB_CF_LOOSE",      true,  CfMode.LOOSE),
    ORB_CF_TIGHT      ("ORB_CF_TIGHT",      true,  CfMode.TIGHT),

    AKAZE             ("AKAZE",             true,  CfMode.NONE),
    AKAZE_CF_LOOSE    ("AKAZE_CF_LOOSE",    true,  CfMode.LOOSE),
    AKAZE_CF_TIGHT    ("AKAZE_CF_TIGHT",    true,  CfMode.TIGHT),

    BRISK             ("BRISK",             true,  CfMode.NONE),
    BRISK_CF_LOOSE    ("BRISK_CF_LOOSE",    true,  CfMode.LOOSE),
    BRISK_CF_TIGHT    ("BRISK_CF_TIGHT",    true,  CfMode.TIGHT),

    KAZE              ("KAZE",              false, CfMode.NONE),
    KAZE_CF_LOOSE     ("KAZE_CF_LOOSE",     false, CfMode.LOOSE),
    KAZE_CF_TIGHT     ("KAZE_CF_TIGHT",     false, CfMode.TIGHT),

    /** Colour-First window search using SIFT, loose tolerance. */
    SIFT_CF1_LOOSE    ("SIFT_CF1_LOOSE",    false, CfMode.LOOSE),
    /** Colour-First window search using SIFT, tight tolerance. */
    SIFT_CF1_TIGHT    ("SIFT_CF1_TIGHT",    false, CfMode.TIGHT);

    // -------------------------------------------------------------------------

    private final String  variantName;
    /** {@code true} for binary descriptors (Hamming distance); {@code false} for float (L2). */
    private final boolean binary;
    private final CfMode  cfMode;

    FeatureVariant(String variantName, boolean binary, CfMode cfMode) {
        this.variantName = variantName;
        this.binary      = binary;
        this.cfMode      = cfMode;
    }

    @Override public String  variantName() { return variantName; }
    public        boolean    binary()      { return binary; }
    public        CfMode     cfMode()      { return cfMode; }

    /** Returns {@code true} if this is a CF1 (Colour-First window) variant. */
    public boolean isCf1() {
        return this == SIFT_CF1_LOOSE || this == SIFT_CF1_TIGHT;
    }

    /** Returns the base detector name, e.g. {@code "SIFT"} for all SIFT-based variants. */
    public String detectorName() {
        return variantName.replaceAll("_CF1?_(LOOSE|TIGHT)$", "");
    }

    @Override public String toString() { return variantName; }
}

