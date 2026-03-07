package org.example;

/**
 * All fixed scene variant labels used in Category A, B, and C scenes.
 *
 * <p>Category D (negative) scenes use dynamically generated labels and are not
 * represented here — use {@link SceneCategory#D_NEGATIVE} to identify them.
 *
 * <p>The {@link #label()} method returns the exact string stored in
 * {@link SceneEntry#variantLabel()} and written to the JSON sidecar on disk.
 * This enum should be used wherever tests filter by variant (e.g. in
 * {@code sceneFilter()} overrides) to avoid raw string comparisons.
 */
public enum SceneVariant {

    // -------------------------------------------------------------------------
    // A_CLEAN — one per background type
    // -------------------------------------------------------------------------
    CLEAN_BG_SOLID_BLACK        ("clean_bg_solid_black",         SceneCategory.A_CLEAN),
    CLEAN_BG_NOISE_LIGHT        ("clean_bg_noise_light",         SceneCategory.A_CLEAN),
    CLEAN_BG_GRADIENT_H_COLOUR  ("clean_bg_gradient_h_colour",   SceneCategory.A_CLEAN),
    CLEAN_BG_RANDOM_MIXED       ("clean_bg_random_mixed",        SceneCategory.A_CLEAN),

    // -------------------------------------------------------------------------
    // B_TRANSFORMED — scale variants
    // -------------------------------------------------------------------------
    SCALE_0_50   ("scale_0.50",      SceneCategory.B_TRANSFORMED),
    SCALE_0_75   ("scale_0.75",      SceneCategory.B_TRANSFORMED),
    SCALE_1_25   ("scale_1.25",      SceneCategory.B_TRANSFORMED),
    SCALE_1_50   ("scale_1.50",      SceneCategory.B_TRANSFORMED),
    SCALE_2_00   ("scale_2.00",      SceneCategory.B_TRANSFORMED),

    // B_TRANSFORMED — rotation variants
    ROT_15       ("rot_15",          SceneCategory.B_TRANSFORMED),
    ROT_30       ("rot_30",          SceneCategory.B_TRANSFORMED),
    ROT_45       ("rot_45",          SceneCategory.B_TRANSFORMED),
    ROT_90       ("rot_90",          SceneCategory.B_TRANSFORMED),
    ROT_180      ("rot_180",         SceneCategory.B_TRANSFORMED),

    // B_TRANSFORMED — combined scale + rotation
    SCALE1_5_ROT45  ("scale1.5_rot45",   SceneCategory.B_TRANSFORMED),
    SCALE0_75_ROT30 ("scale0.75_rot30",  SceneCategory.B_TRANSFORMED),

    // B_TRANSFORMED — offset variants
    OFFSET_TOPLEFT   ("offset_topleft",   SceneCategory.B_TRANSFORMED),
    OFFSET_BOTRIGHT  ("offset_botright",  SceneCategory.B_TRANSFORMED),
    OFFSET_RANDOM42  ("offset_random42",  SceneCategory.B_TRANSFORMED),

    // -------------------------------------------------------------------------
    // C_DEGRADED — noise
    // -------------------------------------------------------------------------
    NOISE_S10    ("noise_s10",       SceneCategory.C_DEGRADED),
    NOISE_S25    ("noise_s25",       SceneCategory.C_DEGRADED),

    // C_DEGRADED — quality degradations
    CONTRAST_40PCT ("contrast_40pct", SceneCategory.C_DEGRADED),
    OCC_25PCT      ("occ_25pct",      SceneCategory.C_DEGRADED),
    OCC_50PCT      ("occ_50pct",      SceneCategory.C_DEGRADED),
    BLUR_5X5       ("blur_5x5",       SceneCategory.C_DEGRADED),
    HUE_SHIFT_40   ("hue_shift_40",   SceneCategory.C_DEGRADED);

    // -------------------------------------------------------------------------

    private final String        label;
    private final SceneCategory category;

    SceneVariant(String label, SceneCategory category) {
        this.label    = label;
        this.category = category;
    }

    /** The exact string stored in {@link SceneEntry#variantLabel()}. */
    public String label() { return label; }

    /** The category this variant belongs to. */
    public SceneCategory category() { return category; }

    /**
     * Returns the {@code SceneVariant} whose label matches the given string,
     * or {@code null} if no match (e.g. a dynamic Category D label).
     */
    public static SceneVariant fromLabel(String label) {
        if (label == null) return null;
        for (SceneVariant v : values()) {
            if (v.label.equals(label)) return v;
        }
        return null;
    }

    /** Returns {@code true} if the given scene entry matches this variant. */
    public boolean matches(SceneEntry scene) {
        return label.equals(scene.variantLabel());
    }

    @Override
    public String toString() { return label; }
}

