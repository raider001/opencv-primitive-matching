package org.example.colour;

/**
 * Colour-filter pre-processing mode applied before matching.
 *
 * <ul>
 *   <li>{@link #NONE}  — full-colour image, no pre-filtering (base variant)</li>
 *   <li>{@link #LOOSE} — pixels outside ±{@value ColourPreFilter#LOOSE}° hue are zeroed</li>
 *   <li>{@link #TIGHT} — pixels outside ±{@value ColourPreFilter#TIGHT}° hue are zeroed</li>
 * </ul>
 */
public enum CfMode {

    /** No colour pre-filter — full BGR image passed to the matcher. */
    NONE(""),

    /** Loose hue tolerance (±15°) — non-matching pixels zeroed. */
    LOOSE("_CF_LOOSE"),

    /** Tight hue tolerance (±8°) — non-matching pixels zeroed. */
    TIGHT("_CF_TIGHT");

    /** Suffix appended to the base variant name to form the full variant name. */
    private final String suffix;

    CfMode(String suffix) { this.suffix = suffix; }

    /** Returns the suffix appended to a base method name, e.g. {@code "_CF_LOOSE"}. */
    public String suffix() { return suffix; }

    /** Returns the hue tolerance used for this mode (delegates to {@link ColourPreFilter}). */
    public double hueTolerance() {
        return switch (this) {
            case LOOSE -> ColourPreFilter.LOOSE;
            case TIGHT -> ColourPreFilter.TIGHT;
            case NONE  -> 0.0;
        };
    }

    /** Returns {@code true} if a colour pre-filter should be applied. */
    public boolean isFiltered() { return this != NONE; }

    /**
     * Returns the {@code CfMode} for a given suffix string,
     * or {@link #NONE} if the suffix is blank/null.
     */
    public static CfMode fromSuffix(String suffix) {
        if (suffix == null || suffix.isEmpty()) return NONE;
        for (CfMode m : values()) {
            if (m.suffix.equals(suffix)) return m;
        }
        return NONE;
    }
}

