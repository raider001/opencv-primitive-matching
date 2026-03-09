package org.example;

import org.example.analytics.AnalysisResult;

/**
 * Marker interface for all per-matcher variant enums.
 *
 * <p>Every matcher defines its own enum (e.g. {@link org.example.matchers.TmVariant},
 * {@link org.example.matchers.HistVariant}) that implements this interface.
 *
 * <p>The {@link #variantName()} method returns the exact string used in
 * {@link AnalysisResult#methodName()} and in annotated-image directory names on disk.
 * Existing on-disk data is unaffected.
 */
public interface MatcherVariant {

    /** The exact variant name string, e.g. {@code "TM_CCOEFF_NORMED_CF_LOOSE"}. */
    String variantName();

    /**
     * Convenience: returns a {@code Set<String>} of variant names from an array of
     * {@code MatcherVariant} values
     */
    static java.util.Set<String> namesOf(MatcherVariant... variants) {
        java.util.Set<String> set = new java.util.LinkedHashSet<>();
        for (MatcherVariant v : variants) set.add(v.variantName());
        return java.util.Collections.unmodifiableSet(set);
    }

    /** Returns a {@code Set<String>} of all variant names in the given enum class. */
    static <E extends Enum<E> & MatcherVariant> java.util.Set<String> allNamesOf(Class<E> cls) {
        java.util.Set<String> set = new java.util.LinkedHashSet<>();
        for (E v : cls.getEnumConstants()) set.add(v.variantName());
        return java.util.Collections.unmodifiableSet(set);
    }
}

