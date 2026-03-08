package org.example.ui;

import org.example.*;

import java.util.*;

/**
 * Single source of truth for what is currently selected in the Benchmark Launcher.
 *
 * <p>All five wizard panels read from and write to this model.  No panel holds
 * its own "selected" state — they delegate entirely to this class.
 *
 * <h2>Lifecycle</h2>
 * <ol>
 *   <li>Constructed once and shared with every panel.</li>
 *   <li>Panels call the {@code set*} / {@code toggle*} methods when the user
 *       interacts with a checkbox.</li>
 *   <li>The launcher calls {@code get*} methods when building a run.</li>
 *   <li>{@link #isMatcherSelected}, {@link #isVariantSelected}, etc. drive the
 *       checkbox state rendered in each table row.</li>
 * </ol>
 */
public final class RunConfiguration {

    // ── Selection state ───────────────────────────────────────────────────

    /** Matchers whose top-level checkbox is ticked. */
    private final Set<MatcherDescriptor>               selectedMatchers  = new LinkedHashSet<>();

    /**
     * Per-matcher variant selection.  If a matcher has no entry here it is
     * treated as "all variants selected" (first-visit default).
     */
    private final Map<MatcherDescriptor, Set<String>>  selectedVariants  = new LinkedHashMap<>();

    /** Selected reference IDs. */
    private final Set<ReferenceId>                     selectedRefs      = new LinkedHashSet<>();

    /** Selected background IDs. */
    private final Set<BackgroundId>                    selectedBgs       = new LinkedHashSet<>();

    /** Selected scene variants. */
    private final Set<SceneVariant>                    selectedScenes    = new LinkedHashSet<>();

    // ── Run options ───────────────────────────────────────────────────────
    private boolean clearPrevious    = false;
    private boolean includeNegatives = false;

    // =========================================================================
    //  Matchers
    // =========================================================================

    public boolean isMatcherSelected(MatcherDescriptor md)  { return selectedMatchers.contains(md); }
    public Set<MatcherDescriptor> getSelectedMatchers()     { return Collections.unmodifiableSet(selectedMatchers); }

    public void setMatcherSelected(MatcherDescriptor md, boolean selected) {
        if (selected) selectedMatchers.add(md);
        else          selectedMatchers.remove(md);
    }

    public void setAllMatchersSelected(List<MatcherDescriptor> all, boolean selected) {
        if (selected) selectedMatchers.addAll(all);
        else          selectedMatchers.clear();
    }

    // =========================================================================
    //  Variants  (per-matcher)
    // =========================================================================

    /**
     * Returns the selected variants for {@code md}.
     * If the matcher has never been visited, defaults to all its variants.
     */
    public Set<String> getSelectedVariants(MatcherDescriptor md) {
        if (md == null) return Collections.emptySet();
        return Collections.unmodifiableSet(
                selectedVariants.computeIfAbsent(md, k -> new LinkedHashSet<>(k.variantNames())));
    }

    public boolean isVariantSelected(MatcherDescriptor md, String variant) {
        if (md == null) return false;
        // If not yet in the map, the default is all-selected
        if (!selectedVariants.containsKey(md)) return true;
        return selectedVariants.get(md).contains(variant);
    }

    public void setVariantSelected(MatcherDescriptor md, String variant, boolean selected) {
        Set<String> variants = selectedVariants.computeIfAbsent(
                md, k -> new LinkedHashSet<>(k.variantNames()));
        if (selected) variants.add(variant);
        else          variants.remove(variant);
    }

    public void setAllVariantsSelected(MatcherDescriptor md, boolean selected) {
        if (md == null) return;
        Set<String> variants = selectedVariants.computeIfAbsent(
                md, k -> new LinkedHashSet<>(k.variantNames()));
        if (selected) variants.addAll(md.variantNames());
        else          variants.clear();
    }

    // =========================================================================
    //  References
    // =========================================================================

    public boolean isRefSelected(ReferenceId rid)  { return selectedRefs.contains(rid); }
    public Set<ReferenceId> getSelectedRefs()      { return Collections.unmodifiableSet(selectedRefs); }

    public void setRefSelected(ReferenceId rid, boolean selected) {
        if (selected) selectedRefs.add(rid);
        else          selectedRefs.remove(rid);
    }

    public void setAllRefsSelected(boolean selected) {
        if (selected) selectedRefs.addAll(Arrays.asList(ReferenceId.values()));
        else          selectedRefs.clear();
    }

    // =========================================================================
    //  Backgrounds
    // =========================================================================

    public boolean isBgSelected(BackgroundId bg)   { return selectedBgs.contains(bg); }
    public Set<BackgroundId> getSelectedBgs()      { return Collections.unmodifiableSet(selectedBgs); }

    public void setBgSelected(BackgroundId bg, boolean selected) {
        if (selected) selectedBgs.add(bg);
        else          selectedBgs.remove(bg);
    }

    public void setAllBgsSelected(boolean selected) {
        if (selected) selectedBgs.addAll(Arrays.asList(BackgroundId.values()));
        else          selectedBgs.clear();
    }

    // =========================================================================
    //  Scene variants
    // =========================================================================

    public boolean isSceneSelected(SceneVariant sv) { return selectedScenes.contains(sv); }
    public Set<SceneVariant> getSelectedScenes()    { return Collections.unmodifiableSet(selectedScenes); }

    public void setSceneSelected(SceneVariant sv, boolean selected) {
        if (selected) selectedScenes.add(sv);
        else          selectedScenes.remove(sv);
    }

    public void setAllScenesSelected(boolean selected) {
        if (selected) {
            Arrays.stream(SceneVariant.values())
                  .filter(sv -> sv.category() != SceneCategory.D_NEGATIVE)
                  .forEach(selectedScenes::add);
        } else {
            selectedScenes.clear();
        }
    }

    // =========================================================================
    //  Run options
    // =========================================================================

    public boolean isClearPrevious()               { return clearPrevious; }
    public void    setClearPrevious(boolean v)     { this.clearPrevious = v; }
    public boolean isIncludeNegatives()            { return includeNegatives; }
    public void    setIncludeNegatives(boolean v)  { this.includeNegatives = v; }

    // =========================================================================
    //  Validation helpers
    // =========================================================================

    /** Returns true if enough is selected to start a run. */
    public boolean isRunnable() {
        return !selectedMatchers.isEmpty()
            && selectedMatchers.stream().anyMatch(md -> !getSelectedVariants(md).isEmpty())
            && !selectedRefs.isEmpty()
            && (!selectedScenes.isEmpty() || includeNegatives);
    }

    /**
     * Returns the effective variant set for a matcher when building a run —
     * falls back to all variants if nothing was explicitly selected.
     */
    public List<String> effectiveVariants(MatcherDescriptor md) {
        Set<String> sel = getSelectedVariants(md);
        return sel.isEmpty() ? new ArrayList<>(md.variantNames()) : new ArrayList<>(sel);
    }
}

