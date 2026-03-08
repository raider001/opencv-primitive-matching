package org.example.ui.panels;

import org.example.*;
import org.example.ui.RunConfiguration;
import org.example.ui.SelectionTable;
import org.example.ui.WizardContext;

import java.util.*;

/**
 * Step 2 — Variants wizard panel.
 *
 * <p>Populated when a matcher is highlighted in {@link MatchersPanel}.
 * Each matcher's checked variant state is stored independently so switching
 * highlight between matchers never loses another matcher's selection.
 *
 * Fires {@code onCheckChanged} whenever a variant checkbox is toggled
 * so the launcher can cascade into the References panel.
 */
public final class VariantsPanel extends WizardStepPanel {

    private final WizardContext   ctx;
    private MatcherDescriptor     currentMd = null;

    public VariantsPanel(WizardContext ctx, RunConfiguration cfg, Runnable onCheckChanged) {
        super("2 — Variants", cfg);
        this.ctx = ctx;
        setOnCheckChanged(onCheckChanged);
        table.setEnabled(false);
    }

    // ── Subclass contract ──────────────────────────────────────────────────

    @Override
    protected List<SelectionTable.RowData> buildRows() {
        List<SelectionTable.RowData> rows = new ArrayList<>();
        if (currentMd != null) {
            for (String vn : currentMd.variantNames()) {
                int gen   = WizardContext.countGeneratedVariantFiles(currentMd, vn);
                int total = WizardContext.expectedVariantTotal();
                rows.add(new SelectionTable.RowData(vn, "[" + currentMd.tag() + "]", gen, total));
            }
        }
        return rows;
    }

    @Override
    protected Set<String> checkedFromConfig() {
        if (currentMd == null) return Set.of();
        return new LinkedHashSet<>(cfg.getSelectedVariants(currentMd));
    }

    @Override
    protected void onTableCheckChanged() {
        if (currentMd == null) return;
        Set<String> checked = table.checkedNames();
        for (String vn : currentMd.variantNames())
            cfg.setVariantSelected(currentMd, vn, checked.contains(vn));
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /** Repopulates for the given matcher, restoring its config-driven checked state. */
    public void populate(MatcherDescriptor md) {
        currentMd = md;
        rebuild();
        table.setEnabled(md != null && !md.variantNames().isEmpty());
    }

    /** Returns the variants currently selected for {@code md} (reads directly from config). */
    public List<String> getSelectedVariants(MatcherDescriptor md) {
        if (md == null) return List.of();
        return new ArrayList<>(cfg.getSelectedVariants(md));
    }
}
