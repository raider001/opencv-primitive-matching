package org.example.ui.panels;

import org.example.*;
import org.example.ui.RunConfiguration;
import org.example.ui.SelectionTable;
import org.example.ui.WizardContext;

import javax.swing.*;
import java.util.*;

/**
 * Step 1 — Matchers wizard panel.
 *
 * <p>Each row has a checkbox (for inclusion in a run) plus a row-highlight
 * that drives the Variants panel cascade.
 *
 * <p>Total per row = {@code Variants × References × Backgrounds × SceneVariants}.
 * Generated per row = sum of annotated PNGs across all variant dirs.
 */
public final class MatchersPanel extends WizardStepPanel {

    private final WizardContext ctx;
    private final Runnable      onHighlightChanged;

    public MatchersPanel(WizardContext ctx, RunConfiguration cfg, Runnable onHighlightChanged) {
        super("1 — Matchers", cfg);
        this.ctx               = ctx;
        this.onHighlightChanged = onHighlightChanged;

        table.jtable().setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        table.jtable().getSelectionModel().addListSelectionListener(
                e -> { if (!e.getValueIsAdjusting()) onHighlightChanged.run(); });
    }

    // ── Subclass contract ──────────────────────────────────────────────────


    @Override
    protected List<SelectionTable.RowData> buildRows() {
        List<SelectionTable.RowData> rows = new ArrayList<>();
        for (MatcherDescriptor md : ctx.matcherList) {
            int variantCount = md.variantNames().size();
            int gen          = WizardContext.countGeneratedMatcherFiles(md);
            int total        = WizardContext.expectedMatcherTotal(variantCount);
            rows.add(new SelectionTable.RowData(
                    md.displayName() + "  [" + md.tag() + "]",
                    variantCount + " variants", gen, total));
        }
        return rows;
    }

    @Override
    protected Set<String> checkedFromConfig() {
        Set<String> names = new LinkedHashSet<>();
        for (MatcherDescriptor md : ctx.matcherList)
            if (cfg.isMatcherSelected(md))
                names.add(md.displayName() + "  [" + md.tag() + "]");
        return names;
    }

    @Override
    protected void onTableCheckChanged() {
        Set<String> checked = table.checkedNames();
        for (MatcherDescriptor md : ctx.matcherList)
            cfg.setMatcherSelected(md, checked.contains(md.displayName() + "  [" + md.tag() + "]"));
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /** Rebuilds rows, preserving both checked state and row highlight. */
    public void refresh() {
        rebuild(table.jtable().getSelectedRow());
    }

    /** Total number of registered matchers. */
    public int totalCount() { return ctx.matcherList.size(); }

    /** Returns the currently highlighted matcher (drives Variants), or {@code null}. */
    public MatcherDescriptor getHighlightedMatcher() {
        int sel = table.jtable().getSelectedRow();
        if (sel >= 0 && sel < ctx.matcherList.size()) return ctx.matcherList.get(sel);
        return null;
    }
}
