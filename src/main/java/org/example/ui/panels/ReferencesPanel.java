package org.example.ui.panels;

import org.example.*;
import org.example.factories.ReferenceId;
import org.example.ui.RunConfiguration;
import org.example.ui.SelectionTable;
import org.example.ui.WizardContext;

import javax.swing.*;
import java.util.*;

public final class ReferencesPanel extends WizardStepPanel {

    private MatcherDescriptor currentMd       = null;
    private List<String>      currentVariants = List.of();

    /** Tracks the latest rebuild request; stale workers are discarded. */
    private volatile long rebuildGeneration = 0;

    public ReferencesPanel(RunConfiguration cfg, Runnable onCheckChanged) {
        super("3 — References", cfg);
        setOnCheckChanged(onCheckChanged);
    }

    @Override
    protected List<SelectionTable.RowData> buildRows() {
        // Called synchronously only by the base rebuild() — we override rebuild(md,variants)
        // to use SwingWorker, so this path is only hit if rebuild() is called directly.
        return buildRowsNow(currentMd, new ArrayList<>(currentVariants));
    }

    @Override
    protected Set<String> checkedFromConfig() {
        Set<String> names = new LinkedHashSet<>();
        for (ReferenceId rid : ReferenceId.values())
            if (cfg.isRefSelected(rid)) names.add(rid.name());
        return names;
    }

    @Override
    protected void onTableCheckChanged() {
        Set<String> checked = table.checkedNames();
        for (ReferenceId rid : ReferenceId.values())
            cfg.setRefSelected(rid, checked.contains(rid.name()));
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /**
     * Rebuilds the table off the EDT — filesystem scans run in a background
     * thread; only the final {@code table.setRows()} call touches the EDT.
     * Rapid successive calls are coalesced: only the latest wins.
     */
    public void rebuild(MatcherDescriptor md, List<String> selectedVariants) {
        this.currentMd       = md;
        this.currentVariants = selectedVariants != null ? selectedVariants : List.of();

        final long generation = ++rebuildGeneration;
        final MatcherDescriptor snapMd       = md;
        final List<String>      snapVariants = new ArrayList<>(this.currentVariants);

        new SwingWorker<List<SelectionTable.RowData>, Void>() {
            @Override protected List<SelectionTable.RowData> doInBackground() {
                return buildRowsNow(snapMd, snapVariants);
            }
            @Override protected void done() {
                if (generation != rebuildGeneration) return; // superseded
                try {
                    table.setRows(get(), checkedFromConfig());
                } catch (Exception ignored) {}
            }
        }.execute();
    }

    // ── Private ────────────────────────────────────────────────────────────

    private static List<SelectionTable.RowData> buildRowsNow(
            MatcherDescriptor md, List<String> variants) {
        List<SelectionTable.RowData> rows = new ArrayList<>();
        for (ReferenceId rid : ReferenceId.values()) {
            int gen = 0;
            if (md != null && !variants.isEmpty())
                for (String vn : variants)
                    gen += WizardContext.countGeneratedForRefInVariant(md, vn, rid);
            rows.add(new SelectionTable.RowData(
                    rid.name(), WizardContext.groupOf(rid),
                    gen, WizardContext.expectedRefTotal()));
        }
        return rows;
    }
}
