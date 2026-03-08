package org.example.ui.panels;

import org.example.ui.RunConfiguration;
import org.example.ui.SelectionTable;

import javax.swing.*;
import java.awt.*;
import java.util.List;
import java.util.Set;

import static org.example.ui.Palette.BG;
import static org.example.ui.Widgets.*;

/**
 * Abstract base for all five wizard-step selection panels.
 *
 * <p>Checkbox state is driven by a shared {@link RunConfiguration} — panels
 * read from it on rebuild and write back to it on every checkbox toggle.
 */
public abstract class WizardStepPanel extends JPanel {

    protected final SelectionTable   table;
    protected final RunConfiguration cfg;
    private   Runnable               onCheckChanged = () -> {};

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param title the panel header text, e.g. {@code "1 — Matchers"}
     */
    protected WizardStepPanel(String title, RunConfiguration cfg) {
        super(new BorderLayout());
        this.cfg = cfg;
        setBackground(BG);

        table = new SelectionTable();
        table.setOnCheckChanged(() -> {
            onTableCheckChanged();
            onCheckChanged.run();
        });

        JPanel outer = titledPanel(title);
        JCheckBox selAll = check("Select all", false);
        selAll.addActionListener(e -> table.selectAll(selAll.isSelected()));
        outer.add(selAll);
        outer.add(vgap(4));
        outer.add(table);  // SelectionTable already contains its own JScrollPane
        add(outer, BorderLayout.CENTER);
    }

    // ── Subclass contract ─────────────────────────────────────────────────

    /** Builds the current list of rows. */
    protected abstract List<SelectionTable.RowData> buildRows();

    /**
     * Returns the set of row names that should be checked according to
     * {@link RunConfiguration}.  Called during every {@link #rebuild()}.
     */
    protected abstract Set<String> checkedFromConfig();

    /**
     * Called when any checkbox in the table changes.  Subclasses write the
     * new state back into {@link #cfg}.
     */
    protected abstract void onTableCheckChanged();

    // ── Common behaviour ──────────────────────────────────────────────────

    /** Rebuilds rows from {@link #buildRows()}, checking rows from {@link #checkedFromConfig()}. */
    protected void rebuild() {
        table.setRows(buildRows(), checkedFromConfig());
    }

    /** Rebuild preserving row highlight at {@code highlightRow}. */
    protected void rebuild(int highlightRow) {
        table.setRows(buildRows(), checkedFromConfig());
        if (highlightRow >= 0 && highlightRow < table.rowCount())
            table.jtable().setRowSelectionInterval(highlightRow, highlightRow);
    }

    /** Sets a callback fired after any checkbox changes and the config has been updated. */
    public void setOnCheckChanged(Runnable r) { this.onCheckChanged = r; }

    /** Returns the names of all currently checked rows. */
    public Set<String> checkedNames() { return table.checkedNames(); }

    /** Selects or deselects all rows. */
    public void selectAll(boolean selected) { table.selectAll(selected); }

    /** Returns the 0-based indices of all checked rows. */
    public List<Integer> selectedIndices() { return table.selectedIndices(); }
}
