package org.example.ui;

import javax.swing.*;
import javax.swing.border.*;
import javax.swing.table.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.function.IntBinaryOperator;

import static org.example.ui.Palette.*;
import static org.example.ui.Widgets.*;

/**
 * A reusable dark-themed selection table used across all five wizard steps.
 *
 * <p>Columns:
 * <pre>
 *  [✔]  Name                     Variants / options     Generated
 * </pre>
 *
 * Row colour:
 * <ul>
 *   <li>Green  — 100% of expected results generated</li>
 *   <li>Orange — partially generated (&gt;0 but &lt;100%)</li>
 *   <li>Grey (DIM) — not generated at all</li>
 * </ul>
 *
 * <p>Each row stores a {@link RowData} record which carries the display name,
 * the variants/options sub-label, and a generation fraction expressed as
 * {@code generated} and {@code total}.  Callers update these values via
 * {@link #updateRow(int, int, int)} and call {@link #repaint()}.
 */
public final class SelectionTable extends JPanel {

    // ── Row data ──────────────────────────────────────────────────────────

    public record RowData(String name, String variantsLabel, int generated, int total) {}

    // ── Internal model ────────────────────────────────────────────────────

    private final DefaultTableModel model;
    private final JTable            table;
    private final List<RowData>     rows = new ArrayList<>();

    // Callback that supplies (generated, total) -> row foreground colour
    private static final IntBinaryOperator GEN_COLOUR_IDX = (gen, tot) -> {
        if (tot == 0 || gen == 0) return 0;   // DIM
        if (gen >= tot)           return 2;   // GREEN
        return 1;                             // ORANGE
    };
    private static final Color[] GEN_COLOURS = { DIM, ORANGE, GREEN };

    // ── Constructor ───────────────────────────────────────────────────────

    public SelectionTable() {
        setLayout(new BoxLayout(this, BoxLayout.Y_AXIS));
        setBackground(PANEL);
        setAlignmentX(Component.LEFT_ALIGNMENT);

        model = new DefaultTableModel(new Object[]{"✔", "Name", "Variants", "Generated"}, 0) {
            @Override public Class<?> getColumnClass(int c) { return c == 0 ? Boolean.class : String.class; }
            @Override public boolean  isCellEditable(int r, int c) { return c == 0; }
        };
        table = new JTable(model);
        table.setBackground(PANEL); table.setForeground(WHITE); table.setFont(BODY);
        table.setGridColor(BORDER); table.setRowHeight(22); table.setShowGrid(true);
        table.setSelectionBackground(ROW_SEL); table.setSelectionForeground(WHITE);
        table.setFillsViewportHeight(true);
        table.getTableHeader().setBackground(BG); table.getTableHeader().setForeground(DIM);
        table.getTableHeader().setFont(SMALL);
        table.getColumnModel().getColumn(0).setMaxWidth(28); table.getColumnModel().getColumn(0).setPreferredWidth(28);
        table.getColumnModel().getColumn(1).setPreferredWidth(220);
        table.getColumnModel().getColumn(2).setPreferredWidth(160);
        table.getColumnModel().getColumn(3).setPreferredWidth(110);

        // Custom renderer that colour-codes rows based on generation status
        DefaultTableCellRenderer rowRenderer = new DefaultTableCellRenderer() {
            @Override public Component getTableCellRendererComponent(
                    JTable t, Object v, boolean sel, boolean foc, int row, int col) {
                super.getTableCellRendererComponent(t,v,sel,foc,row,col);
                Color base = sel ? ROW_SEL : (row % 2 == 0 ? PANEL : ROW_ALT);
                setBackground(base);
                Color fg = WHITE;
                if (row < rows.size()) {
                    RowData rd = rows.get(row);
                    fg = GEN_COLOURS[GEN_COLOUR_IDX.applyAsInt(rd.generated(), rd.total())];
                }
                setForeground(sel ? WHITE : fg);   // keep full white when selected
                setFont(col == 0 ? SMALL : BODY);
                setBorder(new EmptyBorder(0,4,0,4));
                return this;
            }
        };
        for (int c = 1; c < 4; c++) table.getColumnModel().getColumn(c).setCellRenderer(rowRenderer);

        JScrollPane sp = scrollPane(table);
        sp.setAlignmentX(Component.LEFT_ALIGNMENT);
        sp.setMaximumSize(new Dimension(Integer.MAX_VALUE, Integer.MAX_VALUE));
        add(sp);
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /**
     * Replaces all rows with the given list and resets the check-state to
     * {@code defaultSelected} for every row.
     */
    public void setRows(List<RowData> data, boolean defaultSelected) {
        rows.clear(); rows.addAll(data);
        model.setRowCount(0);
        for (RowData rd : data)
            model.addRow(new Object[]{defaultSelected, rd.name(), rd.variantsLabel(),
                    genLabel(rd.generated(), rd.total())});
    }

    /**
     * Updates the generation fraction of an existing row and refreshes the
     * display string in the Generated column.
     */
    public void updateRow(int row, int generated, int total) {
        if (row < 0 || row >= rows.size()) return;
        RowData old = rows.get(row);
        rows.set(row, new RowData(old.name(), old.variantsLabel(), generated, total));
        model.setValueAt(genLabel(generated, total), row, 3);
        table.repaint();
    }

    /** Updates the "Status" string in col 2 (Variants column) for a running row. */
    public void setVariantsLabel(int row, String label) {
        if (row < 0 || row >= rows.size()) return;
        RowData old = rows.get(row);
        rows.set(row, new RowData(old.name(), label, old.generated(), old.total()));
        model.setValueAt(label, row, 2);
    }

    /** Returns the indices (0-based) of all checked rows. */
    public List<Integer> selectedIndices() {
        List<Integer> sel = new ArrayList<>();
        for (int r = 0; r < model.getRowCount(); r++)
            if (Boolean.TRUE.equals(model.getValueAt(r, 0))) sel.add(r);
        return sel;
    }

    /** Returns {@code true} if at least one row is checked. */
    public boolean hasSelection() { return !selectedIndices().isEmpty(); }

    /** Selects or deselects all rows. */
    public void selectAll(boolean selected) {
        for (int r = 0; r < model.getRowCount(); r++) model.setValueAt(selected, r, 0);
    }

    /** Direct access to the underlying JTable (e.g. for attaching list selection listeners). */
    public JTable table() { return table; }

    /** Number of data rows. */
    public int rowCount() { return model.getRowCount(); }

    public RowData rowData(int row) { return rows.get(row); }

    // ── Helpers ───────────────────────────────────────────────────────────

    private static String genLabel(int generated, int total) {
        if (total == 0)          return "—";
        if (generated >= total)  return "✅ " + total + "/" + total;
        if (generated > 0)       return "⬤ " + generated + "/" + total;
        return                          "— 0/" + total;
    }
}

