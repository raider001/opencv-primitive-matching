package org.example.ui.panels;

import org.example.*;
import org.example.ui.RunConfiguration;
import org.example.ui.SelectionTable;
import org.example.ui.WizardContext;

import java.util.*;

/**
 * Step 4 — Backgrounds wizard panel.
 *
 * <p>Generated = catalogue PNGs on disk containing the background name segment.
 * Total = {@code N_SCENE_VARIANTS}.
 * Fires {@code onCheckChanged} to cascade into the Scenes panel.
 */
public final class BackgroundsPanel extends WizardStepPanel {

    private final WizardContext ctx;

    public BackgroundsPanel(WizardContext ctx, RunConfiguration cfg, Runnable onCheckChanged) {
        super("4 — Backgrounds", cfg);
        this.ctx = ctx;
        setOnCheckChanged(onCheckChanged);
    }

    @Override
    protected List<SelectionTable.RowData> buildRows() {
        List<SelectionTable.RowData> rows = new ArrayList<>();
        for (BackgroundId bg : BackgroundId.values())
            rows.add(new SelectionTable.RowData(
                    bg.name(), WizardContext.bgGroupOf(bg),
                    ctx.countGeneratedForBackground(bg),
                    WizardContext.expectedBackgroundTotal()));
        return rows;
    }

    @Override
    protected Set<String> checkedFromConfig() {
        Set<String> names = new LinkedHashSet<>();
        for (BackgroundId bg : BackgroundId.values())
            if (cfg.isBgSelected(bg)) names.add(bg.name());
        return names;
    }

    @Override
    protected void onTableCheckChanged() {
        Set<String> checked = table.checkedNames();
        for (BackgroundId bg : BackgroundId.values())
            cfg.setBgSelected(bg, checked.contains(bg.name()));
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /** Rebuilds the table using the current catalogue file cache. */
    public void rebuild() { super.rebuild(); }
}

