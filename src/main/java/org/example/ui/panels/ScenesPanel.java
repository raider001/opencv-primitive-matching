package org.example.ui.panels;

import org.example.*;
import org.example.ui.RunConfiguration;
import org.example.ui.SelectionTable;
import org.example.ui.WizardContext;

import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

public final class ScenesPanel extends WizardStepPanel {

    private final WizardContext ctx;

    public ScenesPanel(WizardContext ctx, RunConfiguration cfg) {
        super("5 — Scene Variants", cfg);
        this.ctx = ctx;
    }

    @Override
    protected List<SelectionTable.RowData> buildRows() {
        return Arrays.stream(SceneVariant.values())
                .filter(sv -> sv.category() != SceneCategory.D_NEGATIVE)
                .map(sv -> new SelectionTable.RowData(
                        sv.label(), sv.category().name(),
                        ctx.countGeneratedForScene(sv),
                        WizardContext.expectedSceneTotal()))
                .toList();
    }

    @Override
    protected Set<String> checkedFromConfig() {
        Set<String> names = new LinkedHashSet<>();
        for (SceneVariant sv : SceneVariant.values())
            if (sv.category() != SceneCategory.D_NEGATIVE && cfg.isSceneSelected(sv))
                names.add(sv.label());
        return names;
    }

    @Override
    protected void onTableCheckChanged() {
        Set<String> checked = table.checkedNames();
        for (SceneVariant sv : SceneVariant.values())
            if (sv.category() != SceneCategory.D_NEGATIVE)
                cfg.setSceneSelected(sv, checked.contains(sv.label()));
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /** Rebuilds the table using the current catalogue file cache. */
    public void rebuild() { super.rebuild(); }
}
