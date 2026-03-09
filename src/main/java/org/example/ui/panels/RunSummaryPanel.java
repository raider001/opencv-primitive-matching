package org.example.ui.panels;

import org.example.*;
import org.example.factories.BackgroundId;
import org.example.factories.ReferenceId;
import org.example.scene.SceneVariant;
import org.example.ui.RunConfiguration;
import org.example.ui.WizardContext;

import javax.swing.*;
import javax.swing.border.*;
import java.awt.*;

import static org.example.ui.Palette.*;
import static org.example.ui.Widgets.*;

/**
 * A compact read-only panel that summarises the current run configuration —
 * what is checked across all five wizard steps.  Placed next to the Scene
 * Variants panel in row 1 of the wizard.
 *
 * <p>Call {@link #refresh(MatchersPanel, VariantsPanel, ReferencesPanel,
 * BackgroundsPanel, ScenesPanel)} after any selection change to update
 * the display.
 */
public final class RunSummaryPanel extends JPanel {

    // ── Section labels ────────────────────────────────────────────────────
    private final JLabel lblMatchers;
    private final JLabel lblVariants;
    private final JLabel lblRefs;
    private final JLabel lblBgs;
    private final JLabel lblScenes;

    // ── Count + estimated-pairs label ────────────────────────────────────
    private final JLabel lblEstimate;

    public RunSummaryPanel() {
        super(new BorderLayout());
        setBackground(PANEL);

        JPanel outer = titledPanel("▶  Run Summary");

        lblMatchers = summaryRow(outer, "Matchers");
        lblVariants = summaryRow(outer, "Variants");
        lblRefs     = summaryRow(outer, "References");
        lblBgs      = summaryRow(outer, "Backgrounds");
        lblScenes   = summaryRow(outer, "Scene Variants");

        outer.add(vgap(8));
        outer.add(sepLine("── estimate "));
        outer.add(vgap(4));
        lblEstimate = label("—", BOLD13, DIM);
        lblEstimate.setAlignmentX(Component.LEFT_ALIGNMENT);
        outer.add(lblEstimate);

        add(outer, BorderLayout.CENTER);
    }

    // ── Public API ─────────────────────────────────────────────────────────

    /** Refreshes all summary lines directly from the {@link RunConfiguration}. */
    public void refresh(RunConfiguration cfg, int totalMatchers) {
        int mCount = cfg.getSelectedMatchers().size();
        set(lblMatchers, mCount, totalMatchers,
                cfg.getSelectedMatchers().stream().map(MatcherDescriptor::tag)
                   .reduce((a, b) -> a + ", " + b).orElse("none"));

        // Total variant count across all selected matchers
        long totalVariants = cfg.getSelectedMatchers().stream()
                .mapToLong(md -> md.variantNames().size()).sum();
        long selVariants   = cfg.getSelectedMatchers().stream()
                .mapToLong(md -> cfg.getSelectedVariants(md).size()).sum();
        set(lblVariants, (int) selVariants, (int) totalVariants,
                selVariants == 0 ? "none" : selVariants + " across " + mCount + " matcher(s)");

        int rCount = cfg.getSelectedRefs().size();
        set(lblRefs, rCount, WizardContext.N_REFERENCES,
                cfg.getSelectedRefs().isEmpty() ? "none"
                        : cfg.getSelectedRefs().stream().map(ReferenceId::name)
                             .reduce((a, b) -> a + ", " + b).orElse("none"));

        int bCount = cfg.getSelectedBgs().size();
        set(lblBgs, bCount, WizardContext.N_BACKGROUNDS,
                cfg.getSelectedBgs().isEmpty() ? "none"
                        : cfg.getSelectedBgs().stream().map(BackgroundId::name)
                             .reduce((a, b) -> a + ", " + b).orElse("none"));

        int sCount = cfg.getSelectedScenes().size();
        set(lblScenes, sCount, WizardContext.N_SCENE_VARIANTS,
                cfg.getSelectedScenes().isEmpty() ? "none"
                        : cfg.getSelectedScenes().stream().map(SceneVariant::label)
                             .reduce((a, b) -> a + ", " + b).orElse("none"));

        // Estimated pairs = matchers × variants × refs × bgs × scenes
        long pairs = cfg.getSelectedMatchers().stream()
                .mapToLong(md -> cfg.getSelectedVariants(md).size())
                .sum() * rCount * bCount * sCount;
        if (pairs == 0) {
            lblEstimate.setText("— no selection");
            lblEstimate.setForeground(DIM);
        } else {
            lblEstimate.setText(String.format("%,d scene pairs", pairs));
            lblEstimate.setForeground(ORANGE);
        }
    }

    // ── Private helpers ────────────────────────────────────────────────────

    /**
     * Creates a two-line block: a dim key label and a value label,
     * returning the value label so the caller can update it.
     */
    private static JLabel summaryRow(JPanel parent, String key) {
        JLabel keyLbl = label(key, SMALL, DIM);
        keyLbl.setAlignmentX(Component.LEFT_ALIGNMENT);
        keyLbl.setBorder(new EmptyBorder(4, 0, 0, 0));
        parent.add(keyLbl);

        JLabel valLbl = label("—", SMALL, WHITE);
        valLbl.setAlignmentX(Component.LEFT_ALIGNMENT);
        valLbl.setBorder(new EmptyBorder(0, 8, 0, 0));
        parent.add(valLbl);
        return valLbl;
    }

    /** Updates a value label with count/total and a truncated name list. */
    private static void set(JLabel lbl, int count, int total, String names) {
        Color colour = count == 0 ? DIM : count >= total ? GREEN : ORANGE;
        // Truncate long name lists
        String display = names.length() > 60 ? names.substring(0, 57) + "…" : names;
        lbl.setText(count + " / " + total + "  " + display);
        lbl.setForeground(colour);
        lbl.setToolTipText(names); // full list on hover
    }
}
