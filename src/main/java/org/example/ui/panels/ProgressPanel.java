package org.example.ui.panels;

import javax.swing.*;
import javax.swing.border.*;
import java.awt.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import static org.example.ui.Palette.*;
import static org.example.ui.Widgets.*;

/**
 * The full-screen progress panel shown while matchers are running.
 *
 * <p>Layout mirrors ProgressDisplay exactly:
 * <pre>
 *  [TAG]  N matchers × M scenes  |  T threads
 *  Progress  [████████████░░░░░░]  47.3%
 *  Pairs  1,234 / 2,610  |  elapsed 48s  |  ETA ~54s  |  results 885
 *  ── per-matcher ─────────────────────────────────────────────────
 *  Name                    [███░░░░░]  12/26   ◌ running
 *  ...
 *  ── post-processing ────────────────────────────────────────────
 *  ▶ Writing report…
 * </pre>
 *
 * <p>State is driven by the runner thread via the public {@code update*} methods
 * (all thread-safe) plus a 100 ms Swing Timer that flushes everything to the EDT.
 */
public final class ProgressPanel extends JPanel {

    private static final int GLOBAL_BAR_W  = 500;
    private static final int MATCHER_BAR_W = 140;
    private static final int VARIANT_BAR_W = 110;

    // ── Widgets ───────────────────────────────────────────────────────────
    private final JLabel   headerLabel;
    private final BarPanel globalBar;
    private final JLabel   pairsLabel;
    private final JPanel   matcherRows;
    private final JLabel   statusLabel;
    private final JButton  cancelBackBtn;

    // Per-matcher widgets  [matcherIdx]
    private JLabel[]   matcherNameLabels;
    private BarPanel[] matcherBars;
    private JLabel[]   matcherCountLabels;
    private JLabel[]   matcherStateLabels;

    // Per-variant widgets  [matcherIdx][variantIdx]
    private JLabel[][]   variantNameLabels;
    private BarPanel[][] variantBars;
    private JLabel[][]   variantCountLabels;
    private JLabel[][]   variantStateLabels;

    // ── Run state (thread-safe) ───────────────────────────────────────────
    private final ConcurrentHashMap<Integer, AtomicInteger> pairsDone   = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Integer, AtomicInteger> variantDone = new ConcurrentHashMap<>();
    private volatile int[]   totalPairsPerMatcher;
    private volatile int[][] totalPairsPerVariant;
    private volatile int     globalTotal    = 0;
    private final AtomicInteger globalDone  = new AtomicInteger(0);
    private volatile int     globalResultCnt = 0;
    private volatile long    startMs         = 0;
    private volatile String  postStatusText  = "";

    private javax.swing.Timer repaintTimer;

    // ── Constructor ───────────────────────────────────────────────────────

    public ProgressPanel(Runnable onCancelOrBack) {
        setLayout(new BorderLayout()); setBackground(BG);
        setBorder(new EmptyBorder(20,24,16,24));

        JPanel inner = new JPanel();
        inner.setLayout(new BoxLayout(inner, BoxLayout.Y_AXIS));
        inner.setBackground(BG);

        headerLabel = label("", BOLD13, HEADER);
        headerLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        inner.add(headerLabel); inner.add(vgap(12));

        inner.add(label("Progress", SMALL, DIM)); inner.add(vgap(4));
        globalBar = new BarPanel(GLOBAL_BAR_W, 24, BAR_FILL, BAR_EMPTY, BAR_BG, true);
        globalBar.setAlignmentX(Component.LEFT_ALIGNMENT);
        inner.add(globalBar); inner.add(vgap(10));

        pairsLabel = monoLabel("Pairs  0 / 0");
        pairsLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        inner.add(pairsLabel); inner.add(vgap(12));

        inner.add(sepLine("── per-matcher ")); inner.add(vgap(6));

        matcherRows = new JPanel();
        matcherRows.setLayout(new BoxLayout(matcherRows, BoxLayout.Y_AXIS));
        matcherRows.setBackground(BG);
        matcherRows.setAlignmentX(Component.LEFT_ALIGNMENT);

        JScrollPane scroll = new JScrollPane(matcherRows,
                JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED,
                JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
        scroll.setBackground(BG);
        scroll.getViewport().setBackground(BG);
        scroll.setBorder(BorderFactory.createLineBorder(SEP, 1));
        scroll.setAlignmentX(Component.LEFT_ALIGNMENT);
        scroll.setPreferredSize(new Dimension(0, 320));
        scroll.setMaximumSize(new Dimension(Integer.MAX_VALUE, 320));
        inner.add(scroll); inner.add(vgap(10));

        inner.add(sepLine("── post-processing ")); inner.add(vgap(5));
        statusLabel = label("", SMALL, STATUS_LN);
        statusLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
        inner.add(statusLabel);

        add(inner, BorderLayout.CENTER);

        JPanel bottom = new JPanel(new FlowLayout(FlowLayout.RIGHT, 8, 4));
        bottom.setBackground(BG);
        bottom.setBorder(new EmptyBorder(12, 0, 0, 0));
        cancelBackBtn = accentBtn("✕  Cancel", BTN_RED);
        cancelBackBtn.addActionListener(e -> onCancelOrBack.run());
        bottom.add(cancelBackBtn);
        add(bottom, BorderLayout.SOUTH);
    }

    // ── Initialise for a new run ──────────────────────────────────────────

    /**
     * Resets all state and populates per-matcher rows.
     * Call {@link #setMatcherVariants} immediately after for each matcher
     * (before switching to this panel) to insert variant sub-rows.
     */
    public void startRun(String[] matcherNames, int nThreads, int estimatedTotal) {
        int n = matcherNames.length;
        matcherNameLabels  = new JLabel[n];
        matcherBars        = new BarPanel[n];
        matcherCountLabels = new JLabel[n];
        matcherStateLabels = new JLabel[n];
        variantNameLabels  = new JLabel[n][];
        variantBars        = new BarPanel[n][];
        variantCountLabels = new JLabel[n][];
        variantStateLabels = new JLabel[n][];
        totalPairsPerMatcher = new int[n];
        totalPairsPerVariant = new int[n][];
        pairsDone.clear(); variantDone.clear();
        globalDone.set(0);
        globalTotal      = Math.max(estimatedTotal, 1);
        globalResultCnt  = 0;
        startMs          = System.currentTimeMillis();
        postStatusText   = "";

        headerLabel.setText("[RUN]  " + n + " matcher" + (n == 1 ? "" : "s") + "  |  " + nThreads + " threads");
        globalBar.setFraction(0); globalBar.setLabel("0.0%");
        pairsLabel.setText("Pairs  0 / ?");
        statusLabel.setText(""); statusLabel.setForeground(STATUS_LN);
        cancelBackBtn.setText("✕  Cancel"); cancelBackBtn.setBackground(BTN_RED); cancelBackBtn.setEnabled(true);

        matcherRows.removeAll();
        for (int i = 0; i < n; i++) {
            pairsDone.put(i, new AtomicInteger(0));
            matcherRows.add(buildMatcherRow(i, matcherNames[i]));
        }
        matcherRows.revalidate(); matcherRows.repaint();

        if (repaintTimer != null) repaintTimer.stop();
        repaintTimer = new javax.swing.Timer(100, ev -> repaint100ms());
        repaintTimer.start();
    }

    /**
     * Inserts per-variant sub-rows for {@code matcherIdx} directly below its
     * matcher row.  Must be called on the EDT after {@link #startRun} and
     * before the panel is shown.
     */
    public void setMatcherVariants(int matcherIdx, String[] variantNames) {
        int nv = variantNames.length;
        variantNameLabels[matcherIdx]    = new JLabel[nv];
        variantBars[matcherIdx]          = new BarPanel[nv];
        variantCountLabels[matcherIdx]   = new JLabel[nv];
        variantStateLabels[matcherIdx]   = new JLabel[nv];
        totalPairsPerVariant[matcherIdx] = new int[nv];
        for (int vi = 0; vi < nv; vi++) {
            variantDone.put(variantKey(matcherIdx, vi), new AtomicInteger(0));
        }

        // Rebuild the full list: all matchers + their already-registered variant rows,
        // inserting the freshly-built variant rows for matcherIdx.
        matcherRows.removeAll();
        for (int mi = 0; mi < matcherNameLabels.length; mi++) {
            matcherRows.add(matcherRowPanel(mi));
            if (variantNameLabels[mi] == null) continue;
            for (int vi = 0; vi < variantNameLabels[mi].length; vi++) {
                if (mi == matcherIdx && variantNameLabels[mi][vi] == null) {
                    // Build brand-new variant row for this matcher
                    matcherRows.add(buildVariantRow(mi, vi, variantNames[vi]));
                } else if (variantNameLabels[mi][vi] != null) {
                    // Re-attach already-built widgets
                    matcherRows.add(variantRowPanel(mi, vi));
                }
            }
        }
        matcherRows.revalidate(); matcherRows.repaint();
    }

    /** Sets the total pairs for a variant. */
    public void setVariantTotal(int matcherIdx, int variantIdx, int total) {
        if (totalPairsPerVariant != null
                && matcherIdx < totalPairsPerVariant.length
                && totalPairsPerVariant[matcherIdx] != null
                && variantIdx < totalPairsPerVariant[matcherIdx].length) {
            totalPairsPerVariant[matcherIdx][variantIdx] = total;
        }
    }

    /** Increments the done counter for a specific variant. */
    public void incrementVariantDone(int matcherIdx, int variantIdx) {
        variantDone.computeIfAbsent(variantKey(matcherIdx, variantIdx),
                k -> new AtomicInteger(0)).incrementAndGet();
    }

    // ── Single-task / benchmark modes ────────────────────────────────────

    public void startSingleTask(String lbl) {
        resetArrays(0);
        pairsDone.clear(); variantDone.clear();
        globalDone.set(0); globalTotal = 1; globalResultCnt = 0;
        startMs = System.currentTimeMillis(); postStatusText = "";
        headerLabel.setText(lbl);
        globalBar.setFraction(0); globalBar.setLabel("0.0%");
        pairsLabel.setText(""); statusLabel.setText(""); statusLabel.setForeground(STATUS_LN);
        cancelBackBtn.setText("✕  Cancel"); cancelBackBtn.setBackground(BTN_RED); cancelBackBtn.setEnabled(true);
        matcherRows.removeAll(); matcherRows.revalidate(); matcherRows.repaint();
        if (repaintTimer != null) repaintTimer.stop();
        repaintTimer = new javax.swing.Timer(100, ev -> repaint100ms());
        repaintTimer.start();
    }

    public void startBenchmark() {
        resetArrays(0);
        pairsDone.clear(); variantDone.clear();
        globalDone.set(0); globalTotal = 1; globalResultCnt = 0;
        startMs = System.currentTimeMillis(); postStatusText = "";
        headerLabel.setText("[BENCHMARK]  Collating all technique reports…");
        globalBar.setFraction(0); globalBar.setLabel("0.0%");
        pairsLabel.setText(""); statusLabel.setText(""); statusLabel.setForeground(STATUS_LN);
        cancelBackBtn.setText("✕  Cancel"); cancelBackBtn.setBackground(BTN_RED); cancelBackBtn.setEnabled(true);
        matcherRows.removeAll(); matcherRows.revalidate(); matcherRows.repaint();
        if (repaintTimer != null) repaintTimer.stop();
        repaintTimer = new javax.swing.Timer(100, ev -> repaint100ms());
        repaintTimer.start();
    }

    // ── Thread-safe update API ────────────────────────────────────────────

    public void setGlobalDone(int done)  { globalDone.set(done); }
    public void setGlobalTotal(int total){ this.globalTotal = Math.max(total, 1); }

    public void setMatcherTotal(int idx, int total) {
        if (totalPairsPerMatcher != null && idx < totalPairsPerMatcher.length)
            totalPairsPerMatcher[idx] = total;
    }

    public void incrementDone(int matcherIdx) {
        pairsDone.computeIfAbsent(matcherIdx, k -> new AtomicInteger(0)).incrementAndGet();
        globalDone.incrementAndGet();
    }

    public void incrementResults() { synchronized (this) { globalResultCnt++; } }
    public void setPostStatus(String text) { postStatusText = text; }

    public void setMatcherRunning(int idx) {
        SwingUtilities.invokeLater(() -> {
            if (matcherStateLabels != null && idx < matcherStateLabels.length)
                matcherStateLabels[idx].setText("◌ running");
                matcherStateLabels[idx].setForeground(YELLOW);
        });
    }

    public void setMatcherDone(int idx, double acc) {
        SwingUtilities.invokeLater(() -> {
            if (matcherStateLabels != null && idx < matcherStateLabels.length) {
                matcherStateLabels[idx].setText(String.format("✓ done  %.0f%% acc", acc));
                matcherStateLabels[idx].setForeground(GREEN);
                matcherNameLabels[idx].setForeground(GREEN);
            }
        });
    }

    public void setMatcherError(int idx) {
        SwingUtilities.invokeLater(() -> {
            if (matcherStateLabels != null && idx < matcherStateLabels.length) {
                matcherStateLabels[idx].setText("❌ error");
                matcherStateLabels[idx].setForeground(RED);
            }
        });
    }

    public void setMatcherSkipped(int idx) {
        SwingUtilities.invokeLater(() -> {
            if (matcherStateLabels != null && idx < matcherStateLabels.length) {
                matcherStateLabels[idx].setText("⚠ skipped");
                matcherStateLabels[idx].setForeground(YELLOW);
            }
        });
    }

    public void markFinished(boolean cancelled, String extraMsg) {
        SwingUtilities.invokeLater(() -> {
            if (repaintTimer != null) repaintTimer.stop();
            repaint100ms();
            String msg = cancelled ? "Cancelled." : (extraMsg.isEmpty() ? "Complete." : extraMsg);
            statusLabel.setText("▶ " + msg);
            statusLabel.setForeground(cancelled ? YELLOW : GREEN);
            cancelBackBtn.setText("← Back"); cancelBackBtn.setBackground(ACCENT); cancelBackBtn.setEnabled(true);
        });
    }

    public void markError(String msg) {
        SwingUtilities.invokeLater(() -> {
            if (repaintTimer != null) repaintTimer.stop();
            statusLabel.setText("▶ Error: " + msg); statusLabel.setForeground(RED);
            cancelBackBtn.setText("← Back"); cancelBackBtn.setBackground(ACCENT); cancelBackBtn.setEnabled(true);
        });
    }

    public void disableCancelButton() {
        SwingUtilities.invokeLater(() -> {
            cancelBackBtn.setEnabled(false);
            statusLabel.setText("▶ Cancelling…"); statusLabel.setForeground(YELLOW);
        });
    }

    // ── Row builders ──────────────────────────────────────────────────────

    /** Creates and wires all matcher-row widgets, returns a fresh JPanel row. */
    private JPanel buildMatcherRow(int i, String name) {
        String nm = name.length() > 28 ? name.substring(0, 27) + "~" : name;
        matcherNameLabels[i] = monoLabel(padRight(nm, 28));
        matcherNameLabels[i].setForeground(DIM);
        matcherNameLabels[i].setPreferredSize(new Dimension(220, 18));
        matcherNameLabels[i].setMinimumSize(new Dimension(220, 18));
        matcherNameLabels[i].setMaximumSize(new Dimension(220, 18));
        matcherBars[i] = new BarPanel(MATCHER_BAR_W, 14, BAR_FILL, BAR_EMPTY, BAR_BG, false);
        matcherCountLabels[i] = monoLabel("    0/?");
        matcherCountLabels[i].setForeground(DIM);
        matcherCountLabels[i].setPreferredSize(new Dimension(90, 18));
        matcherCountLabels[i].setMinimumSize(new Dimension(90, 18));
        matcherCountLabels[i].setMaximumSize(new Dimension(90, 18));
        matcherStateLabels[i] = monoLabel("◌ pending");
        matcherStateLabels[i].setForeground(DIM);
        return matcherRowPanel(i);
    }

    /** Assembles the stable matcher widgets into a new JPanel row (called on layout rebuild). */
    private JPanel matcherRowPanel(int i) {
        JPanel row = hStack(); row.setBorder(new EmptyBorder(2, 4, 2, 4));
        row.add(matcherNameLabels[i]); row.add(hgap(8));
        row.add(matcherBars[i]);       row.add(hgap(8));
        row.add(matcherCountLabels[i]);row.add(hgap(6));
        row.add(matcherStateLabels[i]);
        return row;
    }

    /** Creates and wires all variant-row widgets for (mi, vi), returns a fresh JPanel row. */
    private JPanel buildVariantRow(int mi, int vi, String name) {
        String nm = name.length() > 26 ? name.substring(0, 25) + "~" : name;
        variantNameLabels[mi][vi] = monoLabel("  ↳ " + padRight(nm, 26));
        variantNameLabels[mi][vi].setForeground(DIM);
        variantNameLabels[mi][vi].setPreferredSize(new Dimension(220, 16));
        variantNameLabels[mi][vi].setMinimumSize(new Dimension(220, 16));
        variantNameLabels[mi][vi].setMaximumSize(new Dimension(220, 16));
        variantBars[mi][vi] = new BarPanel(VARIANT_BAR_W, 12, BAR_FILL, BAR_EMPTY, BAR_BG, false);
        variantCountLabels[mi][vi] = monoLabel("  0/?");
        variantCountLabels[mi][vi].setForeground(DIM);
        variantCountLabels[mi][vi].setPreferredSize(new Dimension(80, 16));
        variantCountLabels[mi][vi].setMinimumSize(new Dimension(80, 16));
        variantCountLabels[mi][vi].setMaximumSize(new Dimension(80, 16));
        variantStateLabels[mi][vi] = monoLabel("◌ pending");
        variantStateLabels[mi][vi].setForeground(DIM);
        variantStateLabels[mi][vi].setFont(SMALL);
        return variantRowPanel(mi, vi);
    }

    /** Assembles the stable variant widgets into a new JPanel row. */
    private JPanel variantRowPanel(int mi, int vi) {
        JPanel row = hStack(); row.setBorder(new EmptyBorder(1, 20, 1, 4));
        row.add(variantNameLabels[mi][vi]); row.add(hgap(8));
        row.add(variantBars[mi][vi]);       row.add(hgap(8));
        row.add(variantCountLabels[mi][vi]);row.add(hgap(6));
        row.add(variantStateLabels[mi][vi]);
        return row;
    }

    // ── EDT repaint ───────────────────────────────────────────────────────

    private void repaint100ms() {
        if (startMs == 0) return;
        int    done    = globalDone.get();
        int    results = globalResultCnt;
        long   elapsed = System.currentTimeMillis() - startMs;
        double pct     = globalTotal > 0 ? done * 100.0 / globalTotal : 0;
        long   eta     = (elapsed > 0 && pct > 0 && pct < 100)
                         ? (long) ((elapsed / pct) * (100 - pct) / 1000) : 0;

        globalBar.setFraction(pct / 100.0);
        globalBar.setLabel(String.format("%.1f%%", pct));
        globalBar.setFillColor(pct >= 100 ? GREEN : pct >= 50 ? BAR_FILL : YELLOW);

        String pairsText = String.format("Pairs  %,d / %,d  |  elapsed %ds", done, globalTotal, elapsed / 1000);
        if (pct > 0 && pct < 100) pairsText += String.format("  |  ETA ~%ds", eta);
        pairsText += String.format("  |  results %,d", results);
        pairsLabel.setText(pairsText);

        if (matcherNameLabels == null) return;
        for (int i = 0; i < matcherNameLabels.length; i++) {
            int     tot = totalPairsPerMatcher != null ? totalPairsPerMatcher[i] : 0;
            int     cnt = pairsDone.getOrDefault(i, new AtomicInteger(0)).get();
            boolean dn  = tot > 0 && cnt >= tot;

            matcherNameLabels[i].setForeground(dn ? GREEN : DIM);
            matcherBars[i].setFraction(tot > 0 ? (double) cnt / tot : 0);
            matcherBars[i].setFillColor(dn ? GREEN : BAR_FILL);
            matcherCountLabels[i].setText(padLeft(String.valueOf(cnt), 4) + "/" + (tot > 0 ? tot : "?"));
            matcherCountLabels[i].setForeground(dn ? GREEN : DIM);
            if (!matcherStateLabels[i].getText().startsWith("✓")
                    && !matcherStateLabels[i].getText().startsWith("❌")
                    && !matcherStateLabels[i].getText().startsWith("⚠")) {
                matcherStateLabels[i].setText(dn ? "✓ done   " : cnt > 0 ? "◌ running" : "◌ pending");
                matcherStateLabels[i].setForeground(dn ? GREEN : cnt > 0 ? YELLOW : DIM);
            }

            // Variant sub-rows
            if (variantNameLabels[i] == null) continue;
            for (int vi = 0; vi < variantNameLabels[i].length; vi++) {
                if (variantNameLabels[i][vi] == null) continue;
                int     vtot = totalPairsPerVariant != null
                        && totalPairsPerVariant[i] != null
                        && vi < totalPairsPerVariant[i].length
                        ? totalPairsPerVariant[i][vi] : 0;
                int     vcnt = variantDone.getOrDefault(variantKey(i, vi), new AtomicInteger(0)).get();
                boolean vdn  = vtot > 0 && vcnt >= vtot;

                variantNameLabels[i][vi].setForeground(vdn ? GREEN : vcnt > 0 ? YELLOW : DIM);
                variantBars[i][vi].setFraction(vtot > 0 ? (double) vcnt / vtot : 0);
                variantBars[i][vi].setFillColor(vdn ? GREEN : BAR_FILL);
                variantCountLabels[i][vi].setText(padLeft(String.valueOf(vcnt), 3) + "/" + (vtot > 0 ? vtot : "?"));
                variantCountLabels[i][vi].setForeground(vdn ? GREEN : vcnt > 0 ? YELLOW : DIM);
                if (!variantStateLabels[i][vi].getText().startsWith("✓")) {
                    variantStateLabels[i][vi].setText(vdn ? "✓ done" : vcnt > 0 ? "◌ running" : "◌ pending");
                    variantStateLabels[i][vi].setForeground(vdn ? GREEN : vcnt > 0 ? YELLOW : DIM);
                }
            }
        }
        String s = postStatusText;
        if (!s.isEmpty()) statusLabel.setText("▶ " + s);
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private void resetArrays(int n) {
        matcherNameLabels  = new JLabel[n];   matcherBars        = new BarPanel[n];
        matcherCountLabels = new JLabel[n];   matcherStateLabels = new JLabel[n];
        variantNameLabels  = new JLabel[n][];  variantBars        = new BarPanel[n][];
        variantCountLabels = new JLabel[n][];  variantStateLabels = new JLabel[n][];
        totalPairsPerMatcher = new int[n];     totalPairsPerVariant = new int[n][];
    }

    private static int    variantKey(int mi, int vi)  { return mi * 10_000 + vi; }
    private static String padRight(String s, int w)   { return s.length() >= w ? s : s + " ".repeat(w - s.length()); }
    private static String padLeft(String s, int w)    { return s.length() >= w ? s : " ".repeat(w - s.length()) + s; }

    // ── BarPanel ──────────────────────────────────────────────────────────

    /**
     * Custom-painted progress bar.  Copied from ProgressDisplay so the UI
     * package is self-contained.
     */
    public static final class BarPanel extends JPanel {
        private double fraction = 0; private String barLabel = ""; private Color fillColor;
        private final Color emptyColor, bgColor; private final int fw, fh; private final boolean showLabel;

        public BarPanel(int w, int h, Color fill, Color empty, Color bg, boolean showLabel) {
            this.fw = w; this.fh = h; this.fillColor = fill; this.emptyColor = empty;
            this.bgColor = bg; this.showLabel = showLabel;
            Dimension d = new Dimension(w, h); setPreferredSize(d); setMinimumSize(d); setMaximumSize(d);
            setOpaque(false);
        }
        public void setFraction(double f) { this.fraction = Math.max(0, Math.min(1, f)); repaint(); }
        public void setLabel(String l)    { this.barLabel = l == null ? "" : l; }
        public void setFillColor(Color c) { this.fillColor = c; }

        @Override protected void paintComponent(Graphics g) {
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING,      RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
            int arc = 5;
            g2.setColor(bgColor);    g2.fillRoundRect(0, 0, fw, fh, arc, arc);
            g2.setColor(emptyColor); g2.fillRoundRect(0, 0, fw, fh, arc, arc);
            int fw2 = (int) Math.round(fraction * fw);
            if (fw2 > 0) {
                g2.setColor(fillColor);
                if (fw2 >= fw) g2.fillRoundRect(0, 0, fw, fh, arc, arc);
                else { g2.fillRoundRect(0, 0, fw2 + arc, fh, arc, arc); g2.fillRect(fw2, 0, 1, fh); }
            }
            if (showLabel && !barLabel.isEmpty()) {
                Font f = new Font(Font.MONOSPACED, Font.BOLD, 11); g2.setFont(f);
                FontMetrics fm = g2.getFontMetrics();
                int tx = (fw - fm.stringWidth(barLabel)) / 2, ty = (fh + fm.getAscent() - fm.getDescent()) / 2;
                g2.setColor(new Color(0, 0, 0, 180)); g2.drawString(barLabel, tx + 1, ty + 1);
                g2.setColor(Color.WHITE);              g2.drawString(barLabel, tx, ty);
            }
            g2.dispose();
        }
    }
}

