package org.example.ui;

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

    // ── Widgets ───────────────────────────────────────────────────────────

    private final JLabel  headerLabel;
    private final BarPanel globalBar;
    private final JLabel  pairsLabel;
    private final JPanel  matcherRows;
    private final JLabel  statusLabel;
    private final JButton cancelBackBtn;

    private JLabel[]  matcherNameLabels;
    private BarPanel[] matcherBars;
    private JLabel[]  matcherCountLabels;
    private JLabel[]  matcherStateLabels;

    // ── Run state (written from worker thread, read on EDT via Timer) ──────
    private final ConcurrentHashMap<Integer, AtomicInteger> pairsDone   = new ConcurrentHashMap<>();
    private volatile int[]   totalPairsPerMatcher;
    private volatile int     globalTotal      = 0;
    private final AtomicInteger globalDone    = new AtomicInteger(0);
    private volatile int     globalResultCnt  = 0;
    private volatile long    startMs          = 0;
    private volatile String  postStatusText   = "";

    private javax.swing.Timer repaintTimer;

    // ── Constructor ───────────────────────────────────────────────────────

    public ProgressPanel(Runnable onCancelOrBack) {
        setLayout(new BorderLayout()); setBackground(BG);
        setBorder(new EmptyBorder(20,24,16,24));

        JPanel inner = new JPanel(); inner.setLayout(new BoxLayout(inner, BoxLayout.Y_AXIS));
        inner.setBackground(BG);

        headerLabel = label("", BOLD13, HEADER); inner.add(headerLabel); inner.add(vgap(12));
        inner.add(label("Progress", SMALL, DIM)); inner.add(vgap(4));
        globalBar = new BarPanel(GLOBAL_BAR_W, 24, BAR_FILL, BAR_EMPTY, BAR_BG, true);
        globalBar.setAlignmentX(Component.LEFT_ALIGNMENT);
        inner.add(globalBar); inner.add(vgap(10));
        pairsLabel = monoLabel("Pairs  0 / 0"); inner.add(pairsLabel); inner.add(vgap(12));

        inner.add(sepLine("── per-matcher ")); inner.add(vgap(6));
        matcherRows = new JPanel(); matcherRows.setLayout(new BoxLayout(matcherRows, BoxLayout.Y_AXIS));
        matcherRows.setBackground(BG); matcherRows.setAlignmentX(Component.LEFT_ALIGNMENT);
        JScrollPane ms = scrollPane(matcherRows);
        ms.setBackground(BG); ms.getViewport().setBackground(BG);
        ms.setBorder(BorderFactory.createLineBorder(SEP,1));
        ms.setAlignmentX(Component.LEFT_ALIGNMENT);
        ms.setPreferredSize(new Dimension(0,320)); ms.setMaximumSize(new Dimension(Integer.MAX_VALUE,320));
        inner.add(ms); inner.add(vgap(10));

        inner.add(sepLine("── post-processing ")); inner.add(vgap(5));
        statusLabel = label("", SMALL, STATUS_LN); inner.add(statusLabel);
        add(inner, BorderLayout.CENTER);

        JPanel bottom = new JPanel(new FlowLayout(FlowLayout.RIGHT,8,4));
        bottom.setBackground(BG); bottom.setBorder(new EmptyBorder(12,0,0,0));
        cancelBackBtn = accentBtn("✕  Cancel", BTN_RED);
        cancelBackBtn.addActionListener(e -> onCancelOrBack.run());
        bottom.add(cancelBackBtn); add(bottom, BorderLayout.SOUTH);
    }

    // ── Initialise for a new run ──────────────────────────────────────────

    /**
     * Resets all state and populates per-matcher rows.  Must be called on EDT
     * before switching to this panel.
     *
     * @param matcherNames display names for each matcher (in run order)
     * @param nThreads     thread count — shown in header
     * @param estimatedTotal rough estimate of total pairs (updated when catalogue loads)
     */
    public void startRun(String[] matcherNames, int nThreads, int estimatedTotal) {
        int n = matcherNames.length;
        matcherNameLabels  = new JLabel[n];
        matcherBars        = new BarPanel[n];
        matcherCountLabels = new JLabel[n];
        matcherStateLabels = new JLabel[n];
        totalPairsPerMatcher = new int[n];
        pairsDone.clear();
        globalDone.set(0); globalTotal = Math.max(estimatedTotal,1); globalResultCnt = 0;
        startMs = System.currentTimeMillis(); postStatusText = "";

        headerLabel.setText("[RUN]  "+n+" matcher"+(n==1?"":"s")+"  |  "+nThreads+" threads");
        globalBar.setFraction(0); globalBar.setLabel("0.0%");
        pairsLabel.setText("Pairs  0 / ?");
        statusLabel.setText(""); statusLabel.setForeground(STATUS_LN);
        cancelBackBtn.setText("✕  Cancel"); cancelBackBtn.setBackground(BTN_RED); cancelBackBtn.setEnabled(true);

        matcherRows.removeAll();
        for (int i = 0; i < n; i++) {
            pairsDone.put(i, new AtomicInteger(0));
            JPanel row = hStack(); row.setBorder(new EmptyBorder(2,4,2,4));
            String nm = matcherNames[i]; if (nm.length()>28) nm=nm.substring(0,27)+"~";
            matcherNameLabels[i] = monoLabel(padRight(nm,28)); matcherNameLabels[i].setForeground(DIM);
            matcherNameLabels[i].setPreferredSize(new Dimension(220,18));
            matcherNameLabels[i].setMinimumSize(new Dimension(220,18));
            matcherNameLabels[i].setMaximumSize(new Dimension(220,18));
            matcherBars[i] = new BarPanel(MATCHER_BAR_W,14,BAR_FILL,BAR_EMPTY,BAR_BG,false);
            matcherCountLabels[i] = monoLabel("    0/?"); matcherCountLabels[i].setForeground(DIM);
            matcherCountLabels[i].setPreferredSize(new Dimension(90,18));
            matcherCountLabels[i].setMinimumSize(new Dimension(90,18));
            matcherCountLabels[i].setMaximumSize(new Dimension(90,18));
            matcherStateLabels[i] = monoLabel("◌ pending"); matcherStateLabels[i].setForeground(DIM);
            row.add(matcherNameLabels[i]); row.add(hgap(8));
            row.add(matcherBars[i]);       row.add(hgap(8));
            row.add(matcherCountLabels[i]);row.add(hgap(6));
            row.add(matcherStateLabels[i]);
            matcherRows.add(row);
        }
        matcherRows.revalidate(); matcherRows.repaint();

        if (repaintTimer != null) repaintTimer.stop();
        repaintTimer = new javax.swing.Timer(100, ev -> repaint100ms());
        repaintTimer.start();
    }

    /**
     * Resets state for a simple single-task generation run (no per-matcher rows).
     * Progress is driven via {@link #setGlobalTotal(int)} and {@link #setGlobalDone(int)}.
     */
    public void startSingleTask(String label) {
        matcherNameLabels    = new JLabel[0]; matcherBars = new BarPanel[0];
        matcherCountLabels   = new JLabel[0]; matcherStateLabels = new JLabel[0];
        totalPairsPerMatcher = new int[0];
        pairsDone.clear(); globalDone.set(0); globalTotal = 1; globalResultCnt = 0;
        startMs = System.currentTimeMillis(); postStatusText = "";
        headerLabel.setText(label);
        globalBar.setFraction(0); globalBar.setLabel("0.0%");
        pairsLabel.setText(""); statusLabel.setText(""); statusLabel.setForeground(STATUS_LN);
        cancelBackBtn.setText("✕  Cancel"); cancelBackBtn.setBackground(BTN_RED); cancelBackBtn.setEnabled(true);
        matcherRows.removeAll(); matcherRows.revalidate(); matcherRows.repaint();
        if (repaintTimer != null) repaintTimer.stop();
        repaintTimer = new javax.swing.Timer(100, ev -> repaint100ms());
        repaintTimer.start();
    }

    /** Sets the global done counter to an absolute value (for single-task runs). */
    public void setGlobalDone(int done) { globalDone.set(done); }

    /** Special header for the benchmark-collation mode. */
    public void startBenchmark() {
        matcherNameLabels  = new JLabel[0]; matcherBars = new BarPanel[0];
        matcherCountLabels = new JLabel[0]; matcherStateLabels = new JLabel[0];
        totalPairsPerMatcher = new int[0];
        pairsDone.clear(); globalDone.set(0); globalTotal=1; globalResultCnt=0;
        startMs = System.currentTimeMillis(); postStatusText="";
        headerLabel.setText("[BENCHMARK]  Collating all technique reports…");
        globalBar.setFraction(0); globalBar.setLabel("0.0%");
        pairsLabel.setText(""); statusLabel.setText(""); statusLabel.setForeground(STATUS_LN);
        cancelBackBtn.setText("✕  Cancel"); cancelBackBtn.setBackground(BTN_RED); cancelBackBtn.setEnabled(true);
        matcherRows.removeAll(); matcherRows.revalidate(); matcherRows.repaint();
        if (repaintTimer!=null) repaintTimer.stop();
        repaintTimer = new javax.swing.Timer(100, ev -> repaint100ms());
        repaintTimer.start();
    }

    // ── Thread-safe update methods ────────────────────────────────────────

    public void setGlobalTotal(int total) { this.globalTotal = Math.max(total,1); }
    public void setMatcherTotal(int idx, int total) {
        if (totalPairsPerMatcher!=null && idx<totalPairsPerMatcher.length)
            totalPairsPerMatcher[idx] = total;
    }
    public void incrementDone(int matcherIdx) {
        pairsDone.getOrDefault(matcherIdx, new AtomicInteger(0)).incrementAndGet();
        globalDone.incrementAndGet();
    }
    public void incrementResults() { synchronized(this) { globalResultCnt++; } }
    public void setPostStatus(String text) { postStatusText = text; }
    public void setMatcherRunning(int idx) {
        SwingUtilities.invokeLater(() -> {
            if (matcherStateLabels!=null && idx<matcherStateLabels.length) {
                matcherStateLabels[idx].setText("◌ running"); matcherStateLabels[idx].setForeground(YELLOW);
            }
        });
    }
    public void setMatcherDone(int idx, double acc) {
        SwingUtilities.invokeLater(() -> {
            if (matcherStateLabels!=null && idx<matcherStateLabels.length) {
                matcherStateLabels[idx].setText(String.format("✓ done  %.0f%% acc", acc));
                matcherStateLabels[idx].setForeground(GREEN);
                matcherNameLabels[idx].setForeground(GREEN);
            }
        });
    }
    public void setMatcherError(int idx) {
        SwingUtilities.invokeLater(() -> {
            if (matcherStateLabels!=null && idx<matcherStateLabels.length) {
                matcherStateLabels[idx].setText("❌ error"); matcherStateLabels[idx].setForeground(RED);
            }
        });
    }
    public void setMatcherSkipped(int idx) {
        SwingUtilities.invokeLater(() -> {
            if (matcherStateLabels!=null && idx<matcherStateLabels.length) {
                matcherStateLabels[idx].setText("⚠ skipped"); matcherStateLabels[idx].setForeground(YELLOW);
            }
        });
    }

    /** Called when the run finishes (success, cancel, or error). */
    public void markFinished(boolean cancelled, String extraMsg) {
        SwingUtilities.invokeLater(() -> {
            if (repaintTimer!=null) repaintTimer.stop();
            repaint100ms(); // final repaint at 100%
            String msg = cancelled ? "Cancelled." : (extraMsg.isEmpty() ? "Complete." : extraMsg);
            statusLabel.setText("▶ " + msg);
            statusLabel.setForeground(cancelled ? YELLOW : GREEN);
            cancelBackBtn.setText("← Back"); cancelBackBtn.setBackground(ACCENT); cancelBackBtn.setEnabled(true);
        });
    }

    public void markError(String msg) {
        SwingUtilities.invokeLater(() -> {
            if (repaintTimer!=null) repaintTimer.stop();
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

    // ── EDT repaint ───────────────────────────────────────────────────────

    private void repaint100ms() {
        if (startMs == 0) return;
        int    done    = globalDone.get();
        int    results = globalResultCnt;
        long   elapsed = System.currentTimeMillis() - startMs;
        double pct     = globalTotal > 0 ? done*100.0/globalTotal : 0;
        long   eta     = (elapsed>0 && pct>0 && pct<100) ? (long)((elapsed/pct)*(100-pct)/1000) : 0;

        globalBar.setFraction(pct/100.0);
        globalBar.setLabel(String.format("%.1f%%", pct));
        globalBar.setFillColor(pct>=100 ? GREEN : pct>=50 ? BAR_FILL : YELLOW);

        StringBuilder sb = new StringBuilder();
        sb.append(String.format("Pairs  %,d / %,d", done, globalTotal));
        sb.append(String.format("  |  elapsed %ds", elapsed/1000));
        if (pct>0 && pct<100) sb.append(String.format("  |  ETA ~%ds", eta));
        sb.append(String.format("  |  results %,d", results));
        pairsLabel.setText(sb.toString());

        if (matcherNameLabels != null) {
            for (int i = 0; i < matcherNameLabels.length; i++) {
                int    tot  = totalPairsPerMatcher!=null ? totalPairsPerMatcher[i] : 0;
                int    cnt  = pairsDone.getOrDefault(i, new AtomicInteger(0)).get();
                boolean dn  = tot>0 && cnt>=tot;
                matcherNameLabels[i].setForeground(dn ? GREEN : DIM);
                matcherBars[i].setFraction(tot>0?(double)cnt/tot:0);
                matcherBars[i].setFillColor(dn ? GREEN : BAR_FILL);
                matcherCountLabels[i].setText(padLeft(String.valueOf(cnt),4)+"/"+(tot>0?tot:"?"));
                matcherCountLabels[i].setForeground(dn ? GREEN : DIM);
                if (!matcherStateLabels[i].getText().startsWith("✓") &&
                    !matcherStateLabels[i].getText().startsWith("❌") &&
                    !matcherStateLabels[i].getText().startsWith("⚠")) {
                    matcherStateLabels[i].setText(dn?"✓ done   ":cnt>0?"◌ running":"◌ pending");
                    matcherStateLabels[i].setForeground(dn?GREEN:cnt>0?YELLOW:DIM);
                }
            }
        }
        String s = postStatusText;
        if (!s.isEmpty()) { statusLabel.setText("▶ "+s); }
    }

    // ── Padding helpers ───────────────────────────────────────────────────

    private static String padRight(String s, int w) {
        return s.length()>=w ? s : s+" ".repeat(w-s.length());
    }
    private static String padLeft(String s, int w) {
        return s.length()>=w ? s : " ".repeat(w-s.length())+s;
    }

    // ── BarPanel ─────────────────────────────────────────────────────────

    /**
     * Custom-painted progress bar.  Copied from ProgressDisplay so the UI
     * package is self-contained.
     */
    public static final class BarPanel extends JPanel {
        private double fraction=0; private String barLabel=""; private Color fillColor;
        private final Color emptyColor, bgColor; private final int fw, fh; private final boolean showLabel;

        public BarPanel(int w, int h, Color fill, Color empty, Color bg, boolean showLabel) {
            this.fw=w; this.fh=h; this.fillColor=fill; this.emptyColor=empty;
            this.bgColor=bg; this.showLabel=showLabel;
            Dimension d=new Dimension(w,h); setPreferredSize(d); setMinimumSize(d); setMaximumSize(d);
            setOpaque(false);
        }
        public void setFraction(double f) { this.fraction=Math.max(0,Math.min(1,f)); repaint(); }
        public void setLabel(String l)    { this.barLabel=l==null?"":l; }
        public void setFillColor(Color c) { this.fillColor=c; }

        @Override protected void paintComponent(Graphics g) {
            Graphics2D g2=(Graphics2D)g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING,      RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
            int arc=5;
            g2.setColor(bgColor);    g2.fillRoundRect(0,0,fw,fh,arc,arc);
            g2.setColor(emptyColor); g2.fillRoundRect(0,0,fw,fh,arc,arc);
            int fw2=(int)Math.round(fraction*fw);
            if (fw2>0) {
                g2.setColor(fillColor);
                if (fw2>=fw) g2.fillRoundRect(0,0,fw,fh,arc,arc);
                else { g2.fillRoundRect(0,0,fw2+arc,fh,arc,arc); g2.fillRect(fw2,0,1,fh); }
            }
            if (showLabel && !barLabel.isEmpty()) {
                Font f=new Font(Font.MONOSPACED,Font.BOLD,11); g2.setFont(f);
                FontMetrics fm=g2.getFontMetrics();
                int tx=(fw-fm.stringWidth(barLabel))/2, ty=(fh+fm.getAscent()-fm.getDescent())/2;
                g2.setColor(new Color(0,0,0,180)); g2.drawString(barLabel,tx+1,ty+1);
                g2.setColor(Color.WHITE);           g2.drawString(barLabel,tx,ty);
            }
            g2.dispose();
        }
    }
}

