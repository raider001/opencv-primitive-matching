package org.example.utilities;

import org.example.factories.ReferenceId;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;

/**
 * Progress display for the analytical test double-loop.
 *
 * <p>Shows a Swing window with a global progress bar:
 * <pre>
 *  [TAG]  N,NNN pairs  |  T threads
 *  Progress  [████████████░░░░░░░░]  47.3%
 *  Pairs  1,234 / 2,610  |  elapsed 48s  |  ETA ~54s  |  885 results
 *  ── post-processing ────────────────────────────────────
 *  ▶ Computing verdicts...
 * </pre>
 *
 * <p>Falls back to plain stderr line output when headless (e.g. CI servers).
 */
public final class ProgressDisplay {

    // ── Colour palette (matches original ANSI colours) ────────────────────
    private static final Color C_BG        = new Color(0x0d, 0x11, 0x17);
    private static final Color C_HEADER    = new Color(0x58, 0xa6, 0xff);
    private static final Color C_WHITE     = new Color(0xc9, 0xd1, 0xd9);
    private static final Color C_DIM       = new Color(0x58, 0x62, 0x6e);
    private static final Color C_GREEN     = new Color(0x56, 0xd3, 0x64);
    private static final Color C_YELLOW    = new Color(0xd2, 0x99, 0x22);
    private static final Color C_BAR_FILL  = new Color(0x38, 0x8b, 0xff);
    private static final Color C_BAR_EMPTY = new Color(0x21, 0x26, 0x2d);
    private static final Color C_BAR_BG    = new Color(0x16, 0x1b, 0x22);
    private static final Color C_SEP       = new Color(0x30, 0x36, 0x3d);
    private static final Color C_STATUS    = new Color(0x79, 0xc0, 0xff);

    private static final Font MONO      = new Font(Font.MONOSPACED, Font.PLAIN,  13);
    private static final Font MONO_BOLD = new Font(Font.MONOSPACED, Font.BOLD,   13);
    private static final Font MONO_SM   = new Font(Font.MONOSPACED, Font.PLAIN,  12);

    private static final int BAR_W = 320;   // global progress bar width (px)

    // ── State ──────────────────────────────────────────────────────────────
    private final String  tag;
    private final int     totalPairs;
    private final int     numThreads;
    private final long    tStart;
    private final boolean headless;

    private volatile int     lastDone    = 0;
    private volatile int     lastResults = 0;
    private volatile long    lastElapsed = 0;
    private volatile String  statusLine  = "";
    private volatile boolean initialised = false;

    // ── Swing widgets ──────────────────────────────────────────────────────
    private JFrame   frame;
    private BarPanel globalBar;
    private JLabel   pairsLabel;
    private JLabel   statusLabel;
    private Timer    repaintTimer;

    // ── Constructor ────────────────────────────────────────────────────────

    @SuppressWarnings("unused") // refs and scenesPerRef kept for call-site compatibility
    public ProgressDisplay(String tag, ReferenceId[] refs, int totalPairs,
                           int scenesPerRef, int numThreads, long tStart) {
        this.tag        = tag;
        this.totalPairs = totalPairs;
        this.numThreads = numThreads;
        this.tStart     = tStart;
        this.headless   = GraphicsEnvironment.isHeadless();
    }

    // ── Public API ─────────────────────────────────────────────────────────

    public void start() {
        initialised = true;
        if (headless) {
            stderr("[%s] Starting: %,d pairs  |  threads: %d%n", tag, totalPairs, numThreads);
            return;
        }
        SwingUtilities.invokeLater(this::buildFrame);
    }

    @SuppressWarnings("unused") // refId kept for call-site compatibility
    public void update(ReferenceId refId, int totalDone, int resultCount) {
        if (!initialised) return;
        lastDone    = totalDone;
        lastResults = resultCount;
        lastElapsed = System.currentTimeMillis() - tStart;
        if (headless) {
            int every = Math.max(1, totalPairs / 20);
            if (totalDone % every == 0 || totalDone == totalPairs) {
                double pct = totalDone * 100.0 / totalPairs;
                long eta   = lastElapsed > 0 && pct > 0
                        ? (long)((lastElapsed / pct) * (100.0 - pct) / 1000) : 0;
                stderr("[%s] %5.1f%%  %,d/%,d  elapsed %ds  ETA ~%ds  results %,d%n",
                        tag, pct, totalDone, totalPairs,
                        lastElapsed / 1000, eta, resultCount);
            }
        }
    }

    public void finish(int totalDone, int resultCount) {
        status("");
        lastDone    = totalDone;
        lastResults = resultCount;
        lastElapsed = System.currentTimeMillis() - tStart;
        if (headless) {
            stderr("[%s] Complete: %,d results in %.1f s%n%n",
                    tag, resultCount, lastElapsed / 1000.0);
            return;
        }
        SwingUtilities.invokeLater(() -> {
            if (repaintTimer != null) repaintTimer.stop();
            repaintAll();
        });
    }

    public void status(String step) {
        if (!initialised) return;
        statusLine = step == null ? "" : step;
        if (headless && step != null && !step.isEmpty())
            stderr("[%s] %s%n", tag, step);
    }

    // ── Frame construction ─────────────────────────────────────────────────

    private void buildFrame() {
        frame = new JFrame("[" + tag + "] Pattern Matching — Progress");
        frame.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
        frame.getContentPane().setBackground(C_BG);

        JPanel root = new JPanel();
        root.setLayout(new BoxLayout(root, BoxLayout.Y_AXIS));
        root.setBackground(C_BG);
        root.setBorder(new EmptyBorder(14, 18, 14, 18));

        // ── Header ─────────────────────────────────────────────────────────
        JLabel headerLabel = label(
                "[" + tag + "]  " + String.format("%,d", totalPairs) + " pairs  |  " + numThreads + " threads",
                MONO_BOLD, C_HEADER);
        root.add(headerLabel);
        root.add(vgap(10));

        // ── Global progress bar + percentage ──────────────────────────────
        root.add(label("Progress", MONO_SM, C_DIM));
        root.add(vgap(3));
        globalBar = new BarPanel(BAR_W, 24, C_BAR_FILL, C_BAR_EMPTY, C_BAR_BG, true);
        globalBar.setAlignmentX(Component.LEFT_ALIGNMENT);
        root.add(globalBar);
        root.add(vgap(8));

        // ── Pairs / elapsed / ETA / results ───────────────────────────────
        pairsLabel = label("Pairs  0 / " + String.format("%,d", totalPairs), MONO, C_WHITE);
        root.add(pairsLabel);
        root.add(vgap(10));

        // ── Post-processing separator + status ────────────────────────────
        root.add(separator());
        root.add(vgap(5));
        statusLabel = label("", MONO, C_STATUS);
        root.add(statusLabel);

        frame.setContentPane(root);
        frame.pack();
        frame.setMinimumSize(new Dimension(560, 160));
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        repaintTimer = new Timer(100, e -> repaintAll());
        repaintTimer.start();
    }

    // ── Repaint (called on EDT by timer) ───────────────────────────────────

    private void repaintAll() {
        int    done    = lastDone;
        int    results = lastResults;
        long   elapsed = lastElapsed;
        double pct     = totalPairs > 0 ? done * 100.0 / totalPairs : 0;
        long   etaSec  = (elapsed > 0 && pct > 0 && pct < 100)
                       ? (long)((elapsed / pct) * (100.0 - pct) / 1000) : 0;

        globalBar.setFraction(pct / 100.0);
        globalBar.setLabel(String.format("%.1f%%", pct));
        globalBar.setFillColor(pct >= 100 ? C_GREEN : pct >= 50 ? C_BAR_FILL : C_YELLOW);

        StringBuilder sb = new StringBuilder();
        sb.append(String.format("Pairs  %,d / %,d", done, totalPairs));
        sb.append(String.format("  |  elapsed %ds", elapsed / 1000));
        if (pct > 0 && pct < 100)
            sb.append(String.format("  |  ETA ~%ds", etaSec));
        sb.append(String.format("  |  results %,d", results));
        pairsLabel.setText(sb.toString());

        String s = statusLine;
        statusLabel.setText(s.isEmpty() ? "" : "▶ " + s);

        frame.repaint();
    }

    // ── Custom bar component ───────────────────────────────────────────────

    /**
     * Filled progress bar rendered via paintComponent — styled to match the
     * ANSI block-character bars from the original terminal display.
     */
    private static final class BarPanel extends JPanel {
        private double  fraction  = 0.0;
        private String  label     = "";
        private Color   fillColor;
        private final Color  emptyColor;
        private final Color  bgColor;
        private final int    fixedW, fixedH;
        private final boolean showLabel;

        BarPanel(int w, int h, Color fill, Color empty, Color bg, boolean showLabel) {
            this.fixedW     = w;
            this.fixedH     = h;
            this.fillColor  = fill;
            this.emptyColor = empty;
            this.bgColor    = bg;
            this.showLabel  = showLabel;
            Dimension d = new Dimension(w, h);
            setPreferredSize(d); setMinimumSize(d); setMaximumSize(d);
            setOpaque(false);
        }

        void setFraction(double f)  { this.fraction  = Math.max(0, Math.min(1, f)); repaint(); }
        void setLabel(String l)     { this.label = l == null ? "" : l; }
        void setFillColor(Color c)  { this.fillColor = c; }

        @Override
        protected void paintComponent(Graphics g) {
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING,      RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

            int w = fixedW, h = fixedH, arc = 5;

            // Track background
            g2.setColor(bgColor);
            g2.fillRoundRect(0, 0, w, h, arc, arc);

            // Empty track
            g2.setColor(emptyColor);
            g2.fillRoundRect(0, 0, w, h, arc, arc);

            // Filled portion
            int fillW = (int) Math.round(fraction * w);
            if (fillW > 0) {
                g2.setColor(fillColor);
                // Use plain rect for the right edge so it doesn't look rounded mid-bar
                if (fillW >= w) {
                    g2.fillRoundRect(0, 0, w, h, arc, arc);
                } else {
                    g2.fillRoundRect(0, 0, fillW + arc, h, arc, arc);
                    g2.fillRect(fillW, 0, 1, h); // square off the right edge
                }
            }

            // Percentage label centred inside the bar (global bar only)
            if (showLabel && !label.isEmpty()) {
                Font f = new Font(Font.MONOSPACED, Font.BOLD, 11);
                g2.setFont(f);
                FontMetrics fm = g2.getFontMetrics();
                int tx = (w - fm.stringWidth(label)) / 2;
                int ty = (h + fm.getAscent() - fm.getDescent()) / 2;
                g2.setColor(new Color(0, 0, 0, 180));
                g2.drawString(label, tx + 1, ty + 1);
                g2.setColor(Color.WHITE);
                g2.drawString(label, tx, ty);
            }

            g2.dispose();
        }
    }

    // ── Factory / layout helpers ───────────────────────────────────────────

    private static JLabel label(String text, Font font, Color fg) {
        JLabel l = new JLabel(text);
        l.setFont(font);
        l.setForeground(fg);
        l.setAlignmentX(Component.LEFT_ALIGNMENT);
        return l;
    }

    private static Component vgap(int h) { return Box.createRigidArea(new Dimension(0, h)); }

    /** Separator row: dim label on the left, coloured line stretching right. */
    private JPanel separator() {
        JPanel p = new JPanel(new BorderLayout(6, 0));
        p.setBackground(C_BG);
        p.setAlignmentX(Component.LEFT_ALIGNMENT);
        p.setMaximumSize(new Dimension(Integer.MAX_VALUE, 16));
        JLabel lbl = label("── post-processing ", MONO_SM, C_DIM);
        p.add(lbl, BorderLayout.WEST);
        JSeparator sep = new JSeparator(JSeparator.HORIZONTAL);
        sep.setForeground(C_SEP);
        sep.setBackground(C_BG);
        p.add(sep, BorderLayout.CENTER);
        return p;
    }

    private static void stderr(String fmt, Object... args) {
        System.err.printf(fmt, args);
        System.err.flush();
    }
}
