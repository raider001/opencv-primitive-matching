package org.example.ui;

import org.example.*;
import org.example.analytics.*;
import org.example.factories.BackgroundFactory;
import org.example.factories.BackgroundId;
import org.example.factories.ReferenceId;
import org.example.factories.ReferenceImageFactory;
import org.example.scene.*;
import org.example.ui.panels.*;

import javax.swing.*;
import javax.swing.border.*;
import java.awt.*;
import java.awt.event.*;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.List;
import java.util.concurrent.*;

import static org.example.ui.Palette.*;
import static org.example.ui.Widgets.*;

/**
 * Main application window for the Pattern Matching Benchmark Launcher.
 *
 * <h2>Wizard flow</h2>
 * <ol>
 *   <li>{@link MatchersPanel}    — pick which technique to run</li>
 *   <li>{@link VariantsPanel}    — pick which variants of the selected matcher</li>
 *   <li>{@link ReferencesPanel}  — pick which reference shapes to use</li>
 *   <li>{@link BackgroundsPanel} — pick which background images</li>
 *   <li>{@link ScenesPanel}      — pick which scene transforms</li>
 * </ol>
 *
 * <h2>Views</h2>
 * <ul>
 *   <li>{@code SELECT}   — the wizard + action buttons</li>
 *   <li>{@code PROGRESS} — replaces the whole window during a run</li>
 * </ul>
 */
public final class BenchmarkLauncher extends JFrame {

    // ── Card names (bottom-right slot only) ───────────────────────────────
    private static final String CARD_SUMMARY  = "SUMMARY";
    private static final String CARD_PROGRESS = "PROGRESS";

    // ── Bottom-right card (summary ↔ progress) ─────────────────────────────
    private final CardLayout bottomCardLayout = new CardLayout();
    private final JPanel     bottomCardPanel  = new JPanel(bottomCardLayout);

    // ── Options ────────────────────────────────────────────────────────────
    private JCheckBox optClear;
    private JCheckBox optNegatives;

    // ── Shared context + run configuration ────────────────────────────────
    private final WizardContext    ctx = new WizardContext();
    private final RunConfiguration cfg = new RunConfiguration();

    // ── Wizard panels ──────────────────────────────────────────────────────
    private final MatchersPanel    matchersPanel;
    private final VariantsPanel    variantsPanel;
    private final ReferencesPanel  referencesPanel;
    private final BackgroundsPanel backgroundsPanel;
    private final ScenesPanel      scenesPanel;
    private final RunSummaryPanel  runSummaryPanel;

    // ── Progress panel ─────────────────────────────────────────────────────
    private final ProgressPanel progressPanel;

    // ── Run state ──────────────────────────────────────────────────────────
    private volatile boolean cancelled = false;
    private volatile Thread  runThread = null;

    // ── Cascade debounce — coalesces rapid variant-checkbox changes ────────
    private final javax.swing.Timer cascadeDebounce = new javax.swing.Timer(150, e -> rebuildReferences());
    { cascadeDebounce.setRepeats(false); }

    // =========================================================================
    //  Constructor
    // =========================================================================

    public BenchmarkLauncher() {
        super("Pattern Matching — Benchmark Launcher");
        setUndecorated(true);                          // drop the OS title bar
        setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        getContentPane().setBackground(BG);
        // Thin border so the window has a visible edge against any desktop background
        getRootPane().setBorder(BorderFactory.createLineBorder(BORDER, 1));

        ctx.reloadCatalogueFileNames();

        // Build panels — all share cfg and wire cascade callbacks
        runSummaryPanel  = new RunSummaryPanel();
        scenesPanel      = new ScenesPanel(ctx, cfg);
        scenesPanel.setOnCheckChanged(this::refreshSummary);
        backgroundsPanel = new BackgroundsPanel(ctx, cfg, () -> { scenesPanel.rebuild(); refreshSummary(); });
        referencesPanel  = new ReferencesPanel(cfg, () -> { backgroundsPanel.rebuild(); refreshSummary(); });
        variantsPanel    = new VariantsPanel(ctx, cfg, this::onVariantSelectionChanged);
        matchersPanel    = new MatchersPanel(ctx, cfg, this::onMatcherSelectionChanged);
        matchersPanel.setOnCheckChanged(this::refreshSummary);

        progressPanel = new ProgressPanel(this::onCancelOrBack);

        // Bottom-right slot: summary ↔ progress
        bottomCardPanel.setBackground(BG);
        bottomCardPanel.add(runSummaryPanel, CARD_SUMMARY);
        bottomCardPanel.add(progressPanel,   CARD_PROGRESS);
        bottomCardLayout.show(bottomCardPanel, CARD_SUMMARY);

        setContentPane(buildSelectView());
        pack();
        setMinimumSize(new Dimension(1000, 820));
        setLocationRelativeTo(null);
        refreshStatus();
    }

    private JPanel buildSelectView() {
        JPanel root = new JPanel(new BorderLayout(0, 0));
        root.setBackground(BG);

        JPanel content = new JPanel(new BorderLayout(8, 8));
        content.setBackground(BG);
        content.setBorder(new EmptyBorder(8, 12, 12, 12));
        content.add(buildWizard(),    BorderLayout.CENTER);
        content.add(buildActionBar(), BorderLayout.SOUTH);

        root.add(buildTopBar(), BorderLayout.NORTH);
        root.add(content,       BorderLayout.CENTER);
        return root;
    }

    private JPanel buildTopBar() {
        // ── Outer strip — full-width dark band ────────────────────────────
        JPanel bar = new JPanel(new BorderLayout(0, 0));
        bar.setBackground(PANEL);
        bar.setBorder(new CompoundBorder(
                BorderFactory.createMatteBorder(0, 0, 1, 0, BORDER),
                new EmptyBorder(0, 12, 0, 8)));

        // ── Left: icon + title + subtitle + options ───────────────────────
        JPanel left = new JPanel();
        left.setLayout(new BoxLayout(left, BoxLayout.X_AXIS));
        left.setBackground(PANEL);
        left.setOpaque(true);

        JLabel icon  = label("⬡", new Font(Font.SANS_SERIF, Font.BOLD, 18), ACCENT_H);
        JLabel title = label("  Pattern Matching", new Font(Font.SANS_SERIF, Font.BOLD, 13), WHITE);
        JLabel sep2  = label("  —  ", SMALL, DIM);
        JLabel sub   = label("Benchmark Launcher", SMALL, DIM);

        optClear     = check("Clear previous runs", false);
        optNegatives = check("Include negatives (Cat D)", false);
        optClear    .setBackground(PANEL);
        optNegatives.setBackground(PANEL);
        optClear    .addActionListener(e -> cfg.setClearPrevious(optClear.isSelected()));
        optNegatives.addActionListener(e -> cfg.setIncludeNegatives(optNegatives.isSelected()));

        // Separator pipe between subtitle and options
        JLabel pipe = label("  |  ", SMALL, BORDER);

        left.add(Box.createRigidArea(new Dimension(0, 36))); // enforce bar height
        left.add(icon);
        left.add(title);
        left.add(sep2);
        left.add(sub);
        left.add(pipe);
        left.add(optClear);
        left.add(Box.createRigidArea(new Dimension(12, 0)));
        left.add(optNegatives);

        // ── Centre: empty draggable spacer ────────────────────────────────
        JPanel centre = new JPanel();
        centre.setBackground(PANEL);
        centre.setOpaque(true);

        // ── Right: refresh + window controls ─────────────────────────────
        JPanel right = new JPanel(new FlowLayout(FlowLayout.RIGHT, 4, 0));
        right.setBackground(PANEL);
        right.setOpaque(true);

        JButton refreshBtn = titleBarBtn(TitleBarIcon.REFRESH, DIM, WHITE);
        refreshBtn.setToolTipText("Refresh status");
        refreshBtn.addActionListener(e -> refreshStatus());

        JButton minBtn   = titleBarBtn(TitleBarIcon.MINIMISE, DIM, WHITE);
        JButton maxBtn   = titleBarBtn(TitleBarIcon.MAXIMISE, DIM, WHITE);
        JButton closeBtn = titleBarBtn(TitleBarIcon.CLOSE,    DIM, RED);
        minBtn  .setToolTipText("Minimise");
        maxBtn  .setToolTipText("Maximise / restore");
        closeBtn.setToolTipText("Close");
        minBtn  .addActionListener(e -> setState(JFrame.ICONIFIED));
        maxBtn  .addActionListener(e -> {
            if (getExtendedState() == JFrame.MAXIMIZED_BOTH) setExtendedState(JFrame.NORMAL);
            else setExtendedState(JFrame.MAXIMIZED_BOTH);
        });
        closeBtn.addActionListener(e -> dispose());

        right.add(refreshBtn);
        right.add(Box.createRigidArea(new Dimension(8, 0)));
        right.add(minBtn); right.add(maxBtn); right.add(closeBtn);

        bar.add(left,   BorderLayout.WEST);
        bar.add(centre, BorderLayout.CENTER);
        bar.add(right,  BorderLayout.EAST);

        // ── Drag-to-move (bar, centre, and left all draggable) ────────────
        MouseAdapter drag = new MouseAdapter() {
            private Point origin;
            @Override public void mousePressed(MouseEvent e)  { origin = e.getPoint(); }
            @Override public void mouseDragged(MouseEvent e)  {
                if (origin == null) return;
                Point loc = getLocation();
                setLocation(loc.x + e.getX() - origin.x, loc.y + e.getY() - origin.y);
            }
        };
        bar.addMouseListener(drag);
        bar.addMouseMotionListener(drag);
        centre.addMouseListener(drag);
        centre.addMouseMotionListener(drag);

        return bar;
    }

    // ── Title-bar icon enum ───────────────────────────────────────────────

    private enum TitleBarIcon {
        REFRESH {
            @Override void paint(Graphics2D g, int w, int h, Color c) {
                g.setColor(c);
                g.setStroke(new BasicStroke(1.6f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
                int cx = w/2, cy = h/2, r = 5;
                g.drawArc(cx-r, cy-r, r*2, r*2, 30, 280);
                // arrowhead
                int ax = cx + r, ay = cy;
                int[] xs = {ax-3, ax+1, ax+1};
                int[] ys = {ay-1, ay-3, ay+2};
                g.fillPolygon(xs, ys, 3);
            }
        },
        MINIMISE {
            @Override void paint(Graphics2D g, int w, int h, Color c) {
                g.setColor(c);
                g.setStroke(new BasicStroke(1.8f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
                int y = h/2 + 3;
                g.drawLine(w/2 - 5, y, w/2 + 5, y);
            }
        },
        MAXIMISE {
            @Override void paint(Graphics2D g, int w, int h, Color c) {
                g.setColor(c);
                g.setStroke(new BasicStroke(1.6f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
                int x = w/2 - 5, y = h/2 - 5, s = 10;
                g.drawRect(x, y, s, s);
            }
        },
        CLOSE {
            @Override void paint(Graphics2D g, int w, int h, Color c) {
                g.setColor(c);
                g.setStroke(new BasicStroke(1.8f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
                int m = 5;
                g.drawLine(w/2 - m, h/2 - m, w/2 + m, h/2 + m);
                g.drawLine(w/2 + m, h/2 - m, w/2 - m, h/2 + m);
            }
        };
        abstract void paint(Graphics2D g, int w, int h, Color c);
    }

    /** A compact icon button for the custom title bar — drawn with Graphics2D, no font. */
    private static JButton titleBarBtn(TitleBarIcon icon, Color normalFg, Color hoverFg) {
        Color hoverBg = hoverFg == RED
                ? new Color(0x6e, 0x1a, 0x1a)
                : new Color(0x2a, 0x2f, 0x38);

        JButton b = new JButton() {
            private Color currentFg  = normalFg;
            private Color currentBg  = PANEL;
            { // instance initialiser — capture fields
                setOpaque(true);
                setContentAreaFilled(true);
                setBorderPainted(false);
                setFocusPainted(false);
                setBackground(PANEL);
                setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
                Dimension d = new Dimension(32, 30);
                setPreferredSize(d); setMinimumSize(d); setMaximumSize(d);
                addMouseListener(new MouseAdapter() {
                    @Override public void mouseEntered(MouseEvent e) { currentFg = hoverFg; currentBg = hoverBg; setBackground(hoverBg); repaint(); }
                    @Override public void mouseExited (MouseEvent e) { currentFg = normalFg; currentBg = PANEL;   setBackground(PANEL);   repaint(); }
                });
            }
            @Override protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                Graphics2D g2 = (Graphics2D) g.create();
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                icon.paint(g2, getWidth(), getHeight(), currentFg);
                g2.dispose();
            }
        };
        return b;
    }

    private JPanel buildWizard() {
        JPanel p = new JPanel(new GridBagLayout());
        p.setBackground(BG);
        GridBagConstraints g = new GridBagConstraints();
        g.fill    = GridBagConstraints.BOTH;
        g.insets  = new Insets(0, 4, 4, 4);

        // ── Row 0: Matchers | Variants | References ───────────────────────
        g.gridy = 0; g.weighty = 0.55;
        g.gridx = 0; g.weightx = 0.36; p.add(matchersPanel,   g);
        g.gridx = 1; g.weightx = 0.32; p.add(variantsPanel,   g);
        g.gridx = 2; g.weightx = 0.32; p.add(referencesPanel, g);

        // ── Row 1: Backgrounds | Scene Variants | Summary ↔ Progress ─────
        g.gridy = 1; g.weighty = 0.45;
        g.gridx = 0; g.weightx = 0.36; p.add(backgroundsPanel,  g);
        g.gridx = 1; g.weightx = 0.32; p.add(scenesPanel,        g);
        g.gridx = 2; g.weightx = 0.32; p.add(bottomCardPanel,    g);

        return p;
    }

    private JPanel buildActionBar() {
        JPanel p = new JPanel(new BorderLayout(8, 0));
        p.setBackground(BG);
        p.setBorder(new EmptyBorder(8, 0, 0, 0));

        JPanel left = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 4));
        left.setBackground(BG);
        JButton genRefBtn    = accentBtn("🖼  Generate References",     PANEL);
        JButton genBgBtn     = accentBtn("🏞  Generate Backgrounds",    PANEL);
        JButton genScenesBtn = accentBtn("🎬  Generate Variant Scenes", PANEL);
        genRefBtn   .addActionListener(e -> onGenerateReferences());
        genBgBtn    .addActionListener(e -> onGenerateBackgrounds());
        genScenesBtn.addActionListener(e -> onGenerateScenes());
        left.add(genRefBtn); left.add(genBgBtn); left.add(genScenesBtn);

        JPanel right = new JPanel(new FlowLayout(FlowLayout.RIGHT, 8, 4));
        right.setBackground(BG);
        JButton benchBtn = accentBtn("📊  Generate Benchmark Report", BTN_GREEN);
        JButton runBtn   = accentBtn("▶  Run Selected",               ACCENT);
        benchBtn.addActionListener(e -> onBenchmark());
        runBtn  .addActionListener(e -> onRun());
        right.add(benchBtn); right.add(runBtn);

        p.add(left,  BorderLayout.WEST);
        p.add(right, BorderLayout.EAST);
        return p;
    }

    // =========================================================================
    //  Cascade callbacks
    // =========================================================================

    private void onMatcherSelectionChanged() {
        MatcherDescriptor md = matchersPanel.getHighlightedMatcher();
        variantsPanel.populate(md);
        rebuildReferences(); // matcher change: rebuild immediately, no need to debounce
        refreshSummary();
    }

    private void onVariantSelectionChanged() {
        // cfg is already updated by VariantsPanel.onTableCheckChanged().
        // Summary is cheap — update immediately.
        refreshSummary();
        // References rebuild is expensive (filesystem scans) — debounce it.
        cascadeDebounce.restart();
    }

    /** Actually rebuilds the references panel (and cascades to backgrounds/scenes). */
    private void rebuildReferences() {
        MatcherDescriptor md     = matchersPanel.getHighlightedMatcher();
        List<String> selVariants = variantsPanel.getSelectedVariants(md);
        referencesPanel.rebuild(md, selVariants);
    }

    private void refreshSummary() {
        runSummaryPanel.refresh(cfg, matchersPanel.totalCount());
    }

    private void showProgress() {
        bottomCardLayout.show(bottomCardPanel, CARD_PROGRESS);
    }

    private void showSummary() {
        bottomCardLayout.show(bottomCardPanel, CARD_SUMMARY);
    }

    // =========================================================================
    //  Status refresh
    // =========================================================================

    private void refreshStatus() {
        ctx.reloadCatalogueFileNames();
        matchersPanel.refresh();
        onMatcherSelectionChanged();
    }

    // =========================================================================
    //  Generation actions
    // =========================================================================

    private void onGenerateReferences() {
        progressPanel.startSingleTask("Generating reference images…");
        showProgress();
        cancelled = false;
        runThread = new Thread(() -> {
            try {
                OpenCvLoader.load();
                Path outDir = Paths.get("test_output", "references");
                Files.createDirectories(outDir);
                ReferenceId[] ids = ReferenceId.values();
                SwingUtilities.invokeLater(() -> progressPanel.setGlobalTotal(ids.length));
                for (int i = 0; i < ids.length; i++) {
                    if (cancelled) break;
                    ReferenceId id = ids[i];
                    progressPanel.setPostStatus("Writing " + id.name() + "…");
                    org.opencv.core.Mat img = ReferenceImageFactory.build(id);
                    org.opencv.imgcodecs.Imgcodecs.imwrite(
                            outDir.resolve(id.name() + ".png").toString(), img);
                    img.release();
                    final int done = i + 1;
                    SwingUtilities.invokeLater(() -> progressPanel.setGlobalDone(done));
                }
                progressPanel.markFinished(cancelled,
                        cancelled ? "Cancelled." : "References written to test_output/references/");
                SwingUtilities.invokeLater(this::refreshStatus);
            } catch (Exception ex) { progressPanel.markError(ex.getMessage()); }
        }, "gen-references");
        runThread.setDaemon(true); runThread.start();
    }

    private void onGenerateBackgrounds() {
        progressPanel.startSingleTask("Generating background images…");
        showProgress();
        cancelled = false;
        runThread = new Thread(() -> {
            try {
                OpenCvLoader.load();
                Path outDir = Paths.get("test_output", "backgrounds");
                Files.createDirectories(outDir);
                BackgroundId[] ids = BackgroundId.values();
                SwingUtilities.invokeLater(() -> progressPanel.setGlobalTotal(ids.length));
                for (int i = 0; i < ids.length; i++) {
                    if (cancelled) break;
                    BackgroundId id = ids[i];
                    progressPanel.setPostStatus("Writing " + id.name() + "…");
                    org.opencv.core.Mat bg = BackgroundFactory.build(id, 640, 480);
                    org.opencv.imgcodecs.Imgcodecs.imwrite(
                            outDir.resolve(id.name() + ".png").toString(), bg);
                    bg.release();
                    final int done = i + 1;
                    SwingUtilities.invokeLater(() -> progressPanel.setGlobalDone(done));
                }
                progressPanel.markFinished(cancelled,
                        cancelled ? "Cancelled." : "Backgrounds written to test_output/backgrounds/");
                SwingUtilities.invokeLater(this::refreshStatus);
            } catch (Exception ex) { progressPanel.markError(ex.getMessage()); }
        }, "gen-backgrounds");
        runThread.setDaemon(true); runThread.start();
    }

    private void onGenerateScenes() {
        Path refDir = Paths.get("test_output", "references");
        Path bgDir  = Paths.get("test_output", "backgrounds");
        boolean refsOk = Files.isDirectory(refDir) &&
                WizardContext.countPngs(refDir) >= WizardContext.N_REFERENCES;
        boolean bgsOk  = Files.isDirectory(bgDir)  &&
                WizardContext.countPngs(bgDir)  >= WizardContext.N_BACKGROUNDS;
        if (!refsOk || !bgsOk) {
            StringBuilder msg = new StringBuilder("Cannot generate scenes — missing prerequisites:\n");
            if (!refsOk) msg.append("  • References not yet generated (use 🖼 Generate References)\n");
            if (!bgsOk)  msg.append("  • Backgrounds not yet generated (use 🏞 Generate Backgrounds)\n");
            warn(msg.toString().trim()); return;
        }
        progressPanel.startSingleTask("Generating variant scenes (catalogue)…");
        showProgress();
        cancelled = false;

        runThread = new Thread(() -> {
            try {
                OpenCvLoader.load();
                Path outDir = Paths.get("test_output", "catalogue_samples");
                Files.createDirectories(outDir);
                progressPanel.setPostStatus("Building catalogue…");
                List<SceneEntry> catalogue  = SceneCatalogue.build();
                List<SceneEntry> multiShape = SceneGenerator.buildMultiShape();
                List<SceneEntry> all        = new ArrayList<>(catalogue);
                all.addAll(multiShape);
                SwingUtilities.invokeLater(() -> progressPanel.setGlobalTotal(all.size()));
                for (int i = 0; i < all.size(); i++) {
                    if (cancelled) break;
                    SceneEntry scene    = all.get(i);
                    String     baseName = WizardContext.buildCatalogueName(scene);
                    org.opencv.imgcodecs.Imgcodecs.imwrite(
                            outDir.resolve(baseName + ".png").toString(), scene.sceneMat());
                    Files.writeString(outDir.resolve(baseName + ".json"),
                            SceneMetadata.toJson(baseName, scene));
                    final int done = i + 1;
                    if (done % 50 == 0 || done == all.size()) {
                        final String status = "Saved " + done + " / " + all.size() + " scenes…";
                        SwingUtilities.invokeLater(() -> {
                            progressPanel.setPostStatus(status);
                            progressPanel.setGlobalDone(done);
                        });
                    }
                }
                progressPanel.markFinished(cancelled, cancelled ? "Cancelled." :
                        "Scenes written to test_output/catalogue_samples/  (" + all.size() + " total)");
                SwingUtilities.invokeLater(this::refreshStatus);
            } catch (Exception ex) { progressPanel.markError(ex.getMessage()); }
        }, "gen-scenes");
        runThread.setDaemon(true); runThread.start();
    }

    // =========================================================================
    //  Run / Benchmark actions
    // =========================================================================

    private void onRun() {
        Set<MatcherDescriptor> checkedMatchers = cfg.getSelectedMatchers();
        if (checkedMatchers.isEmpty()) { warn("No matchers selected (tick at least one checkbox)."); return; }

        Map<MatcherDescriptor, Set<String>> matcherVariants = new LinkedHashMap<>();
        for (MatcherDescriptor md : checkedMatchers) {
            Set<String> variants = new LinkedHashSet<>(cfg.getSelectedVariants(md));
            if (!variants.isEmpty()) matcherVariants.put(md, variants);
        }
        if (matcherVariants.isEmpty()) { warn("No variants available for selected matchers."); return; }

        Set<ReferenceId>  refs      = cfg.getSelectedRefs();
        Set<SceneVariant> sceneVars = cfg.getSelectedScenes();
        if (refs.isEmpty())      { warn("No references selected."); return; }
        if (sceneVars.isEmpty() && !optNegatives.isSelected()) { warn("No scene variants selected."); return; }

        boolean clear   = optClear.isSelected();
        boolean inclNeg = optNegatives.isSelected();
        int     nThr    = Math.max(1, Math.min(16, Runtime.getRuntime().availableProcessors() - 1));

        List<MatcherDescriptor> runOrder = new ArrayList<>(matcherVariants.keySet());
        String[] names = runOrder.stream().map(MatcherDescriptor::displayName).toArray(String[]::new);
        // Total is unknown until the catalogue is loaded — set 0 to show "?" until then
        progressPanel.startRun(names, nThr, 0);
        // Register variant sub-rows for each matcher immediately
        for (int mi = 0; mi < runOrder.size(); mi++) {
            String[] variantArr = matcherVariants.get(runOrder.get(mi)).toArray(String[]::new);
            progressPanel.setMatcherVariants(mi, variantArr);
        }
        showProgress();
        cancelled = false;

        runThread = new Thread(() -> {
            try {
                OpenCvLoader.load();
                List<SceneEntry> cat = SceneCatalogueLoader.load(refs.toArray(ReferenceId[]::new));
                final boolean fi = inclNeg;
                cat = cat.stream().filter(sc -> {
                    if (sc.category() == SceneCategory.D_NEGATIVE) return fi;
                    if (!refs.contains(sc.primaryReferenceId()))    return false;
                    return sceneVars.stream().anyMatch(sv -> sv.matches(sc));
                }).toList();
                final List<SceneEntry> catalogue = cat;
                // Each (ref, scene) pair is one unit of work per matcher
                final int pairsPerMatcher = refs.size() * catalogue.size();
                // Global total = one unit per (matcher, ref, scene) combination
                final int realTotal = runOrder.size() * pairsPerMatcher;
                SwingUtilities.invokeLater(() -> {
                    progressPanel.setGlobalTotal(realTotal);
                    for (int i = 0; i < runOrder.size(); i++) {
                        // Matcher bar: refs × scenes
                        progressPanel.setMatcherTotal(i, pairsPerMatcher);
                        // Variant bar: same refs × scenes (each variant runs on every pair)
                        String[] variantArr = matcherVariants.get(runOrder.get(i)).toArray(String[]::new);
                        for (int vi = 0; vi < variantArr.length; vi++) {
                            progressPanel.setVariantTotal(i, vi, pairsPerMatcher);
                        }
                    }
                });

                for (int mi = 0; mi < runOrder.size(); mi++) {
                    if (cancelled) break;
                    MatcherDescriptor md  = runOrder.get(mi);
                    Set<String> variants  = matcherVariants.get(md);
                    // Build variant-name → index map for progress tracking
                    Map<String,Integer> variantIndex = new LinkedHashMap<>();
                    int vi = 0;
                    for (String vn : variants) variantIndex.put(vn, vi++);
                    progressPanel.setMatcherRunning(mi);
                    runOneMatcher(md, variants, variantIndex, refs, catalogue, clear, mi, nThr);
                }

                progressPanel.markFinished(cancelled, cancelled ? "Cancelled." : "All matchers complete.");
                SwingUtilities.invokeLater(this::refreshStatus);
            } catch (Exception ex) { progressPanel.markError(ex.getMessage()); }
        }, "benchmark-runner");
        runThread.setDaemon(true); runThread.start();
    }

    private void onBenchmark() {
        progressPanel.startBenchmark();
        showProgress();
        cancelled = false;
        runThread = new Thread(() -> {
            try {
                BenchmarkReportRunner.run(msg -> progressPanel.setPostStatus(msg));
                progressPanel.markFinished(false, "Done: test_output/benchmark/report.html");
                SwingUtilities.invokeLater(this::refreshStatus);
            } catch (Exception ex) { progressPanel.markError(ex.getMessage()); }
        }, "benchmark-report");
        runThread.setDaemon(true); runThread.start();
    }

    private void onCancelOrBack() {
        Thread t = runThread;
        if (t != null && t.isAlive()) {
            cancelled = true; t.interrupt();
            progressPanel.disableCancelButton();
        } else {
            showSummary();
            refreshStatus();
        }
    }

    // ── Core runner ───────────────────────────────────────────────────────

    private void runOneMatcher(MatcherDescriptor md, Set<String> activeVariants,
                                Map<String,Integer> variantIndex,
                                Set<ReferenceId> refs, List<SceneEntry> catalogue,
                                boolean clear, int midx, int nThr) {
        Path outDir = md.outputDir().toAbsolutePath().normalize();
        try {
            if (clear && Files.isDirectory(outDir)) clearDir(outDir);
            Files.createDirectories(outDir);
            if (catalogue.isEmpty()) { progressPanel.setMatcherSkipped(midx); return; }

            record RefEntry(ReferenceId id, org.opencv.core.Mat mat) {}
            List<RefEntry> refEntries = refs.stream()
                    .map(id -> new RefEntry(id, ReferenceImageFactory.build(id))).toList();
            record WorkItem(RefEntry ref, SceneEntry scene) {}
            List<WorkItem> work = new ArrayList<>(refs.size() * catalogue.size());
            for (RefEntry re : refEntries)
                for (SceneEntry sc : catalogue) work.add(new WorkItem(re, sc));

            ConcurrentLinkedQueue<AnalysisResult>        bag      = new ConcurrentLinkedQueue<>();
            ConcurrentHashMap<AnalysisResult,SceneEntry> sceneMap = new ConcurrentHashMap<>(work.size() * 2);
            final int fmidx = midx;

            // ── Incremental resume: load existing sidecar results ──────────
            Set<String> doneKeys = Collections.synchronizedSet(new HashSet<>());
            if (!clear) {
                Map<AnalysisResult, SceneEntry> prior = ResultMetadataStore.loadAll(outDir);
                // Group prior results by their (ref, scene, variant) work item key to avoid
                // counting each variant multiple times for the same work item.
                Set<String> countedPairs = Collections.synchronizedSet(new HashSet<>());
                prior.forEach((r, sc) -> {
                    bag.add(r);
                    if (sc != null) sceneMap.put(r, sc);
                    doneKeys.add(ResultMetadataStore.skipKey(r));
                    progressPanel.incrementResults();
                    // Variant done: count once per (variantName, ref, scene) pair
                    Integer vi = variantIndex.get(r.methodName());
                    String pairKey = r.methodName() + "|" + r.referenceId() + "|"
                            + (sc != null ? sc.variantLabel() + sc.backgroundId() : "?");
                    if (vi != null && countedPairs.add(pairKey)) {
                        progressPanel.incrementVariantDone(fmidx, vi);
                    }
                });
                // Matcher done: count unique (ref, scene) work items regardless of how many
                // variants they produced
                Set<String> countedWorkItems = new HashSet<>();
                prior.forEach((r, sc) -> {
                    String wk = String.valueOf(r.referenceId())
                            + "|" + (sc != null ? sc.variantLabel() + sc.backgroundId() : "?");
                    if (countedWorkItems.add(wk)) {
                        progressPanel.incrementDone(fmidx);
                    }
                });
            }

            ForkJoinPool pool = new ForkJoinPool(nThr);
            try {
                pool.submit(() -> work.parallelStream().forEach(item -> {
                    if (cancelled) return;
                    List<AnalysisResult> itemResults =
                        md.run(item.ref().id(), item.ref().mat(), item.scene(), activeVariants, outDir)
                          .stream().filter(r -> activeVariants.contains(r.methodName()))
                          .filter(r -> doneKeys.add(ResultMetadataStore.skipKey(r)))
                          .toList();
                    // Track which variants produced results this work item
                    Set<Integer> variantsHit = new HashSet<>();
                    for (AnalysisResult r : itemResults) {
                        bag.add(r);
                        sceneMap.put(r, item.scene());
                        ResultMetadataStore.write(r, item.scene(), outDir);
                        Integer vi = variantIndex.get(r.methodName());
                        if (vi != null) variantsHit.add(vi);
                        progressPanel.incrementResults();
                    }
                    // Increment each variant's done count once per work item (not once per result)
                    for (int vi : variantsHit) progressPanel.incrementVariantDone(fmidx, vi);
                    progressPanel.incrementDone(fmidx);
                })).get();
            } finally {
                pool.shutdown();
                refEntries.forEach(re -> re.mat().release());
            }

            if (cancelled) return;

            List<AnalysisResult>           results = new ArrayList<>(bag);
            Map<AnalysisResult,SceneEntry> sMap    = new LinkedHashMap<>(sceneMap);
            Map<AnalysisResult, DetectionVerdict> vd = new LinkedHashMap<>();
            for (AnalysisResult r : results) {
                SceneEntry sc = sMap.get(r);
                if (sc != null) vd.put(r, DetectionVerdict.evaluate(r, sc));
            }
            progressPanel.setPostStatus("Writing report: " + md.displayName() + "…");
            List<PerformanceProfile> profiles = PerformanceProfiler.profileAll(results);
            HtmlReportWriter.write(results, profiles, md.displayName(),
                    outDir.resolve("report.html"), vd, sMap);
            long   correct = vd.values().stream().filter(v -> v == DetectionVerdict.CORRECT).count();
            double acc     = vd.isEmpty() ? 0 : 100.0 * correct / vd.size();
            progressPanel.setMatcherDone(midx, acc);

        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt(); cancelled = true;
        } catch (Exception ex) {
            progressPanel.setMatcherError(midx);
        }
    }

    // ── Utilities ─────────────────────────────────────────────────────────

    private static void clearDir(Path dir) throws IOException {
        try (var walk = Files.walk(dir)) {
            walk.sorted(Comparator.reverseOrder())
                .filter(p -> !p.equals(dir))
                .forEach(p -> { try { Files.deleteIfExists(p); } catch (IOException ignored) {} });
        }
    }

    private void warn(String msg) {
        JOptionPane.showMessageDialog(this, msg, "Warning", JOptionPane.WARNING_MESSAGE);
    }

    // ── Entry point ───────────────────────────────────────────────────────

    public static void open() {
        if (GraphicsEnvironment.isHeadless()) {
            System.err.println("[BenchmarkLauncher] Headless — UI unavailable."); return;
        }
        SwingUtilities.invokeLater(() -> new BenchmarkLauncher().setVisible(true));
    }

    public static void main(String[] args) { open(); }
}

