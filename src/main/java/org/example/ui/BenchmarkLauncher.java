package org.example.ui;

import org.example.*;

import javax.swing.*;
import javax.swing.border.*;
import javax.swing.table.*;
import java.awt.*;
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
 * The selection UI is a five-step vertical wizard:
 * <ol>
 *   <li><b>Matchers</b> — pick which techniques to run</li>
 *   <li><b>Variants</b> — pick which variants of the selected matchers</li>
 *   <li><b>References</b> — pick which reference shapes to use</li>
 *   <li><b>Backgrounds</b> — pick which background images</li>
 *   <li><b>Scene Variants</b> — pick which scene transforms</li>
 * </ol>
 * Each step is a {@link SelectionTable} that only becomes active once the
 * previous step has at least one row selected.  Row colours encode prior
 * generation status: green = 100%, orange = partial, grey = none.
 *
 * <h2>Views</h2>
 * <ul>
 *   <li>{@code SELECT} — the wizard + action buttons</li>
 *   <li>{@code PROGRESS} — replaces the whole window during a run; shows the
 *       detailed ProgressDisplay-style panel.  A Cancel / ← Back button
 *       returns to the SELECT view.</li>
 * </ul>
 */
public final class BenchmarkLauncher extends JFrame {

    // ── Card names ─────────────────────────────────────────────────────────
    private static final String CARD_SELECT   = "SELECT";
    private static final String CARD_PROGRESS = "PROGRESS";

    // ── Root layout ────────────────────────────────────────────────────────
    private final CardLayout  cardLayout = new CardLayout();
    private final JPanel      cardPanel  = new JPanel(cardLayout);

    // ── Options (top bar) ──────────────────────────────────────────────────
    private JCheckBox optClear;
    private JCheckBox optNegatives;

    // ── Wizard tables ──────────────────────────────────────────────────────
    /** Step 1 – Matchers */
    private SelectionTable tblMatchers;
    /** Step 2 – Variants (repopulated when matcher selection changes) */
    private SelectionTable tblVariants;
    /** Step 3 – References */
    private SelectionTable tblRefs;
    /** Step 4 – Backgrounds */
    private SelectionTable tblBgs;
    /** Step 5 – Scene Variants */
    private SelectionTable tblScenes;

    /** Ordered list matching tblMatchers rows. */
    private final List<MatcherDescriptor> matcherList = new ArrayList<>();

    // ── Progress panel ─────────────────────────────────────────────────────
    private ProgressPanel progressPanel;

    // ── Run state ──────────────────────────────────────────────────────────
    private volatile boolean    cancelled  = false;
    private volatile Thread     runThread  = null;

    // ── Catalogue files (cached at startup for fast status checks) ─────────
    private Set<String> catFileNames = Collections.emptySet();

    // =========================================================================
    //  Constructor
    // =========================================================================

    public BenchmarkLauncher() {
        super("Pattern Matching — Benchmark Launcher");
        setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        getContentPane().setBackground(BG);
        loadCatalogueFileNames();

        progressPanel = new ProgressPanel(this::onCancelOrBack);

        cardPanel.setBackground(BG);
        cardPanel.add(buildSelectView(),   CARD_SELECT);
        cardPanel.add(progressPanel,       CARD_PROGRESS);

        setContentPane(cardPanel);
        pack();
        setMinimumSize(new Dimension(1400, 860));
        setLocationRelativeTo(null);
        refreshStatus();
    }

    // =========================================================================
    //  SELECT view
    // =========================================================================

    private JPanel buildSelectView() {
        JPanel root = new JPanel(new BorderLayout(8,8));
        root.setBackground(BG); root.setBorder(new EmptyBorder(12,12,12,12));
        root.add(buildTopBar(),    BorderLayout.NORTH);
        root.add(buildWizard(),    BorderLayout.CENTER);
        root.add(buildActionBar(), BorderLayout.SOUTH);
        return root;
    }

    // ── Top bar ───────────────────────────────────────────────────────────

    private JPanel buildTopBar() {
        JPanel p = new JPanel(new BorderLayout()); p.setBackground(BG);
        p.setBorder(new EmptyBorder(0,0,8,0));
        p.add(titleLabel("Pattern Matching — Benchmark Launcher"), BorderLayout.WEST);

        JPanel opts = new JPanel(new FlowLayout(FlowLayout.RIGHT,12,0)); opts.setBackground(BG);
        optClear     = check("Clear previous output before run", false);
        optNegatives = check("Include Category D (negative) scenes", false);
        JButton refreshBtn = smallBtn("↺ Refresh Status");
        refreshBtn.addActionListener(e -> refreshStatus());
        opts.add(optClear); opts.add(optNegatives); opts.add(refreshBtn);
        p.add(opts, BorderLayout.EAST); return p;
    }

    // ── 5-column wizard ───────────────────────────────────────────────────

    private JPanel buildWizard() {
        JPanel p = new JPanel(new GridBagLayout()); p.setBackground(BG);
        GridBagConstraints g = new GridBagConstraints();
        g.fill = GridBagConstraints.BOTH; g.insets = new Insets(0,4,0,4); g.weighty = 1.0;

        g.gridx=0; g.weightx=0.22; p.add(buildMatchersStep(),  g);
        g.gridx=1; g.weightx=0.22; p.add(buildVariantsStep(),  g);
        g.gridx=2; g.weightx=0.22; p.add(buildRefsStep(),      g);
        g.gridx=3; g.weightx=0.17; p.add(buildBgsStep(),       g);
        g.gridx=4; g.weightx=0.17; p.add(buildScenesStep(),    g);
        return p;
    }

    // ── Step 1: Matchers ──────────────────────────────────────────────────

    private JPanel buildMatchersStep() {
        JPanel outer = titledPanel("1 — Matchers");

        tblMatchers = new SelectionTable();
        matcherList.clear();
        matcherList.addAll(MatcherRegistry.ALL);

        tblMatchers.setRows(buildMatcherRows(), false);

        // Single-row highlight selection: one click → populate Variants.
        // The checkbox column is hidden; selection is driven by row highlight.
        tblMatchers.table().setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        TableColumnModel tcm = tblMatchers.table().getColumnModel();
        tcm.getColumn(0).setMaxWidth(0); tcm.getColumn(0).setMinWidth(0); tcm.getColumn(0).setPreferredWidth(0);

        tblMatchers.table().getSelectionModel().addListSelectionListener(
                e -> { if (!e.getValueIsAdjusting()) onMatcherSelectionChanged(); });

        outer.add(vgap(4));
        outer.add(scrollPane(tblMatchers)); return outer;
    }

    private List<SelectionTable.RowData> buildMatcherRows() {
        List<SelectionTable.RowData> rows = new ArrayList<>();
        for (MatcherDescriptor md : matcherList) {
            int total = md.variantNames().size();
            boolean hasReport = Files.exists(md.outputDir().resolve("report.html"));
            int generated = hasReport ? total : 0;
            rows.add(new SelectionTable.RowData(
                    md.displayName() + "  [" + md.tag() + "]",
                    total + " variants",
                    generated, total));
        }
        return rows;
    }

    // ── Step 2: Variants ──────────────────────────────────────────────────

    private JPanel buildVariantsStep() {
        JPanel outer = titledPanel("2 — Variants");

        tblVariants = new SelectionTable();
        // Initially disabled — populated when a matcher row is highlighted
        tblVariants.setEnabled(false);

        // Cascade: checking/unchecking a variant rebuilds the References table
        tblVariants.table().getModel().addTableModelListener(
                e -> { if (e.getColumn() == 0) onVariantSelectionChanged(); });

        JCheckBox selAll = check("Select all", false);
        selAll.addActionListener(e -> tblVariants.selectAll(selAll.isSelected()));

        outer.add(selAll); outer.add(vgap(4));
        outer.add(scrollPane(tblVariants)); return outer;
    }

    private void onMatcherSelectionChanged() {
        int sel = tblMatchers.table().getSelectedRow();
        List<SelectionTable.RowData> rows = new ArrayList<>();
        if (sel >= 0 && sel < matcherList.size()) {
            MatcherDescriptor md = matcherList.get(sel);
            for (String vn : md.variantNames()) {
                int gen   = countGeneratedVariantFiles(md, vn);
                int total = expectedVariantTotal();
                rows.add(new SelectionTable.RowData(vn, "[" + md.tag() + "]", gen, total));
            }
        }
        tblVariants.setRows(rows, false);
        tblVariants.setEnabled(!rows.isEmpty());
        // cascade
        onVariantSelectionChanged();
    }

    private void onVariantSelectionChanged() {
        rebuildRefsTable();
    }

    // ── Step 3: References ────────────────────────────────────────────────

    private JPanel buildRefsStep() {
        JPanel outer = titledPanel("3 — References");
        tblRefs = new SelectionTable();
        tblRefs.table().getModel().addTableModelListener(
                e -> { if (e.getColumn()==0) onRefSelectionChanged(); });

        JCheckBox selAll = check("Select all", false);
        selAll.addActionListener(e -> tblRefs.selectAll(selAll.isSelected()));
        outer.add(selAll); outer.add(vgap(4));
        outer.add(scrollPane(tblRefs)); return outer;
    }

    private void rebuildRefsTable() {
        // Generated = catalogue PNGs already on disk for that ReferenceId
        //             = how many (Background × SceneVariant) combos exist.
        // Total     = Backgrounds × SceneVariants (fixed from enum sizes).
        List<SelectionTable.RowData> rows = new ArrayList<>();
        for (ReferenceId rid : ReferenceId.values()) {
            int gen   = countGeneratedForRef(rid);
            int total = expectedRefTotal();
            rows.add(new SelectionTable.RowData(rid.name(), groupOf(rid), gen, total));
        }
        tblRefs.setRows(rows, false);
        onRefSelectionChanged();
    }

    private void onRefSelectionChanged() {
        rebuildBgsTable();
    }

    // ── Step 4: Backgrounds ───────────────────────────────────────────────

    private JPanel buildBgsStep() {
        JPanel outer = titledPanel("4 — Backgrounds");
        tblBgs = new SelectionTable();
        tblBgs.table().getModel().addTableModelListener(
                e -> { if (e.getColumn()==0) onBgSelectionChanged(); });

        JCheckBox selAll = check("Select all", false);
        selAll.addActionListener(e -> tblBgs.selectAll(selAll.isSelected()));
        outer.add(selAll); outer.add(vgap(4));
        outer.add(scrollPane(tblBgs)); return outer;
    }

    private void rebuildBgsTable() {
        List<SelectionTable.RowData> rows = new ArrayList<>();
        for (BackgroundId bg : BackgroundId.values()) {
            // Generated = how many scene variants have been produced for this background
            // Catalogue filenames: {cat}__{variant}__{ref}__{bg}.png
            // The bg segment is the last __ segment before .png
            String bgSeg = "__" + bg.name().toLowerCase();
            int gen = catFileNames.isEmpty() ? 0 : (int) catFileNames.stream()
                    .filter(n -> n.endsWith(".png") && n.contains(bgSeg))
                    .count();
            // Total = 1 background × N_SCENE_VARIANTS
            rows.add(new SelectionTable.RowData(bg.name(), bgGroupOf(bg), gen, N_SCENE_VARIANTS));
        }
        tblBgs.setRows(rows, false);
        onBgSelectionChanged();
    }

    private void onBgSelectionChanged() {
        rebuildScenesTable();
    }

    // ── Step 5: Scene Variants ────────────────────────────────────────────

    private JPanel buildScenesStep() {
        JPanel outer = titledPanel("5 — Scene Variants");
        tblScenes = new SelectionTable();

        JCheckBox selAll = check("Select all", false);
        selAll.addActionListener(e -> tblScenes.selectAll(selAll.isSelected()));
        outer.add(selAll); outer.add(vgap(4));
        outer.add(scrollPane(tblScenes)); return outer;
    }

    private void rebuildScenesTable() {
        List<SceneVariant> allSv = Arrays.stream(SceneVariant.values())
                .filter(sv -> sv.category() != SceneCategory.D_NEGATIVE).toList();
        List<SelectionTable.RowData> rows = new ArrayList<>();
        for (SceneVariant sv : allSv) {
            String key = sv.label().replace(".","-").replace(" ","-").toLowerCase();
            boolean done = !catFileNames.isEmpty() && catFileNames.stream().anyMatch(n -> n.contains(key));
            rows.add(new SelectionTable.RowData(sv.label(), sv.category().name(), done?1:0, 1));
        }
        tblScenes.setRows(rows, false);
    }

    // ── Action bar ────────────────────────────────────────────────────────

    private JPanel buildActionBar() {
        JPanel p = new JPanel(new BorderLayout(8, 0));
        p.setBackground(BG); p.setBorder(new EmptyBorder(8,0,0,0));

        // ── Left: data-generation buttons ─────────────────────────────────
        JPanel left = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 4));
        left.setBackground(BG);

        JButton genRefBtn = accentBtn("🖼  Generate References", PANEL);
        genRefBtn.addActionListener(e -> onGenerateReferences());

        JButton genBgBtn = accentBtn("🏞  Generate Backgrounds", PANEL);
        genBgBtn.addActionListener(e -> onGenerateBackgrounds());

        JButton genScenesBtn = accentBtn("🎬  Generate Variant Scenes", PANEL);
        genScenesBtn.addActionListener(e -> onGenerateScenes());

        left.add(genRefBtn); left.add(genBgBtn); left.add(genScenesBtn);

        // ── Right: run buttons ─────────────────────────────────────────────
        JPanel right = new JPanel(new FlowLayout(FlowLayout.RIGHT, 8, 4));
        right.setBackground(BG);
        JButton benchBtn = accentBtn("📊  Generate Benchmark Report", BTN_GREEN);
        benchBtn.addActionListener(e -> onBenchmark());
        JButton runBtn = accentBtn("▶  Run Selected", ACCENT);
        runBtn.addActionListener(e -> onRun());
        right.add(benchBtn); right.add(runBtn);

        p.add(left,  BorderLayout.WEST);
        p.add(right, BorderLayout.EAST);
        return p;
    }

    // ── Generation actions ────────────────────────────────────────────────

    private void onGenerateReferences() {
        progressPanel.startSingleTask("Generating reference images…");
        cardLayout.show(cardPanel, CARD_PROGRESS);
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
            } catch (Exception ex) {
                progressPanel.markError(ex.getMessage());
            }
        }, "gen-references");
        runThread.setDaemon(true);
        runThread.start();
    }

    private void onGenerateBackgrounds() {
        progressPanel.startSingleTask("Generating background images…");
        cardLayout.show(cardPanel, CARD_PROGRESS);
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
            } catch (Exception ex) {
                progressPanel.markError(ex.getMessage());
            }
        }, "gen-backgrounds");
        runThread.setDaemon(true);
        runThread.start();
    }

    private void onGenerateScenes() {
        // Pre-condition: references and backgrounds must already be generated
        Path refDir = Paths.get("test_output", "references");
        Path bgDir  = Paths.get("test_output", "backgrounds");
        boolean refsOk = Files.isDirectory(refDir) &&
                countPngs(refDir) >= ReferenceId.values().length;
        boolean bgsOk  = Files.isDirectory(bgDir)  &&
                countPngs(bgDir)  >= BackgroundId.values().length;

        if (!refsOk || !bgsOk) {
            StringBuilder msg = new StringBuilder("Cannot generate scenes — missing prerequisites:\n");
            if (!refsOk) msg.append("  • Reference images not yet generated (use 🖼 Generate References)\n");
            if (!bgsOk)  msg.append("  • Background images not yet generated (use 🏞 Generate Backgrounds)\n");
            warn(msg.toString().trim());
            return;
        }

        progressPanel.startSingleTask("Generating variant scenes (catalogue)…");
        cardLayout.show(cardPanel, CARD_PROGRESS);
        cancelled = false;

        runThread = new Thread(() -> {
            try {
                OpenCvLoader.load();
                Path outDir = Paths.get("test_output", "catalogue_samples");
                Files.createDirectories(outDir);

                progressPanel.setPostStatus("Building catalogue…");
                java.util.List<SceneEntry> catalogue = SceneCatalogue.build();
                java.util.List<SceneEntry> multiShape = SceneGenerator.buildMultiShape();
                java.util.List<SceneEntry> all = new java.util.ArrayList<>(catalogue);
                all.addAll(multiShape);

                SwingUtilities.invokeLater(() -> progressPanel.setGlobalTotal(all.size()));

                for (int i = 0; i < all.size(); i++) {
                    if (cancelled) break;
                    SceneEntry scene = all.get(i);
                    String baseName = buildCatalogueName(scene);

                    // Save clean PNG
                    org.opencv.imgcodecs.Imgcodecs.imwrite(
                            outDir.resolve(baseName + ".png").toString(), scene.sceneMat());

                    // Save JSON sidecar
                    String json = SceneMetadata.toJson(baseName, scene);
                    Files.write(outDir.resolve(baseName + ".json"),
                            json.getBytes(java.nio.charset.StandardCharsets.UTF_8));

                    final int done = i + 1;
                    if (done % 50 == 0 || done == all.size()) {
                        final String status = "Saved " + done + " / " + all.size() + " scenes…";
                        SwingUtilities.invokeLater(() -> {
                            progressPanel.setPostStatus(status);
                            progressPanel.setGlobalDone(done);
                        });
                    }
                }

                progressPanel.markFinished(cancelled,
                        cancelled ? "Cancelled." :
                        "Scenes written to test_output/catalogue_samples/  (" + all.size() + " total)");
                SwingUtilities.invokeLater(this::refreshStatus);
            } catch (Exception ex) {
                progressPanel.markError(ex.getMessage());
            }
        }, "gen-scenes");
        runThread.setDaemon(true);
        runThread.start();
    }

    /** Mirrors the filename logic in SceneCatalogueTest so names are consistent. */
    private static String buildCatalogueName(SceneEntry scene) {
        String cat     = scene.category().name().toLowerCase();
        String ref     = scene.primaryReferenceId() != null
                ? scene.primaryReferenceId().name().toLowerCase() : "negative";
        String variant = scene.variantLabel().toLowerCase().replaceAll("[^a-z0-9_]", "_");
        String bg      = scene.backgroundId().name().toLowerCase();
        return cat + "__" + variant + "__" + ref + "__" + bg;
    }

    /** Counts PNG files in a directory (non-recursive). */
    private static int countPngs(Path dir) {
        try (var s = Files.list(dir)) {
            return (int) s.filter(p -> p.toString().endsWith(".png")).count();
        } catch (IOException e) { return 0; }
    }

    // =========================================================================
    //  Actions
    // =========================================================================

    private void onRun() {
        // ── Collect selections ────────────────────────────────────────────
        int matcherRow = tblMatchers.table().getSelectedRow();
        if (matcherRow < 0 || matcherRow >= matcherList.size()) {
            warn("No matcher selected."); return;
        }
        MatcherDescriptor selectedMd = matcherList.get(matcherRow);

        // Collect checked variants for the selected matcher
        List<String> allVariantNames = selectedMd.variantNames();
        Set<String> activeVariants = new LinkedHashSet<>();
        for (int idx : tblVariants.selectedIndices()) {
            if (idx < allVariantNames.size()) activeVariants.add(allVariantNames.get(idx));
        }
        if (activeVariants.isEmpty()) { warn("No variants selected."); return; }

        Map<MatcherDescriptor, Set<String>> active = new LinkedHashMap<>();
        active.put(selectedMd, activeVariants);

        List<ReferenceId> allRids = List.of(ReferenceId.values());
        Set<ReferenceId>  refs    = new LinkedHashSet<>();
        for (int i : tblRefs.selectedIndices()) refs.add(allRids.get(i));
        if (refs.isEmpty()) { warn("No references selected."); return; }

        List<SceneVariant> allSv = Arrays.stream(SceneVariant.values())
                .filter(sv -> sv.category()!=SceneCategory.D_NEGATIVE).toList();
        Set<SceneVariant> sceneVars = new LinkedHashSet<>();
        for (int i : tblScenes.selectedIndices())
            if (i < allSv.size()) sceneVars.add(allSv.get(i));
        if (sceneVars.isEmpty() && !optNegatives.isSelected()) {
            warn("No scene variants selected."); return;
        }

        boolean clear   = optClear.isSelected();
        boolean inclNeg = optNegatives.isSelected();
        int     nThr    = Math.max(1, Math.min(16, Runtime.getRuntime().availableProcessors()-1));
        List<MatcherDescriptor> runOrder = new ArrayList<>(active.keySet());

        String[] names = runOrder.stream().map(MatcherDescriptor::displayName).toArray(String[]::new);
        progressPanel.startRun(names, nThr, runOrder.size()*refs.size()*10);
        cardLayout.show(cardPanel, CARD_PROGRESS);
        cancelled = false;

        runThread = new Thread(() -> {
            try {
                OpenCvLoader.load();
                List<SceneEntry> cat = SceneCatalogueLoader.load(refs.toArray(ReferenceId[]::new));
                final boolean fi = inclNeg;
                cat = cat.stream().filter(sc -> {
                    if (sc.category()==SceneCategory.D_NEGATIVE) return fi;
                    if (!refs.contains(sc.primaryReferenceId())) return false;
                    return sceneVars.stream().anyMatch(sv -> sv.matches(sc));
                }).toList();
                final List<SceneEntry> catalogue = cat;
                final int realTotal = runOrder.size()*catalogue.size();
                SwingUtilities.invokeLater(() -> {
                    progressPanel.setGlobalTotal(realTotal);
                    for (int i=0;i<runOrder.size();i++) progressPanel.setMatcherTotal(i, catalogue.size());
                });

                for (int mi = 0; mi < runOrder.size(); mi++) {
                    if (cancelled) break;
                    MatcherDescriptor md = runOrder.get(mi);
                    progressPanel.setMatcherRunning(mi);
                    runOneMatcher(md, active.get(md), refs, catalogue, clear, mi, nThr);
                }

                final boolean wasCancelled = cancelled;
                progressPanel.markFinished(wasCancelled, "All matchers complete.");
                SwingUtilities.invokeLater(this::refreshStatus);
            } catch (Exception ex) {
                progressPanel.markError(ex.getMessage());
            }
        }, "benchmark-runner");
        runThread.setDaemon(true);
        runThread.start();
    }

    private void onBenchmark() {
        progressPanel.startBenchmark();
        cardLayout.show(cardPanel, CARD_PROGRESS);
        cancelled = false;

        runThread = new Thread(() -> {
            try {
                BenchmarkReportRunner.run(msg -> progressPanel.setPostStatus(msg));
                progressPanel.markFinished(false, "Done: test_output/benchmark/report.html");
                SwingUtilities.invokeLater(this::refreshStatus);
            } catch (Exception ex) {
                progressPanel.markError(ex.getMessage());
            }
        }, "benchmark-report");
        runThread.setDaemon(true);
        runThread.start();
    }

    private void onCancelOrBack() {
        Thread t = runThread;
        if (t != null && t.isAlive()) {
            cancelled = true; t.interrupt();
            progressPanel.disableCancelButton();
        } else {
            cardLayout.show(cardPanel, CARD_SELECT);
            refreshStatus();
        }
    }

    // ── Core runner ───────────────────────────────────────────────────────

    private void runOneMatcher(MatcherDescriptor md, Set<String> activeVariants,
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
            List<WorkItem> work = new ArrayList<>(refs.size()*catalogue.size());
            for (RefEntry re : refEntries) for (SceneEntry sc : catalogue) work.add(new WorkItem(re,sc));

            ConcurrentLinkedQueue<AnalysisResult>        bag      = new ConcurrentLinkedQueue<>();
            ConcurrentHashMap<AnalysisResult,SceneEntry> sceneMap = new ConcurrentHashMap<>(work.size()*2);
            final int fmidx = midx;

            ForkJoinPool pool = new ForkJoinPool(nThr);
            try {
                pool.submit(() -> work.parallelStream().forEach(item -> {
                    if (cancelled) return;
                    md.run(item.ref().id(), item.ref().mat(), item.scene(), activeVariants, outDir)
                      .stream().filter(r -> activeVariants.contains(r.methodName()))
                      .forEach(r -> { bag.add(r); sceneMap.put(r, item.scene()); });
                    progressPanel.incrementDone(fmidx);
                    progressPanel.incrementResults();
                })).get();
            } finally { pool.shutdown(); refEntries.forEach(re -> re.mat().release()); }

            if (cancelled) return;

            List<AnalysisResult>            results = new ArrayList<>(bag);
            Map<AnalysisResult,SceneEntry>  sMap    = new LinkedHashMap<>(sceneMap);
            Map<AnalysisResult,DetectionVerdict> vd = new LinkedHashMap<>();
            for (AnalysisResult r : results) {
                SceneEntry sc = sMap.get(r); if (sc!=null) vd.put(r, DetectionVerdict.evaluate(r,sc));
            }
            progressPanel.setPostStatus("Writing report: "+md.displayName()+"…");
            List<PerformanceProfile> profiles = PerformanceProfiler.profileAll(results);
            HtmlReportWriter.write(results, profiles, md.displayName(),
                    outDir.resolve("report.html"), vd, sMap);
            long   correct = vd.values().stream().filter(v -> v==DetectionVerdict.CORRECT).count();
            double acc     = vd.isEmpty()?0:100.0*correct/vd.size();
            progressPanel.setMatcherDone(midx, acc);

        } catch (InterruptedException ie) { Thread.currentThread().interrupt(); cancelled=true;
        } catch (Exception ex) { progressPanel.setMatcherError(midx); }
    }

    // =========================================================================
    //  Status refresh
    // =========================================================================

    private void refreshStatus() {
        loadCatalogueFileNames();
        // Preserve current row selection across the rebuild
        int prevSel = tblMatchers.table().getSelectedRow();
        tblMatchers.setRows(buildMatcherRows(), false);
        if (prevSel >= 0 && prevSel < tblMatchers.rowCount()) {
            tblMatchers.table().setRowSelectionInterval(prevSel, prevSel);
        }
        // Cascade rebuild
        onMatcherSelectionChanged();
    }

    private void loadCatalogueFileNames() {
        Set<String> files = new HashSet<>();
        try {
            Path catDir = Paths.get("test_output","catalogue_samples");
            if (Files.isDirectory(catDir)) {
                try (var s = Files.list(catDir)) {
                    s.map(p -> p.getFileName().toString().toLowerCase()).forEach(files::add);
                }
            }
        } catch (IOException ignored) {}
        catFileNames = files;
    }

    // ── Generation-count helpers ──────────────────────────────────────────

    /** Total non-negative SceneVariant count — computed once. */
    private static final int N_SCENE_VARIANTS = (int) Arrays.stream(SceneVariant.values())
            .filter(sv -> sv.category() != SceneCategory.D_NEGATIVE).count();

    /** Total BackgroundId count. */
    private static final int N_BACKGROUNDS = BackgroundId.values().length;

    /** Total ReferenceId count. */
    private static final int N_REFERENCES = ReferenceId.values().length;

    /**
     * Expected total for one variant = References × Backgrounds × SceneVariants.
     */
    private static int expectedVariantTotal() {
        return N_REFERENCES * N_BACKGROUNDS * N_SCENE_VARIANTS;
    }

    /**
     * Expected total for one reference = Backgrounds × SceneVariants.
     */
    private static int expectedRefTotal() {
        return N_BACKGROUNDS * N_SCENE_VARIANTS;
    }

    /**
     * Counts generated PNGs for a specific variant by scanning its annotated
     * subdirectory: {@code outputDir/annotated/{variantName}/}.
     */
    private static int countGeneratedVariantFiles(MatcherDescriptor md, String variant) {
        Path varDir = md.outputDir().resolve("annotated").resolve(variant);
        if (!Files.isDirectory(varDir)) return 0;
        try (var s = Files.list(varDir)) {
            return (int) s.filter(p -> p.toString().endsWith(".png")).count();
        } catch (IOException e) { return 0; }
    }

    /**
     * Counts how many catalogue PNGs have already been generated for a given
     * reference — i.e. how many (Background × SceneVariant) combos are on disk.
     * Catalogue filenames: {@code {cat}__{variant}__{ref}__{bg}.png}
     */
    private int countGeneratedForRef(ReferenceId rid) {
        if (catFileNames.isEmpty()) return 0;
        String refSeg = "__" + rid.name().toLowerCase() + "__";
        return (int) catFileNames.stream()
                .filter(n -> n.endsWith(".png") && n.contains(refSeg))
                .count();
    }

    // =========================================================================
    //  Domain helpers
    // =========================================================================

    private static String groupOf(ReferenceId rid) {
        String n = rid.name();
        if (n.startsWith("LINE_"))        return "Lines";
        if (n.startsWith("CIRCLE_")||n.startsWith("ELLIPSE_")) return "Circles & Ellipses";
        if (n.startsWith("RECT_ROTATED")) return "Rotated Rectangles";
        if (n.startsWith("RECT_"))        return "Rectangles";
        if (n.startsWith("TRIANGLE_")||n.startsWith("PENTAGON_")||n.startsWith("HEXAGON_")||
            n.startsWith("HEPTAGON_") ||n.startsWith("OCTAGON_") ||n.startsWith("STAR_"))
                                           return "Polygons & Stars";
        if (n.startsWith("POLYLINE_"))    return "Polylines";
        if (n.startsWith("ARC_"))         return "Arcs";
        if (n.startsWith("CONCAVE_")||n.startsWith("IRREGULAR_")) return "Concave / Irregular";
        if (n.startsWith("COMPOUND_"))    return "Compound Shapes";
        if (n.startsWith("GRID_")||n.startsWith("CHECKER")||n.equals("CROSSHAIR"))
                                           return "Grids & Patterns";
        if (n.startsWith("TEXT_"))         return "Text";
        if (n.startsWith("BICOLOUR_")||n.startsWith("TRICOLOUR_")) return "Multi-Colour";
        return "Other";
    }

    private static String bgGroupOf(BackgroundId bg) {
        String n = bg.name();
        if (n.startsWith("BG_SOLID"))    return "Solid";
        if (n.startsWith("BG_GRADIENT")) return "Gradient";
        if (n.startsWith("BG_NOISE"))    return "Noise";
        if (n.startsWith("BG_GRID"))     return "Grid";
        if (n.startsWith("BG_RANDOM"))   return "Random";
        return "Other";
    }

    private static void clearDir(Path dir) throws IOException {
        try (var walk = Files.walk(dir)) {
            walk.sorted(Comparator.reverseOrder()).filter(p -> !p.equals(dir))
                .forEach(p -> { try { Files.deleteIfExists(p); } catch (IOException ignored) {} });
        }
    }

    private void warn(String msg) {
        JOptionPane.showMessageDialog(this, msg, "Warning", JOptionPane.WARNING_MESSAGE);
    }

    // =========================================================================
    //  Entry point
    // =========================================================================

    public static void open() {
        if (GraphicsEnvironment.isHeadless()) {
            System.err.println("[BenchmarkLauncher] Headless — UI unavailable."); return;
        }
        SwingUtilities.invokeLater(() -> new BenchmarkLauncher().setVisible(true));
    }

    public static void main(String[] args) { open(); }
}

