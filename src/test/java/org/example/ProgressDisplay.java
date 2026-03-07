package org.example;

import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * In-place terminal progress display for the analytical test double-loop.
 *
 * <p>Renders a fixed-height block of lines using ANSI escape codes to move the
 * cursor back up and overwrite previous output.  Every call to {@link #update}
 * rewrites the exact same screen region, so the terminal never scrolls during
 * a run.
 *
 * <p>Output is written to <b>stderr</b> so that Maven Surefire's stdout
 * buffering never interferes with the in-place rewrite.
 *
 * <p>Layout (fixed line count = {@code 4 + refs.length}):
 * <pre>
 *  Line 0  [tag] N refs × M scenes | T threads
 *  Line 1  Progress  [████████████████░░░░░░░░░░░░░░░░]  47.3%
 *  Line 2  Pairs     1,234 / 2,610  |  elapsed 48s  |  ETA ~54s  |  885 results
 *  Line 3  ── per-reference ──────────────────────────────────────
 *  Line 4+ one line per ref:  REF_NAME  [████░░░░░░]  12/26  ✓ done / ◌ running
 * </pre>
 *
 * <p>Falls back to plain line-by-line output (every ~5%) when ANSI is not
 * detected.
 */
public final class ProgressDisplay {

    // -------------------------------------------------------------------------
    // ANSI codes
    // -------------------------------------------------------------------------
    private static final String RESET    = "\u001B[0m";
    private static final String BOLD     = "\u001B[1m";
    private static final String DIM      = "\u001B[2m";
    private static final String GREEN    = "\u001B[32m";
    private static final String YELLOW   = "\u001B[33m";
    private static final String CYAN     = "\u001B[36m";
    private static final String MAGENTA  = "\u001B[35m";
    private static final String WHITE    = "\u001B[97m";
    private static final String BG_BAR   = "\u001B[48;5;236m";
    private static final String FG_FILL  = "\u001B[38;5;75m";
    private static final String FG_EMPTY = "\u001B[38;5;238m";
    private static final String UP       = "\u001B[%dA";        // cursor up N lines (kept for reference)
    private static final String COL1     = "\u001B[1G";         // move to column 1
    private static final String ERASE    = "\u001B[2K";         // erase entire line

    private static final int BAR_WIDTH = 40;

    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------
    private final String        tag;
    private final ReferenceId[] refs;
    private final int           totalPairs;
    private final int           numThreads;
    private final long          tStart;
    private final boolean       ansiSupported;
    private final PrintStream   out;          // always stderr
    private final int           BLOCK_LINES;

    private final ConcurrentHashMap<ReferenceId, AtomicInteger> refDone = new ConcurrentHashMap<>();
    private final int scenesPerRef;

    private volatile boolean initialised  = false;
    private volatile String  statusLine   = "";   // current post-processing step

    // -------------------------------------------------------------------------
    // Constructor
    // -------------------------------------------------------------------------

    public ProgressDisplay(String tag, ReferenceId[] refs, int totalPairs,
                           int scenesPerRef, int numThreads, long tStart) {
        this.tag          = tag;
        this.refs         = refs;
        this.totalPairs   = totalPairs;
        this.scenesPerRef = scenesPerRef;
        this.numThreads   = numThreads;
        this.tStart       = tStart;
        this.BLOCK_LINES  = 4 + refs.length + 2;  // header+bar+numbers+sep+refs + post-sep+status
        this.out          = openRawStderr();

        for (ReferenceId r : refs) refDone.put(r, new AtomicInteger(0));

        this.ansiSupported = detectAnsi();
    }

    /**
     * Opens a {@link PrintStream} backed directly by {@link FileDescriptor#err},
     * bypassing any stream replacement done by Surefire or IntelliJ's test runner.
     * Falls back to {@link System#err} if the raw open fails.
     */
    private static PrintStream openRawStderr() {
        try {
            return new PrintStream(new FileOutputStream(FileDescriptor.err), true, "UTF-8");
        } catch (Exception e) {
            return System.err;
        }
    }

    /**
     * Detects whether the current environment supports ANSI escape codes.
     *
     * <p>Checks (in order):
     * <ul>
     *   <li>{@code WT_SESSION} env var — Windows Terminal</li>
     *   <li>{@code ANSICON} env var — ConEmu / Cmder ANSI shim</li>
     *   <li>{@code COLORTERM} env var — any colour-capable terminal</li>
     *   <li>{@code TERM} env var — non-dumb Unix terminal</li>
     *   <li>{@code idea.vendor.name}, {@code idea.active}, {@code java.class.path}
     *       containing "idea_rt" — IntelliJ's test runner</li>
     *   <li>{@code TERM_PROGRAM} — e.g. iTerm2, VSCode terminal</li>
     * </ul>
     */
    private static boolean detectAnsi() {
        // Windows Terminal
        if (System.getenv("WT_SESSION") != null) return true;
        // ANSI shims (ConEmu, ANSICON, Cmder)
        if (System.getenv("ANSICON") != null)    return true;
        // Any colour-capable terminal declares this
        if (System.getenv("COLORTERM") != null)  return true;
        // Generic TERM (Unix / Git-Bash / WSL)
        String term = System.getenv("TERM");
        if (term != null && !term.isEmpty() && !term.equals("dumb")) return true;
        // VS Code or other named terminal programs
        if (System.getenv("TERM_PROGRAM") != null) return true;
        // IntelliJ run/test runner (sets one or more of these)
        if (System.getProperty("idea.vendor.name") != null) return true;
        if (System.getProperty("idea.active")      != null) return true;
        String cp = System.getProperty("java.class.path", "");
        if (cp.contains("idea_rt")) return true;
        return false;
    }

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    private static final String CLEAR_SCREEN  = "\u001B[2J";   // erase entire display
    private static final String CURSOR_HOME   = "\u001B[H";    // move cursor to top-left

    /** Prints the initial blank block and renders the 0% state. */
    public void start() {
        if (!ansiSupported) {
            out.printf("%n[%s] Starting: %d refs \u00d7 %d scenes  |  threads: %d%n",
                    tag, refs.length, scenesPerRef, numThreads);
            out.flush();
            initialised = true;
            return;
        }
        // Clear the entire screen and move to top-left so the block owns the
        // terminal from a known position — no stale lines above us to collide with.
        out.print(CLEAR_SCREEN + CURSOR_HOME);
        out.flush();
        initialised = true;
        render(0, 0, 0);
    }

    /**
     * Called after each pair completes.
     *
     * @param refId       the reference that just completed a scene
     * @param totalDone   total pairs completed so far
     * @param resultCount total AnalysisResult objects collected so far
     */
    public void update(ReferenceId refId, int totalDone, int resultCount) {
        if (!initialised) return;
        refDone.computeIfAbsent(refId, __ -> new AtomicInteger(0)).incrementAndGet();
        if (ansiSupported) {
            render(totalDone, resultCount, System.currentTimeMillis() - tStart);
        } else {
            int reportEvery = Math.max(1, totalPairs / 20);
            if (totalDone % reportEvery == 0 || totalDone == totalPairs) {
                long   elapsed = System.currentTimeMillis() - tStart;
                double pct     = totalDone * 100.0 / totalPairs;
                long   eta     = elapsed > 0 ? (long)((elapsed / pct) * (100.0 - pct) / 1000) : 0;
                out.printf("[%s] %5.1f%%  %,d/%,d  elapsed %ds  ETA ~%ds  results %,d%n",
                        tag, pct, totalDone, totalPairs, elapsed / 1000, eta, resultCount);
                out.flush();
            }
        }
    }

    /** Renders 100% and prints a completion line below the block. */
    public void finish(int totalDone, int resultCount) {
        status("");   // clear status line at end
        if (!initialised) return;
        if (ansiSupported) {
            render(totalDone, resultCount, System.currentTimeMillis() - tStart);
            // Move cursor below the block so subsequent output doesn't overwrite it
            out.print(String.format("\u001B[%dB", BLOCK_LINES + 1));
        }
        double elapsed = (System.currentTimeMillis() - tStart) / 1000.0;
        out.printf("%n[%s] Complete: %,d results in %.1f s%n%n", tag, resultCount, elapsed);
        out.flush();
    }

    /**
     * Updates the post-processing status line at the bottom of the display.
     * Call this for each significant step during {@code @AfterAll} output writing.
     * In plain (non-ANSI) mode, prints a timestamped line instead.
     *
     * @param step short description, e.g. {@code "Computing verdicts..."}
     */
    public void status(String step) {
        if (!initialised) return;
        statusLine = step == null ? "" : step;
        if (ansiSupported) {
            // Re-use last known done/result counts — just redraw the status area
            renderStatusOnly();
        } else if (!step.isEmpty()) {
            out.printf("[%s] %s%n", tag, step);
            out.flush();
        }
    }

    // -------------------------------------------------------------------------
    // Rendering
    // -------------------------------------------------------------------------

    private synchronized void render(int done, int results, long elapsedMs) {
        double pct    = totalPairs > 0 ? done * 100.0 / totalPairs : 0;
        long   etaSec = (elapsedMs > 0 && pct > 0 && pct < 100)
                      ? (long)((elapsedMs / pct) * (100.0 - pct) / 1000) : 0;

        StringBuilder sb = new StringBuilder();

        // Always return to top-left of the cleared screen — no line counting needed
        sb.append(CURSOR_HOME);

        // ── Line 0: header ──────────────────────────────────────────────────
        sb.append(ERASE).append(COL1);
        sb.append(BOLD).append(CYAN).append("[").append(tag).append("]").append(RESET);
        sb.append("  ").append(WHITE).append(refs.length).append(RESET).append(" refs \u00d7 ");
        sb.append(WHITE).append(scenesPerRef).append(RESET).append(" scenes");
        sb.append(DIM).append("  |  ").append(numThreads).append(" threads").append(RESET);
        sb.append("\n").append(COL1);

        // ── Line 1: progress bar ─────────────────────────────────────────────
        int filled = Math.max(0, Math.min(BAR_WIDTH, (int) Math.round(pct / 100.0 * BAR_WIDTH)));
        sb.append(ERASE).append(COL1);
        sb.append(DIM).append("Progress  ").append(RESET);
        sb.append(BG_BAR).append(FG_FILL).append("\u2588".repeat(filled));
        sb.append(FG_EMPTY).append("\u2592".repeat(BAR_WIDTH - filled)).append(RESET);
        sb.append("  ");
        String barColour = pct >= 100 ? GREEN : pct >= 50 ? CYAN : YELLOW;
        sb.append(barColour).append(BOLD).append(padLeft(String.format("%.1f%%", pct), 6)).append(RESET);
        sb.append("\n").append(COL1);

        // ── Line 2: numbers ───────────────────────────────────────────────────
        sb.append(ERASE).append(COL1);
        sb.append(DIM).append("Pairs     ").append(RESET);
        sb.append(WHITE).append(String.format("%,d", done)).append(RESET);
        sb.append(DIM).append(" / ").append(RESET).append(String.format("%,d", totalPairs));
        sb.append(DIM).append("  |  elapsed ").append(RESET);
        sb.append(CYAN).append(elapsedMs / 1000).append("s").append(RESET);
        if (pct < 100 && pct > 0) {
            sb.append(DIM).append("  |  ETA ~").append(RESET);
            sb.append(YELLOW).append(etaSec).append("s").append(RESET);
        }
        sb.append(DIM).append("  |  results ").append(RESET);
        sb.append(MAGENTA).append(String.format("%,d", results)).append(RESET);
        sb.append("\n").append(COL1);

        // ── Line 3: separator ─────────────────────────────────────────────────
        sb.append(ERASE).append(COL1);
        sb.append(DIM).append("\u2500\u2500 per-reference ");
        sb.append("\u2500".repeat(BAR_WIDTH + 18)).append(RESET);
        sb.append("\n").append(COL1);

        // ── Lines 4+: one per ref ─────────────────────────────────────────────
        for (ReferenceId ref : refs) {
            int refCount = refDone.getOrDefault(ref, new AtomicInteger(0)).get();
            boolean complete = refCount >= scenesPerRef;
            int mini = Math.max(0, Math.min(12,
                    scenesPerRef > 0 ? refCount * 12 / scenesPerRef : 0));

            sb.append(ERASE).append(COL1);
            String refName = ref.name().length() > 28
                    ? ref.name().substring(0, 27) + "~" : ref.name();
            sb.append(complete ? GREEN : DIM).append(padRight(refName, 30)).append(RESET);
            sb.append("  ");
            sb.append(BG_BAR);
            sb.append(complete ? GREEN : FG_FILL).append("\u2588".repeat(mini));
            sb.append(FG_EMPTY).append("\u2592".repeat(12 - mini)).append(RESET);
            sb.append("  ");
            sb.append(padLeft(String.valueOf(refCount), 4));
            sb.append(DIM).append("/").append(RESET);
            sb.append(padRight(String.valueOf(scenesPerRef), 4));
            sb.append("  ");
            sb.append(complete ? GREEN + "\u2713 done   " : YELLOW + "\u25cc running");
            sb.append(RESET).append("\n").append(COL1);
        }

        // ── Post-processing section ───────────────────────────────────────────
        sb.append(ERASE).append(COL1);
        sb.append(DIM).append("\u2500\u2500 post-processing ");
        sb.append("\u2500".repeat(BAR_WIDTH + 15)).append(RESET);
        sb.append("\n").append(COL1);

        sb.append(ERASE).append(COL1);
        if (!statusLine.isEmpty()) {
            sb.append(CYAN).append("\u25b6 ").append(RESET);
            sb.append(WHITE).append(statusLine).append(RESET);
        }
        // no trailing newline — cursor stays on this line

        out.print(sb);
        out.flush();
    }

    /**
     * Redraws only the post-processing status line without touching the rest of
     * the block — used by {@link #status(String)} during {@code @AfterAll}.
     */
    private synchronized void renderStatusOnly() {
        // Row of the status line = BLOCK_LINES - 1 lines below the top (0-indexed)
        // Since we use CURSOR_HOME we navigate down to the exact row.
        int statusRow = BLOCK_LINES;   // 1-based row from top of cleared screen
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("\u001B[%d;1H", statusRow));   // move to row N, col 1
        sb.append(ERASE).append(COL1);
        if (!statusLine.isEmpty()) {
            sb.append(CYAN).append("\u25b6 ").append(RESET);
            sb.append(WHITE).append(statusLine).append(RESET);
        }
        out.print(sb);
        out.flush();
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private static String padLeft(String s, int width) {
        return s.length() >= width ? s : " ".repeat(width - s.length()) + s;
    }

    private static String padRight(String s, int width) {
        return s.length() >= width ? s : s + " ".repeat(width - s.length());
    }
}

