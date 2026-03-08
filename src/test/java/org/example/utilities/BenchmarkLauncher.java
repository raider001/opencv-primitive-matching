package org.example.utilities;
/**
 * Backwards-compatibility shim — the full implementation has moved to
 * {@link org.example.ui.BenchmarkLauncher}.
 *
 * @deprecated Use {@code org.example.ui.BenchmarkLauncher} directly.
 */
@Deprecated
public final class BenchmarkLauncher {
    private BenchmarkLauncher() {}
    /** Opens the benchmark-launcher window. */
    public static void open()              { org.example.ui.BenchmarkLauncher.open(); }
    public static void main(String[] args) { open(); }
}