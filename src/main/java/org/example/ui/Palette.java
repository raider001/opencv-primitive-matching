package org.example.ui;

import java.awt.*;

/**
 * Shared colour and font constants for the Benchmark Launcher UI.
 * Matches the ProgressDisplay dark-mode palette exactly.
 */
public final class Palette {

    private Palette() {}

    // ── Background / panel colours ────────────────────────────────────────
    public static final Color BG        = new Color(0x0d, 0x11, 0x17);
    public static final Color PANEL     = new Color(0x16, 0x1b, 0x22);
    public static final Color BORDER    = new Color(0x30, 0x36, 0x3d);
    public static final Color ROW_ALT   = new Color(0x13, 0x18, 0x1f);
    public static final Color ROW_SEL   = new Color(0x1a, 0x28, 0x45);
    public static final Color BAR_FILL  = new Color(0x38, 0x8b, 0xff);
    public static final Color BAR_EMPTY = new Color(0x21, 0x26, 0x2d);
    public static final Color BAR_BG    = new Color(0x16, 0x1b, 0x22);
    public static final Color SEP       = new Color(0x30, 0x36, 0x3d);

    // ── Foreground / text colours ─────────────────────────────────────────
    public static final Color HEADER    = new Color(0x58, 0xa6, 0xff);
    public static final Color WHITE     = new Color(0xc9, 0xd1, 0xd9);
    public static final Color DIM       = new Color(0x58, 0x62, 0x6e);
    public static final Color GREEN     = new Color(0x56, 0xd3, 0x64);
    public static final Color ORANGE    = new Color(0xe3, 0x8c, 0x1a);
    public static final Color YELLOW    = new Color(0xd2, 0x99, 0x22);
    public static final Color RED       = new Color(0xf8, 0x51, 0x49);
    public static final Color STATUS_LN = new Color(0x79, 0xc0, 0xff);

    // ── Accent colours ────────────────────────────────────────────────────
    public static final Color ACCENT    = new Color(0x1f, 0x6f, 0xeb);
    public static final Color ACCENT_H  = new Color(0x38, 0x8b, 0xff);
    public static final Color BTN_RED   = new Color(0x4a, 0x1a, 0x1a);
    public static final Color BTN_GREEN = new Color(0x1a, 0x40, 0x20);

    // ── Fonts ─────────────────────────────────────────────────────────────
    public static final Font TITLE = new Font(Font.MONOSPACED, Font.BOLD,  14);
    public static final Font HDR   = new Font(Font.MONOSPACED, Font.BOLD,  12);
    public static final Font BODY  = new Font(Font.MONOSPACED, Font.PLAIN, 12);
    public static final Font SMALL = new Font(Font.MONOSPACED, Font.PLAIN, 11);
    public static final Font BTN   = new Font(Font.SANS_SERIF,  Font.BOLD,  12);
    public static final Font BOLD13= new Font(Font.MONOSPACED, Font.BOLD,  13);
    public static final Font ITALIC10 = new Font(Font.MONOSPACED, Font.ITALIC, 10);
}

