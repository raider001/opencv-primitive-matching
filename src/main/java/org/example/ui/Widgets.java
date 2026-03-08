package org.example.ui;

import javax.swing.*;
import javax.swing.border.*;
import java.awt.*;
import java.awt.event.*;

import static org.example.ui.Palette.*;

/**
 * Static factory helpers that produce consistently styled Swing components.
 */
public final class Widgets {

    private Widgets() {}

    // ── Labels ────────────────────────────────────────────────────────────

    public static JLabel label(String text, Font font, Color fg) {
        JLabel l = new JLabel(text); l.setFont(font); l.setForeground(fg);
        l.setAlignmentX(Component.LEFT_ALIGNMENT); return l;
    }

    public static JLabel headerLabel(String text) { return label(text, HDR, HEADER); }
    public static JLabel monoLabel(String text)   { return label(text, SMALL, WHITE); }
    public static JLabel dimLabel(String text)    { return label(text, SMALL, DIM); }
    public static JLabel titleLabel(String text)  { return label(text, TITLE, HEADER); }
    public static JLabel bold13(String text)      { return label(text, BOLD13, HEADER); }

    public static JLabel groupHeader(String text) {
        JLabel l = label(text, new Font(Font.MONOSPACED, Font.BOLD, 11), HEADER);
        l.setBorder(new EmptyBorder(5,2,1,2)); return l;
    }

    public static JLabel sectionLabel(String text) {
        JLabel l = label(text, SMALL, DIM); l.setBorder(new EmptyBorder(4,0,2,0)); return l;
    }

    // ── Buttons ───────────────────────────────────────────────────────────

    public static JButton accentBtn(String text, Color bg) {
        JButton b = new JButton(text); b.setFont(BTN); b.setBackground(bg); b.setForeground(Color.WHITE);
        b.setBorderPainted(false); b.setFocusPainted(false);
        b.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        b.setBorder(new EmptyBorder(8,16,8,16));
        b.addMouseListener(new MouseAdapter() {
            final Color orig = bg;
            @Override public void mouseEntered(MouseEvent e) { b.setBackground(ACCENT_H); }
            @Override public void mouseExited(MouseEvent e)  { b.setBackground(orig); }
        }); return b;
    }

    public static JButton smallBtn(String text) {
        JButton b = new JButton(text); b.setFont(SMALL); b.setBackground(PANEL); b.setForeground(WHITE);
        b.setBorder(new CompoundBorder(new LineBorder(BORDER), new EmptyBorder(3,8,3,8)));
        b.setFocusPainted(false); b.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR)); return b;
    }

    // ── Checkboxes ────────────────────────────────────────────────────────

    public static JCheckBox check(String text, boolean selected) {
        JCheckBox cb = new JCheckBox(text, selected);
        cb.setFont(BODY); cb.setForeground(WHITE); cb.setBackground(PANEL);
        cb.setAlignmentX(Component.LEFT_ALIGNMENT); cb.setFocusPainted(false); return cb;
    }

    public static JCheckBox smallCheck(String text, boolean selected) {
        JCheckBox cb = new JCheckBox(text, selected);
        cb.setFont(SMALL); cb.setForeground(WHITE); cb.setBackground(PANEL);
        cb.setAlignmentX(Component.LEFT_ALIGNMENT); cb.setFocusPainted(false); return cb;
    }

    // ── Panels ────────────────────────────────────────────────────────────

    /**
     * A titled, dark, BoxLayout-Y panel with a blue header label.
     */
    public static JPanel titledPanel(String title) {
        JPanel p = new JPanel(); p.setLayout(new BoxLayout(p, BoxLayout.Y_AXIS));
        p.setBackground(PANEL);
        p.setBorder(new CompoundBorder(new LineBorder(BORDER,1), new EmptyBorder(8,8,8,8)));
        JLabel l = headerLabel(title);
        l.setAlignmentX(Component.LEFT_ALIGNMENT); l.setBorder(new EmptyBorder(0,0,6,0));
        p.add(l); return p;
    }

    /** A BoxLayout-Y container for stacking items vertically. */
    public static JPanel vStack() {
        JPanel p = new JPanel(); p.setLayout(new BoxLayout(p, BoxLayout.Y_AXIS));
        p.setBackground(PANEL); p.setAlignmentX(Component.LEFT_ALIGNMENT); return p;
    }

    /** A BoxLayout-X container for stacking items horizontally. */
    public static JPanel hStack() {
        JPanel p = new JPanel(); p.setLayout(new BoxLayout(p, BoxLayout.X_AXIS));
        p.setBackground(BG); p.setAlignmentX(Component.LEFT_ALIGNMENT); return p;
    }

    // ── Separators ────────────────────────────────────────────────────────

    public static JPanel sepLine(String text) {
        JPanel p = new JPanel(new BorderLayout(6,0)); p.setBackground(BG);
        p.setAlignmentX(Component.LEFT_ALIGNMENT); p.setMaximumSize(new Dimension(Integer.MAX_VALUE,16));
        p.add(dimLabel(text), BorderLayout.WEST);
        JSeparator sep = new JSeparator(); sep.setForeground(SEP); sep.setBackground(BG);
        p.add(sep, BorderLayout.CENTER); return p;
    }

    // ── Scroll panes ─────────────────────────────────────────────────────

    public static JScrollPane scrollPane(Component c) {
        JScrollPane sp = new JScrollPane(c, JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED,
                JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
        sp.setBackground(PANEL); sp.getViewport().setBackground(PANEL);
        sp.setBorder(BorderFactory.createLineBorder(BORDER,1));
        sp.setAlignmentX(Component.LEFT_ALIGNMENT);
        sp.setMaximumSize(new Dimension(Integer.MAX_VALUE,Integer.MAX_VALUE)); return sp;
    }

    public static JScrollPane scrollPane(Component c, int prefH) {
        JScrollPane sp = scrollPane(c); sp.setPreferredSize(new Dimension(0, prefH)); return sp;
    }

    // ── Gaps ─────────────────────────────────────────────────────────────

    public static Component vgap(int h) { return Box.createRigidArea(new Dimension(0,h)); }
    public static Component hgap(int w) { return Box.createRigidArea(new Dimension(w,0)); }

    // ── Colour helpers ────────────────────────────────────────────────────

    /**
     * Returns GREEN if all generated, ORANGE if partially generated, DIM if none.
     *
     * @param generated number of items already generated
     * @param total     total possible
     */
    public static Color generationColour(int generated, int total) {
        if (total == 0 || generated == 0) return DIM;
        if (generated >= total)           return GREEN;
        return ORANGE;
    }

    /** Status text colour for matcher table (✅/▶/❌ prefixes). */
    public static Color statusColour(String s) {
        if (s.startsWith("✅"))                     return GREEN;
        if (s.startsWith("▶")||s.startsWith("◌")) return YELLOW;
        if (s.startsWith("❌")||s.startsWith("✕")) return RED;
        return DIM;
    }
}

