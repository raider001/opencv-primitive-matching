package org.example.factories;

/**
 * Enumeration of all synthetic reference images used in the matching test suite.
 *
 * Each ID encodes the shape/pattern drawn and whether it is outlined, filled, or gradient-filled.
 * Foreground colour and canvas background are assigned by {@link ReferenceImageFactory}
 * using {@code ordinal() % 8} and {@code ordinal() % 4} respectively.
 */
public enum ReferenceId {

    // -------------------------------------------------------------------------
    // Lines  (8)
    // -------------------------------------------------------------------------
    LINE_H,           // Single horizontal line through centre
    LINE_V,           // Single vertical line through centre
    LINE_DIAG_45,     // Diagonal line at 45°
    LINE_DIAG_135,    // Diagonal line at 135°
    LINE_CROSS,       // Horizontal + vertical cross (+)
    LINE_X,           // Two diagonal lines (×)
    LINE_MULTI_H,     // Three evenly spaced horizontal lines
    LINE_MULTI_V,     // Three evenly spaced vertical lines

    // -------------------------------------------------------------------------
    // Circles & Ellipses  (8)
    // -------------------------------------------------------------------------
    CIRCLE_OUTLINE,           // Single circle outline
    CIRCLE_FILLED,            // Solid filled circle
    CIRCLE_FILLED_GRADIENT,   // Filled circle with radial gradient fill
    CIRCLE_SMALL,             // Small circle (radius ~16 px)
    CIRCLE_LARGE,             // Large circle (radius ~56 px)
    CIRCLE_CONCENTRIC,        // Three concentric circle outlines
    ELLIPSE_H,                // Wide horizontal ellipse outline
    ELLIPSE_V,                // Tall vertical ellipse outline

    // -------------------------------------------------------------------------
    // Rectangles  (5)
    // -------------------------------------------------------------------------
    RECT_OUTLINE,             // Rectangle outline
    RECT_FILLED,              // Solid filled rectangle
    RECT_FILLED_GRADIENT,     // Filled rectangle with linear gradient fill
    RECT_SQUARE,              // Perfect square outline
    RECT_THIN,                // Thin wide rectangle (aspect ratio ~4:1)

    // -------------------------------------------------------------------------
    // Regular Polygons — outline & filled  (14)
    // -------------------------------------------------------------------------
    TRIANGLE_OUTLINE,         // Equilateral triangle outline
    TRIANGLE_FILLED,          // Solid filled triangle
    TRIANGLE_FILLED_GRADIENT, // Filled triangle with linear gradient fill
    PENTAGON_OUTLINE,         // Regular pentagon outline
    PENTAGON_FILLED,          // Solid filled pentagon
    HEXAGON_OUTLINE,          // Regular hexagon outline
    HEXAGON_FILLED,           // Solid filled hexagon
    HEPTAGON_OUTLINE,         // Regular heptagon (7 sides) outline
    OCTAGON_OUTLINE,          // Regular octagon outline
    OCTAGON_FILLED,           // Solid filled octagon
    STAR_4_OUTLINE,           // 4-pointed star outline
    STAR_5_OUTLINE,           // 5-pointed star outline
    STAR_5_FILLED,            // 5-pointed star filled
    STAR_6_OUTLINE,           // 6-pointed star (Star of David) outline

    // -------------------------------------------------------------------------
    // Polylines — open & closed paths  (12)
    // -------------------------------------------------------------------------
    POLYLINE_ZIGZAG_H,        // Horizontal zigzag (open polyline, 6 points)
    POLYLINE_ZIGZAG_V,        // Vertical zigzag (open polyline, 6 points)
    POLYLINE_WAVE,            // Sine-wave approximation (open polyline, 16 points)
    POLYLINE_SPIRAL,          // Inward spiral (open polyline, 32 points)
    POLYLINE_ARROW_RIGHT,     // Right-pointing arrow (closed polyline)
    POLYLINE_ARROW_LEFT,      // Left-pointing arrow (closed polyline)
    POLYLINE_L_SHAPE,         // L-shaped path (closed)
    POLYLINE_T_SHAPE,         // T-shaped path (closed)
    POLYLINE_PLUS_SHAPE,      // Plus/cross shape (closed filled poly)
    POLYLINE_CHEVRON,         // Chevron / caret shape (closed)
    POLYLINE_DIAMOND,         // Diamond (rotated square, closed)
    POLYLINE_PARALLELOGRAM,   // Parallelogram (closed)

    // -------------------------------------------------------------------------
    // Arcs & Partial Curves  (6)
    // -------------------------------------------------------------------------
    ARC_QUARTER,          // Quarter-circle arc (90°)
    ARC_HALF,             // Semicircle arc (180°)
    ARC_THREE_QUARTER,    // Three-quarter arc (270°)
    ARC_OPEN_ELLIPSE,     // Partial ellipse arc (~200°, horizontal)
    ARC_BRACKET_LEFT,     // Left bracket shape — two arcs facing right
    ARC_BRACKET_RIGHT,    // Right bracket shape — two arcs facing left

    // -------------------------------------------------------------------------
    // Concave / Irregular Polygons  (6)
    // -------------------------------------------------------------------------
    CONCAVE_ARROW_HEAD,   // Simple arrowhead (concave base notch)
    CONCAVE_MOON,         // Crescent/moon shape (concave cutout)
    CONCAVE_PAC_MAN,      // Pac-Man shape (filled circle with wedge removed)
    IRREGULAR_QUAD,       // Irregular quadrilateral (no parallel sides)
    IRREGULAR_PENTA,      // Irregular asymmetric pentagon
    IRREGULAR_STAR,       // Irregular 5-pointed star (unequal tip lengths)

    // -------------------------------------------------------------------------
    // Rotated / Skewed Rectangles  (4)
    // -------------------------------------------------------------------------
    RECT_ROTATED_15,      // Rectangle rotated 15°
    RECT_ROTATED_30,      // Rectangle rotated 30°
    RECT_ROTATED_45,      // Rectangle rotated 45° (diamond aspect)
    RECT_ROTATED_60,      // Rectangle rotated 60°

    // -------------------------------------------------------------------------
    // Dashed & Dotted Lines  (5)
    // -------------------------------------------------------------------------
    LINE_DASHED_H,        // Horizontal dashed line (8px on, 8px off)
    LINE_DASHED_DIAG,     // Diagonal dashed line at 45°
    LINE_DOTTED_H,        // Horizontal dotted line (2px dots, 6px gap)
    LINE_DASHED_CROSS,    // Dashed cross (+) both axes
    LINE_DASHED_RECT,     // Dashed rectangle outline

    // -------------------------------------------------------------------------
    // Compound / Nested Shapes  (7)
    // -------------------------------------------------------------------------
    COMPOUND_CIRCLE_IN_RECT,    // Circle inscribed inside a rectangle
    COMPOUND_RECT_IN_CIRCLE,    // Rectangle inscribed inside a circle
    COMPOUND_TRIANGLE_IN_CIRCLE,// Triangle inscribed inside a circle
    COMPOUND_CONCENTRIC_RECTS,  // Three concentric rectangles
    COMPOUND_BULLSEYE,          // Filled circle + two outline rings (target)
    COMPOUND_CROSS_IN_CIRCLE,   // Circle with cross inside
    COMPOUND_NESTED_TRIANGLES,  // Two triangles, one inverted (Star of David variant)

    // -------------------------------------------------------------------------
    // Grids & Patterns  (7)
    // -------------------------------------------------------------------------
    GRID_2X2,           // 2×2 grid lines
    GRID_4X4,           // 4×4 grid lines
    GRID_8X8,           // 8×8 fine grid lines
    GRID_DOT_4X4,       // 4×4 array of filled dots
    CHECKERBOARD_2X2,   // 2×2 filled checkerboard (foreground + black alternating)
    CHECKERBOARD_4X4,   // 4×4 filled checkerboard (foreground + black alternating)
    CROSSHAIR,          // Fine crosshair with centre dot

    // -------------------------------------------------------------------------
    // Text  (6)
    // -------------------------------------------------------------------------
    TEXT_A,      // Single large letter "A"
    TEXT_X,      // Single large letter "X"
    TEXT_O,      // Single large letter "O"
    TEXT_HELLO,  // Small word "HELLO"
    TEXT_123,    // Digits "123"
    TEXT_MIXED,  // Mixed "Ab3" (mixed case + digit)

    // -------------------------------------------------------------------------
    // Multi-Colour Shapes  (5) — Milestone 21
    // These shapes use two or three distinct hues on a solid-black canvas.
    // They are used to validate the Multi-Colour-First (MCF1) region proposal
    // engine which must propose windows for each colour independently.
    // -------------------------------------------------------------------------
    BICOLOUR_CIRCLE_RING,    // Circle outline in palette colour, filled centre in complementary hue (+90°)
    BICOLOUR_RECT_HALVES,    // Rectangle split horizontally: top half hue A, bottom half hue B (≥60° apart)
    TRICOLOUR_TRIANGLE,      // Equilateral triangle with three 120°-apart hue regions
    BICOLOUR_CROSSHAIR_RING, // Crosshair lines in one colour, surrounding circle ring in a second colour
    BICOLOUR_CHEVRON_FILLED, // Chevron outline in colour A, interior flood-filled with colour B

    // -------------------------------------------------------------------------
    // Individual Characters — size 12 Mono style (66)
    // Each entry renders a single character centred on a 128×128 canvas using
    // FONT_HERSHEY_PLAIN at the largest scale that keeps the glyph within the
    // canvas bounds.  Used by CharacterMatchingTest for per-character and
    // alphabet-scene VectorMatcher validation.
    // -------------------------------------------------------------------------

    // Lowercase a–z  (26)
    CHAR_LOWER_A, CHAR_LOWER_B, CHAR_LOWER_C, CHAR_LOWER_D, CHAR_LOWER_E,
    CHAR_LOWER_F, CHAR_LOWER_G, CHAR_LOWER_H, CHAR_LOWER_I, CHAR_LOWER_J,
    CHAR_LOWER_K, CHAR_LOWER_L, CHAR_LOWER_M, CHAR_LOWER_N, CHAR_LOWER_O,
    CHAR_LOWER_P, CHAR_LOWER_Q, CHAR_LOWER_R, CHAR_LOWER_S, CHAR_LOWER_T,
    CHAR_LOWER_U, CHAR_LOWER_V, CHAR_LOWER_W, CHAR_LOWER_X, CHAR_LOWER_Y,
    CHAR_LOWER_Z,

    // Uppercase A–Z  (26)
    CHAR_UPPER_A, CHAR_UPPER_B, CHAR_UPPER_C, CHAR_UPPER_D, CHAR_UPPER_E,
    CHAR_UPPER_F, CHAR_UPPER_G, CHAR_UPPER_H, CHAR_UPPER_I, CHAR_UPPER_J,
    CHAR_UPPER_K, CHAR_UPPER_L, CHAR_UPPER_M, CHAR_UPPER_N, CHAR_UPPER_O,
    CHAR_UPPER_P, CHAR_UPPER_Q, CHAR_UPPER_R, CHAR_UPPER_S, CHAR_UPPER_T,
    CHAR_UPPER_U, CHAR_UPPER_V, CHAR_UPPER_W, CHAR_UPPER_X, CHAR_UPPER_Y,
    CHAR_UPPER_Z,

    // Digits 0–9  (10)
    CHAR_0, CHAR_1, CHAR_2, CHAR_3, CHAR_4,
    CHAR_5, CHAR_6, CHAR_7, CHAR_8, CHAR_9,

    // Punctuation  (4)
    CHAR_PERIOD,  // Full stop "."
    CHAR_COMMA,   // Comma ","
    CHAR_SQUOTE,  // Single quotation mark "'"
    CHAR_DQUOTE   // Double quotation mark "\""
}



