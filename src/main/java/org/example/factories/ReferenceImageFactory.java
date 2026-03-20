package org.example.factories;

import org.example.colour.ColourFirstLocator;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Generates all 47 synthetic reference {@link Mat} images.
 *
 * <p>Each image is 128×128 px, 3-channel BGR colour.
 *
 * <p><b>Foreground colour</b> — cycles through an 8-colour palette by {@code id.ordinal() % 8}:
 * white, red, green, blue, yellow, cyan, magenta, orange.
 *
 * <p><b>Canvas background</b> — cycles through 4 solid fills by {@code id.ordinal() % 4}:
 * solid black, solid dark grey, solid dark blue, solid dark green.
 *
 * <p>Gradient-fill variants ({@code *_GRADIENT}) paint the shape interior with a linear
 * colour sweep from the foreground colour to a lighter complementary tone using scanline
 * {@link Imgproc#line} strokes over the filled mask.
 */
public final class ReferenceImageFactory {

    public static final int SIZE = 128;

    // -------------------------------------------------------------------------
    // Foreground colour palette  (BGR)
    // -------------------------------------------------------------------------
    private static final Scalar[] FG_PALETTE = {
        new Scalar(255, 255, 255),  // 0 white
        new Scalar(0,   0,   220),  // 1 red
        new Scalar(0,   200,   0),  // 2 green
        new Scalar(220,   0,   0),  // 3 blue
        new Scalar(0,   220, 220),  // 4 yellow
        new Scalar(220, 220,   0),  // 5 cyan
        new Scalar(220,   0, 220),  // 6 magenta
        new Scalar(0,   140, 255),  // 7 orange
    };

    // Lighter complementary tones for gradient endpoints  (BGR)
    private static final Scalar[] FG_LIGHT = {
        new Scalar(200, 200, 200),  // 0 light grey
        new Scalar(120, 120, 255),  // 1 light red/pink
        new Scalar(120, 255, 120),  // 2 light green
        new Scalar(255, 120, 120),  // 3 light blue
        new Scalar(120, 255, 255),  // 4 light yellow
        new Scalar(255, 255, 120),  // 5 light cyan
        new Scalar(255, 120, 255),  // 6 light magenta
        new Scalar(120, 200, 255),  // 7 light orange
    };

    private ReferenceImageFactory() {}

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    private static final Path REFERENCES_DIR = Paths.get("test_output", "references");

    /**
     * Returns a fresh 128×128 BGR {@link Mat} for the given reference ID.
     *
     * <p>Loads from {@code test_output/references/<ID>.png} if the file exists on disk,
     * so the suite reuses the previously generated (and potentially manually curated)
     * images rather than regenerating them each run.  Falls back to programmatic
     * generation if the file is absent or cannot be decoded.
     */
    public static Mat build(ReferenceId id) {
        Path file = REFERENCES_DIR.resolve(id.name() + ".png");
        if (Files.exists(file)) {
            Mat loaded = Imgcodecs.imread(file.toAbsolutePath().toString());
            if (loaded != null && !loaded.empty()) {
                // Ensure it is exactly SIZE×SIZE BGR
                if (loaded.cols() == SIZE && loaded.rows() == SIZE
                        && loaded.channels() == 3) {
                    return loaded;
                }
                // Wrong size — resize to SIZE×SIZE
                Mat resized = new Mat();
                Imgproc.resize(loaded, resized, new Size(SIZE, SIZE));
                loaded.release();
                return resized;
            }
            if (loaded != null) loaded.release();
        }
        // Fallback: generate programmatically
        return buildProgrammatic(id);
    }

    /** Builds the reference image entirely programmatically (no disk I/O). */
    public static Mat buildProgrammatic(ReferenceId id) {
        int idx = id.ordinal();
        Scalar fg   = FG_PALETTE[idx % FG_PALETTE.length];
        Scalar fgLt = FG_LIGHT  [idx % FG_LIGHT.length];
        Mat canvas  = buildBackground(idx);
        drawShape(canvas, id, fg, fgLt);
        return canvas;
    }

    /**
     * Builds an 8-bit single-channel foreground mask for the given reference Mat.
     *
     * <p>Pixels that are non-black (any channel > {@value #MASK_THRESH}) are set to 255
     * (foreground); all-black pixels are set to 0 (background / transparent).
     *
     * <p>Since references always use a solid-black canvas, this cleanly isolates the
     * drawn shape.  The returned {@link Mat} is the same size as {@code refMat} and
     * is owned by the caller (must be released when done).
     *
     * @param refMat 3-channel BGR reference image (not modified)
     * @return single-channel CV_8UC1 mask, same dimensions as {@code refMat}
     */
    public static Mat buildMask(Mat refMat) {
        Mat grey = new Mat();
        Imgproc.cvtColor(refMat, grey, Imgproc.COLOR_BGR2GRAY);
        Mat mask = new Mat();
        Imgproc.threshold(grey, mask, MASK_THRESH, 255, Imgproc.THRESH_BINARY);
        grey.release();
        return mask;
    }

    /** Threshold above which a greyscale pixel is considered foreground for masking. */
    public static final int MASK_THRESH = 10;

    /** Returns the foreground {@link Scalar} (BGR) assigned to this reference ID. */
    public static Scalar foregroundColour(ReferenceId id) {
        return FG_PALETTE[id.ordinal() % FG_PALETTE.length];
    }

    /** Returns a human-readable name for the foreground colour of the given ID. */
    public static String foregroundColourName(ReferenceId id) {
        String[] names = {"white","red","green","blue","yellow","cyan","magenta","orange"};
        return names[id.ordinal() % names.length];
    }

    /** Returns a human-readable name for the background fill of the given ID. */
    public static String backgroundFillName(ReferenceId id) { // NOSONAR id kept for API compat
        return "solid black";
    }

    // -------------------------------------------------------------------------
    // Background builders
    // -------------------------------------------------------------------------

    private static Mat buildBackground(@SuppressWarnings("unused") int idx) {
        // Always solid black — the shape must sit on a clean black canvas so it
        // composes correctly onto any scene background and colour pre-filters can
        // isolate it accurately.  Non-black reference canvases contaminate template
        // matching, histogram comparison, and CF mask quality.
        return new Mat(SIZE, SIZE, CvType.CV_8UC3, new Scalar(0, 0, 0));
    }


    // -------------------------------------------------------------------------
    // Shape dispatcher
    // -------------------------------------------------------------------------

    private static void drawShape(Mat m, ReferenceId id, Scalar fg, Scalar fgLt) {
        switch (id) {
            // Lines
            case LINE_H          -> Imgproc.line(m, new Point(8, SIZE/2), new Point(SIZE-9, SIZE/2), fg, 2);
            case LINE_V          -> Imgproc.line(m, new Point(SIZE/2, 8), new Point(SIZE/2, SIZE-9), fg, 2);
            case LINE_DIAG_45    -> Imgproc.line(m, new Point(8, SIZE-9), new Point(SIZE-9, 8), fg, 2);
            case LINE_DIAG_135   -> Imgproc.line(m, new Point(8, 8), new Point(SIZE-9, SIZE-9), fg, 2);
            case LINE_CROSS      -> {
                Imgproc.line(m, new Point(8, SIZE/2), new Point(SIZE-9, SIZE/2), fg, 2);
                Imgproc.line(m, new Point(SIZE/2, 8), new Point(SIZE/2, SIZE-9), fg, 2);
            }
            case LINE_X          -> {
                Imgproc.line(m, new Point(8, 8), new Point(SIZE-9, SIZE-9), fg, 2);
                Imgproc.line(m, new Point(SIZE-9, 8), new Point(8, SIZE-9), fg, 2);
            }
            case LINE_MULTI_H    -> {
                for (int y : new int[]{SIZE/4, SIZE/2, 3*SIZE/4})
                    Imgproc.line(m, new Point(8, y), new Point(SIZE-9, y), fg, 2);
            }
            case LINE_MULTI_V    -> {
                for (int x : new int[]{SIZE/4, SIZE/2, 3*SIZE/4})
                    Imgproc.line(m, new Point(x, 8), new Point(x, SIZE-9), fg, 2);
            }

            // Circles & Ellipses
            case CIRCLE_OUTLINE        -> Imgproc.circle(m, centre(), SIZE/2-8, fg, 2);
            case CIRCLE_FILLED         -> Imgproc.circle(m, centre(), SIZE/2-8, fg, -1);
            case CIRCLE_FILLED_GRADIENT-> drawGradientFilledCircle(m, fg, fgLt);
            case CIRCLE_SMALL          -> Imgproc.circle(m, centre(), 16, fg, 2);
            case CIRCLE_LARGE          -> Imgproc.circle(m, centre(), SIZE/2-4, fg, 2);
            case CIRCLE_CONCENTRIC     -> {
                for (int r : new int[]{SIZE/2-8, SIZE/3, SIZE/5})
                    Imgproc.circle(m, centre(), r, fg, 2);
            }
            case ELLIPSE_H -> Imgproc.ellipse(m, centre(), new Size(SIZE/2-8, SIZE/4-4), 0, 0, 360, fg, 2);
            case ELLIPSE_V -> Imgproc.ellipse(m, centre(), new Size(SIZE/4-4, SIZE/2-8), 0, 0, 360, fg, 2);

            // Rectangles & Polygons
            case RECT_OUTLINE        -> Imgproc.rectangle(m, new Point(12, 24), new Point(SIZE-13, SIZE-25), fg, 2);
            case RECT_FILLED         -> Imgproc.rectangle(m, new Point(12, 24), new Point(SIZE-13, SIZE-25), fg, -1);
            case RECT_FILLED_GRADIENT-> drawGradientFilledRect(m, new Point(12, 24), new Point(SIZE-13, SIZE-25), fg, fgLt);
            case RECT_SQUARE         -> Imgproc.rectangle(m, new Point(16, 16), new Point(SIZE-17, SIZE-17), fg, 2);
            case RECT_THIN           -> Imgproc.rectangle(m, new Point(8, 48), new Point(SIZE-9, SIZE-49), fg, 2);
            case TRIANGLE_OUTLINE    -> drawTriangle(m, fg, false);
            case TRIANGLE_FILLED     -> drawTriangle(m, fg, true);
            case TRIANGLE_FILLED_GRADIENT -> drawGradientFilledTriangle(m, fg, fgLt);
            case PENTAGON_OUTLINE    -> drawRegularPolygon(m, 5, fg, false);
            case PENTAGON_FILLED     -> drawRegularPolygon(m, 5, fg, true);
            case HEXAGON_OUTLINE     -> drawRegularPolygon(m, 6, fg, false);
            case HEXAGON_FILLED      -> drawRegularPolygon(m, 6, fg, true);
            case HEPTAGON_OUTLINE    -> drawRegularPolygon(m, 7, fg, false);
            case OCTAGON_OUTLINE     -> drawRegularPolygon(m, 8, fg, false);
            case OCTAGON_FILLED      -> drawRegularPolygon(m, 8, fg, true);
            case STAR_4_OUTLINE      -> drawStar(m, 4, fg, false);
            case STAR_5_OUTLINE      -> drawStar(m, 5, fg, false);
            case STAR_5_FILLED       -> drawStar(m, 5, fg, true);
            case STAR_6_OUTLINE      -> drawStar(m, 6, fg, false);

            // Polylines
            case POLYLINE_ZIGZAG_H      -> drawZigzag(m, fg, true);
            case POLYLINE_ZIGZAG_V      -> drawZigzag(m, fg, false);
            case POLYLINE_WAVE          -> drawWave(m, fg);
            case POLYLINE_SPIRAL        -> drawSpiral(m, fg);
            case POLYLINE_ARROW_RIGHT   -> drawArrow(m, fg, true);
            case POLYLINE_ARROW_LEFT    -> drawArrow(m, fg, false);
            case POLYLINE_L_SHAPE       -> drawLShape(m, fg);
            case POLYLINE_T_SHAPE       -> drawTShape(m, fg);
            case POLYLINE_PLUS_SHAPE    -> drawPlusShape(m, fg);
            case POLYLINE_CHEVRON       -> drawChevron(m, fg);
            case POLYLINE_DIAMOND       -> drawDiamond(m, fg);
            case POLYLINE_PARALLELOGRAM -> drawParallelogram(m, fg);

            // Arcs & Partial Curves
            case ARC_QUARTER        -> Imgproc.ellipse(m, centre(), new Size(SIZE/2-10, SIZE/2-10), 0,   0,  90, fg, 2);
            case ARC_HALF           -> Imgproc.ellipse(m, centre(), new Size(SIZE/2-10, SIZE/2-10), 0,   0, 180, fg, 2);
            case ARC_THREE_QUARTER  -> Imgproc.ellipse(m, centre(), new Size(SIZE/2-10, SIZE/2-10), 0,   0, 270, fg, 2);
            case ARC_OPEN_ELLIPSE   -> Imgproc.ellipse(m, centre(), new Size(SIZE/2-10, SIZE/4-6),  0,  20, 220, fg, 2);
            case ARC_BRACKET_LEFT  -> {
                Imgproc.ellipse(m, new Point(SIZE*0.6, SIZE*0.35), new Size(22, 22), 0, 90,  270, fg, 2);
                Imgproc.ellipse(m, new Point(SIZE*0.6, SIZE*0.65), new Size(22, 22), 0, 90,  270, fg, 2);
            }
            case ARC_BRACKET_RIGHT -> {
                Imgproc.ellipse(m, new Point(SIZE*0.4, SIZE*0.35), new Size(22, 22), 0, 270, 450, fg, 2);
                Imgproc.ellipse(m, new Point(SIZE*0.4, SIZE*0.65), new Size(22, 22), 0, 270, 450, fg, 2);
            }

            // Concave / Irregular Polygons
            case CONCAVE_ARROW_HEAD   -> drawArrowHead(m, fg);
            case CONCAVE_MOON         -> drawMoon(m, fg);
            case CONCAVE_PAC_MAN      -> drawPacMan(m, fg);
            case IRREGULAR_QUAD       -> drawIrregularQuad(m, fg);
            case IRREGULAR_PENTA      -> drawIrregularPenta(m, fg);
            case IRREGULAR_STAR       -> drawIrregularStar(m, fg);

            // Rotated Rectangles
            case RECT_ROTATED_15 -> drawRotatedRect(m, fg, 15);
            case RECT_ROTATED_30 -> drawRotatedRect(m, fg, 30);
            case RECT_ROTATED_45 -> drawRotatedRect(m, fg, 45);
            case RECT_ROTATED_60 -> drawRotatedRect(m, fg, 60);

            // Dashed & Dotted Lines
            case LINE_DASHED_H    -> drawDashedLine(m, fg, new Point(8, SIZE/2.0), new Point(SIZE-9, SIZE/2.0), 8, 8, false);
            case LINE_DASHED_DIAG -> drawDashedLine(m, fg, new Point(8, SIZE-9),   new Point(SIZE-9, 8),         8, 8, false);
            case LINE_DOTTED_H    -> drawDashedLine(m, fg, new Point(8, SIZE/2.0), new Point(SIZE-9, SIZE/2.0), 2, 6, false);
            case LINE_DASHED_CROSS -> {
                drawDashedLine(m, fg, new Point(8, SIZE/2.0), new Point(SIZE-9, SIZE/2.0), 8, 8, false);
                drawDashedLine(m, fg, new Point(SIZE/2.0, 8), new Point(SIZE/2.0, SIZE-9), 8, 8, false);
            }
            case LINE_DASHED_RECT -> drawDashedRect(m, fg, new Point(16, 16), new Point(SIZE-17, SIZE-17), 8, 8);

            // Compound / Nested Shapes
            case COMPOUND_CIRCLE_IN_RECT -> {
                Imgproc.rectangle(m, new Point(10, 10), new Point(SIZE-11, SIZE-11), fg, 2);
                Imgproc.circle(m, centre(), SIZE/2-16, fg, 2);
            }
            case COMPOUND_RECT_IN_CIRCLE -> {
                Imgproc.circle(m, centre(), SIZE/2-10, fg, 2);
                Imgproc.rectangle(m, new Point(28, 28), new Point(SIZE-29, SIZE-29), fg, 2);
            }
            case COMPOUND_TRIANGLE_IN_CIRCLE -> {
                Imgproc.circle(m, centre(), SIZE/2-10, fg, 2);
                drawTriangle(m, fg, false);
            }
            case COMPOUND_CONCENTRIC_RECTS -> {
                for (int pad : new int[]{10, 26, 42})
                    Imgproc.rectangle(m, new Point(pad, pad), new Point(SIZE-pad-1, SIZE-pad-1), fg, 2);
            }
            case COMPOUND_BULLSEYE -> {
                Imgproc.circle(m, centre(), SIZE/2-10, fg, 2);
                Imgproc.circle(m, centre(), SIZE/2-28, fg, 2);
                Imgproc.circle(m, centre(), 12, fg, -1);
            }
            case COMPOUND_CROSS_IN_CIRCLE -> {
                Imgproc.circle(m, centre(), SIZE/2-10, fg, 2);
                Imgproc.line(m, new Point(SIZE/2.0, 14), new Point(SIZE/2.0, SIZE-15), fg, 2);
                Imgproc.line(m, new Point(14, SIZE/2.0), new Point(SIZE-15, SIZE/2.0), fg, 2);
            }
            case COMPOUND_NESTED_TRIANGLES -> {
                drawTriangle(m, fg, false);
                drawInvertedTriangle(m, fg);
            }

            // Grids & Patterns
            case GRID_2X2 -> drawGrid(m, 2, fg);
            case GRID_4X4 -> drawGrid(m, 4, fg);
            case GRID_8X8 -> drawGrid(m, 8, fg);
            case GRID_DOT_4X4 -> drawDotGrid(m, 4, fg);
            case CHECKERBOARD_2X2 -> drawCheckerboard(m, 2, fg);
            case CHECKERBOARD_4X4 -> drawCheckerboard(m, 4, fg);
            case CROSSHAIR -> {
                int cx = SIZE / 2, cy = SIZE / 2;
                Imgproc.line(m, new Point(8, cy), new Point(SIZE-9, cy), fg, 1);
                Imgproc.line(m, new Point(cx, 8), new Point(cx, SIZE-9), fg, 1);
                Imgproc.circle(m, centre(), 3, fg, -1);
            }

            // Text
            case TEXT_A     -> drawCentredText(m, "A",     fg, 3.0, 6);
            case TEXT_X     -> drawCentredText(m, "X",     fg, 3.0, 6);
            case TEXT_O     -> drawCentredText(m, "O",     fg, 3.0, 6);
            case TEXT_HELLO -> drawCentredText(m, "HELLO", fg, 1.1, 2);
            case TEXT_123   -> drawCentredText(m, "123",   fg, 1.8, 3);
            case TEXT_MIXED -> drawCentredText(m, "Ab3",   fg, 1.8, 3);

            // Multi-colour shapes (Milestone 21)
            case BICOLOUR_CIRCLE_RING    -> drawBicolourCircleRing(m);
            case BICOLOUR_RECT_HALVES    -> drawBicolourRectHalves(m);
            case TRICOLOUR_TRIANGLE      -> drawTricolourTriangle(m);
            case BICOLOUR_CROSSHAIR_RING -> drawBicolourCrosshairRing(m);
            case BICOLOUR_CHEVRON_FILLED -> drawBicolourChevronFilled(m);
        }
    }

    // =========================================================================
    // Multi-colour foreground colour registration  (Milestone 21)
    // =========================================================================

    // Fixed saturated BGR colours used by the multi-colour shapes (well-separated hues)
    static final Scalar MCOL_RED     = new Scalar(  0,   0, 220);  // H≈0
    static final Scalar MCOL_GREEN   = new Scalar(  0, 200,   0);  // H≈60
    static final Scalar MCOL_BLUE    = new Scalar(220,   0,   0);  // H≈120
    static final Scalar MCOL_YELLOW  = new Scalar(  0, 220, 220);  // H≈30
    static final Scalar MCOL_CYAN    = new Scalar(220, 220,   0);  // H≈90
    static final Scalar MCOL_MAGENTA = new Scalar(220,   0, 220);  // H≈150
    static final Scalar MCOL_ORANGE  = new Scalar(  0, 140, 255);  // H≈15

    /**
     * Returns the list of distinct foreground BGR colours drawn on this reference.
     *
     * <p>Single-colour references return a list of exactly one element
     * ({@link #foregroundColour(ReferenceId)}).  Multi-colour references (BICOLOUR_*
     * and TRICOLOUR_*) return each distinct hue in drawing order (primary colour first).
     *
     * <p>This is the authoritative source used by
     * {@link ColourFirstLocator#proposeMulti} to build per-channel HSV windows.
     */
    public static List<Scalar> foregroundColours(ReferenceId id) {
        return switch (id) {
            case BICOLOUR_CIRCLE_RING    -> List.of(MCOL_RED,    MCOL_GREEN);
            case BICOLOUR_RECT_HALVES    -> List.of(MCOL_CYAN,   MCOL_MAGENTA);
            case TRICOLOUR_TRIANGLE      -> List.of(MCOL_RED,    MCOL_GREEN,   MCOL_BLUE);
            case BICOLOUR_CROSSHAIR_RING -> List.of(MCOL_YELLOW, MCOL_CYAN);
            case BICOLOUR_CHEVRON_FILLED -> List.of(MCOL_ORANGE, MCOL_MAGENTA);
            default                      -> List.of(foregroundColour(id));
        };
    }

    // -------------------------------------------------------------------------
    // Gradient fill helpers
    // -------------------------------------------------------------------------

    private static void drawGradientFilledCircle(Mat m, Scalar fg, Scalar fgLt) {
        int r = SIZE / 2 - 8;
        // Fill mask
        Mat mask = Mat.zeros(SIZE, SIZE, CvType.CV_8UC1);
        Imgproc.circle(mask, centre(), r, new Scalar(255), -1);
        // Paint scanlines with gradient colour
        for (int y = (int) centre().y - r; y <= (int) centre().y + r; y++) {
            double t = (double)(y - ((int) centre().y - r)) / (2.0 * r);
            Scalar col = blend(fg, fgLt, t);
            Imgproc.line(m, new Point(0, y), new Point(SIZE - 1, y), col, 1);
        }
        // Restore pixels outside mask to their background value by re-clearing them
        // Simpler: composite using the mask
        Mat coloured = m.clone();
        m.setTo(new Scalar(0, 0, 0));  // will be overwritten by copyTo
        // Restore bg first, then composite shape
        Mat bg = buildBackground(0); // will be re-done properly below
        bg.release();
        // Practical approach: just redraw bg then apply gradient via mask
        // (The canvas was already set up by build() before drawShape was called,
        //  so we simply apply the gradient lines only within the mask region)
        coloured.copyTo(m, mask);
        mask.release();
        coloured.release();
        // Draw outline on top
        Imgproc.circle(m, centre(), r, fg, 2);
    }

    private static void drawGradientFilledRect(Mat m, Point tl, Point br, Scalar fg, Scalar fgLt) {
        int top = (int) tl.y, bot = (int) br.y;
        for (int y = top; y <= bot; y++) {
            double t = (double)(y - top) / (bot - top);
            Scalar col = blend(fg, fgLt, t);
            Imgproc.line(m, new Point(tl.x, y), new Point(br.x, y), col, 1);
        }
        Imgproc.rectangle(m, tl, br, fg, 2);
    }

    private static void drawGradientFilledTriangle(Mat m, Scalar fg, Scalar fgLt) {
        int pad = 10;
        List<Point> pts = trianglePoints(pad);
        // Find bounding y
        int minY = pts.stream().mapToInt(p -> (int) p.y).min().orElse(0);
        int maxY = pts.stream().mapToInt(p -> (int) p.y).max().orElse(SIZE);
        // Create mask
        MatOfPoint mop = new MatOfPoint();
        mop.fromList(pts);
        Mat mask = Mat.zeros(SIZE, SIZE, CvType.CV_8UC1);
        Imgproc.fillPoly(mask, List.of(mop), new Scalar(255));
        // Scanline gradient
        for (int y = minY; y <= maxY; y++) {
            double t = (double)(y - minY) / Math.max(1, maxY - minY);
            Scalar col = blend(fg, fgLt, t);
            Imgproc.line(m, new Point(0, y), new Point(SIZE - 1, y), col, 1);
        }
        // Composite
        Mat coloured = m.clone();
        coloured.copyTo(m, mask);
        mask.release();
        coloured.release();
        // Outline
        Imgproc.polylines(m, List.of(mop), true, fg, 2);
    }

    // -------------------------------------------------------------------------
    // Shape helpers
    // -------------------------------------------------------------------------

    private static void drawTriangle(Mat m, Scalar fg, boolean filled) {
        List<Point> pts = trianglePoints(10);
        MatOfPoint mop = new MatOfPoint();
        mop.fromList(pts);
        if (filled) {
            Imgproc.fillPoly(m, List.of(mop), fg);
        } else {
            Imgproc.polylines(m, List.of(mop), true, fg, 2);
        }
    }

    private static List<Point> trianglePoints(int pad) {
        return List.of(
            new Point(SIZE / 2.0, pad),
            new Point(SIZE - pad, SIZE - pad),
            new Point(pad, SIZE - pad)
        );
    }

    private static void drawRegularPolygon(Mat m, int sides, Scalar fg, boolean filled) {
        double r = SIZE / 2.0 - 10;        List<Point> pts = new ArrayList<>();
        for (int i = 0; i < sides; i++) {
            double angle = Math.toRadians(-90 + (360.0 / sides) * i);
            pts.add(new Point(SIZE / 2.0 + r * Math.cos(angle),
                              SIZE / 2.0 + r * Math.sin(angle)));
        }
        MatOfPoint mop = new MatOfPoint();
        mop.fromList(pts);
        if (filled) {
            Imgproc.fillPoly(m, List.of(mop), fg);
        } else {
            Imgproc.polylines(m, List.of(mop), true, fg, 2);
        }
    }

    private static void drawGrid(Mat m, int divisions, Scalar fg) {
        int step = SIZE / divisions;
        for (int i = 0; i <= divisions; i++) {
            int pos = Math.min(i * step, SIZE - 1);
            Imgproc.line(m, new Point(pos, 0), new Point(pos, SIZE - 1), fg, 1);
            Imgproc.line(m, new Point(0, pos), new Point(SIZE - 1, pos), fg, 1);
        }
    }

    private static void drawDotGrid(Mat m, int divisions, Scalar fg) {
        int step = SIZE / (divisions + 1);
        for (int row = 1; row <= divisions; row++) {
            for (int col = 1; col <= divisions; col++) {
                Imgproc.circle(m, new Point(col * step, row * step), 4, fg, -1);
            }
        }
    }

    private static void drawCheckerboard(Mat m, int divisions, Scalar fg) {
        int cellW = SIZE / divisions;
        int cellH = SIZE / divisions;
        for (int row = 0; row < divisions; row++) {
            for (int col = 0; col < divisions; col++) {
                if ((row + col) % 2 == 0) {
                    Point tl = new Point((double) col * cellW, (double) row * cellH);
                    Point br = new Point(tl.x + cellW, tl.y + cellH);
                    Imgproc.rectangle(m, tl, br, fg, -1);
                }
            }
        }
    }

    private static void drawCentredText(Mat m, String text, Scalar fg,
                                        double fontScale, int thickness) {
        int font = Imgproc.FONT_HERSHEY_SIMPLEX;
        int[] baseline = {0};
        Size ts = Imgproc.getTextSize(text, font, fontScale, thickness, baseline);
        Point org = new Point(
            (SIZE - ts.width)  / 2.0,
            (SIZE + ts.height) / 2.0
        );
        Imgproc.putText(m, text, org, font, fontScale, fg, thickness, Imgproc.LINE_AA, false);
    }

    // -------------------------------------------------------------------------
    // Star helper
    // -------------------------------------------------------------------------

    /**
     * Draws a star with {@code points} tips.
     * Inner radius = 40% of outer radius.
     */
    private static void drawStar(Mat m, int points, Scalar fg, boolean filled) {
        double outerR = SIZE / 2.0 - 10;
        double innerR = outerR * 0.4;
        List<Point> pts = new ArrayList<>();
        for (int i = 0; i < points * 2; i++) {
            double angle = Math.toRadians(-90 + (180.0 / points) * i);
            double r = (i % 2 == 0) ? outerR : innerR;
            pts.add(new Point(SIZE / 2.0 + r * Math.cos(angle),
                              SIZE / 2.0 + r * Math.sin(angle)));
        }
        MatOfPoint mop = new MatOfPoint();
        mop.fromList(pts);
        if (filled) {
            Imgproc.fillPoly(m, List.of(mop), fg);
        } else {
            Imgproc.polylines(m, List.of(mop), true, fg, 2);
        }
    }

    // -------------------------------------------------------------------------
    // Polyline helpers
    // -------------------------------------------------------------------------

    /** Horizontal (isHorizontal=true) or vertical zigzag open polyline. */
    private static void drawZigzag(Mat m, Scalar fg, boolean isHorizontal) {
        int steps = 6;
        int amp   = 28;
        List<Point> pts = new ArrayList<>();
        if (isHorizontal) {
            int xStep = (SIZE - 20) / steps;
            for (int i = 0; i <= steps; i++) {
                int x = 10 + i * xStep;
                int y = (SIZE / 2) + (i % 2 == 0 ? -amp : amp);
                pts.add(new Point(x, y));
            }
        } else {
            int yStep = (SIZE - 20) / steps;
            for (int i = 0; i <= steps; i++) {
                int y = 10 + i * yStep;
                int x = (SIZE / 2) + (i % 2 == 0 ? -amp : amp);
                pts.add(new Point(x, y));
            }
        }
        MatOfPoint mop = new MatOfPoint();
        mop.fromList(pts);
        Imgproc.polylines(m, List.of(mop), false, fg, 2);
    }

    /** Sine-wave approximation as an open polyline. */
    private static void drawWave(Mat m, Scalar fg) {
        int steps = 32;
        int amp   = 30;
        List<Point> pts = new ArrayList<>();
        for (int i = 0; i <= steps; i++) {
            double t = (double) i / steps;
            int x = (int) (10 + t * (SIZE - 20));
            int y = (int) (SIZE / 2.0 - amp * Math.sin(t * 2 * Math.PI));
            pts.add(new Point(x, y));
        }
        MatOfPoint mop = new MatOfPoint();
        mop.fromList(pts);
        Imgproc.polylines(m, List.of(mop), false, fg, 2);
    }

    /** Inward Archimedean spiral as an open polyline. */
    private static void drawSpiral(Mat m, Scalar fg) {
        int steps  = 80;
        double turns = 2.5;
        List<Point> pts = new ArrayList<>();
        for (int i = 0; i <= steps; i++) {
            double t     = (double) i / steps;
            double angle = t * turns * 2 * Math.PI;
            double r     = (SIZE / 2.0 - 8) * (1.0 - t * 0.8);
            pts.add(new Point(SIZE / 2.0 + r * Math.cos(angle),
                              SIZE / 2.0 + r * Math.sin(angle)));
        }
        MatOfPoint mop = new MatOfPoint();
        mop.fromList(pts);
        Imgproc.polylines(m, List.of(mop), false, fg, 2);
    }

    /** Right or left-pointing arrow (closed polygon). */
    private static void drawArrow(Mat m, Scalar fg, boolean pointsRight) {
        int cx = SIZE / 2, cy = SIZE / 2;
        int hw = 45, hh = 20, headW = 28, headH = 36;
        List<Point> pts;
        if (pointsRight) {
            pts = List.of(
                new Point(cx - hw,       cy - hh),
                new Point(cx,            cy - hh),
                new Point(cx,            cy - headH),
                new Point(cx + hw,       cy),
                new Point(cx,            cy + headH),
                new Point(cx,            cy + hh),
                new Point(cx - hw,       cy + hh)
            );
        } else {
            pts = List.of(
                new Point(cx + hw,       cy - hh),
                new Point(cx,            cy - hh),
                new Point(cx,            cy - headH),
                new Point(cx - hw,       cy),
                new Point(cx,            cy + headH),
                new Point(cx,            cy + hh),
                new Point(cx + hw,       cy + hh)
            );
        }
        MatOfPoint mop = new MatOfPoint();
        mop.fromList(pts);
        Imgproc.polylines(m, List.of(mop), true, fg, 2);
    }

    /** L-shaped closed polygon. */
    private static void drawLShape(Mat m, Scalar fg) {
        List<Point> pts = List.of(
            new Point(20,        20),
            new Point(50,        20),
            new Point(50,        90),
            new Point(108,       90),
            new Point(108,       108),
            new Point(20,        108)
        );
        MatOfPoint mop = new MatOfPoint();
        mop.fromList(pts);
        Imgproc.polylines(m, List.of(mop), true, fg, 2);
    }

    /** T-shaped closed polygon. */
    private static void drawTShape(Mat m, Scalar fg) {
        List<Point> pts = List.of(
            new Point(16,   16),
            new Point(112,  16),
            new Point(112,  44),
            new Point(76,   44),
            new Point(76,   112),
            new Point(52,   112),
            new Point(52,   44),
            new Point(16,   44)
        );
        MatOfPoint mop = new MatOfPoint();
        mop.fromList(pts);
        Imgproc.polylines(m, List.of(mop), true, fg, 2);
    }

    /** Filled plus/cross shape. */
    private static void drawPlusShape(Mat m, Scalar fg) {
        int arm = 20, ctr = 44;
        List<Point> pts = List.of(
            new Point(ctr,        16),
            new Point(SIZE-ctr,   16),
            new Point(SIZE-ctr,   ctr),
            new Point(SIZE-16,    ctr),
            new Point(SIZE-16,    SIZE-ctr),
            new Point(SIZE-ctr,   SIZE-ctr),
            new Point(SIZE-ctr,   SIZE-16),
            new Point(ctr,        SIZE-16),
            new Point(ctr,        SIZE-ctr),
            new Point(16,         SIZE-ctr),
            new Point(16,         ctr),
            new Point(ctr,        ctr)
        );
        MatOfPoint mop = new MatOfPoint();
        mop.fromList(pts);
        Imgproc.fillPoly(m, List.of(mop), fg);
        Imgproc.polylines(m, List.of(mop), true, fg, 2);
    }

    /** Chevron/caret closed polyline. */
    private static void drawChevron(Mat m, Scalar fg) {
        int cx = SIZE / 2, cy = SIZE / 2;
        List<Point> pts = List.of(
            new Point(16,        cy),
            new Point(cx,        16),
            new Point(SIZE - 16, cy),
            new Point(SIZE - 16, SIZE - 16),
            new Point(cx,        cy + 20),
            new Point(16,        SIZE - 16)
        );
        MatOfPoint mop = new MatOfPoint();
        mop.fromList(pts);
        Imgproc.polylines(m, List.of(mop), true, fg, 2);
    }

    /** Diamond (square rotated 45°). */
    private static void drawDiamond(Mat m, Scalar fg) {
        int cx = SIZE / 2, cy = SIZE / 2, r = SIZE / 2 - 10;
        List<Point> pts = List.of(
            new Point(cx,     cy - r),
            new Point(cx + r, cy),
            new Point(cx,     cy + r),
            new Point(cx - r, cy)
        );
        MatOfPoint mop = new MatOfPoint();
        mop.fromList(pts);
        Imgproc.polylines(m, List.of(mop), true, fg, 2);
    }

    /** Parallelogram (horizontal offset on top edge). */
    private static void drawParallelogram(Mat m, Scalar fg) {
        int shear = 24;
        List<Point> pts = List.of(
            new Point(16 + shear, 24),
            new Point(SIZE - 16,  24),
            new Point(SIZE - 16 - shear, SIZE - 24),
            new Point(16,         SIZE - 24)
        );
        MatOfPoint mop = new MatOfPoint();
        mop.fromList(pts);
        Imgproc.polylines(m, List.of(mop), true, fg, 2);
    }

    // -------------------------------------------------------------------------
    // Concave & irregular polygon helpers
    // -------------------------------------------------------------------------

    /** Simple upward-pointing arrowhead with a concave notch cut into the base. */
    private static void drawArrowHead(Mat m, Scalar fg) {
        List<Point> pts = List.of(
            new Point(SIZE / 2.0, 10),          // tip
            new Point(SIZE - 14,  SIZE - 14),   // bottom-right
            new Point(SIZE / 2.0, SIZE - 36),   // concave notch
            new Point(14,         SIZE - 14)    // bottom-left
        );
        MatOfPoint mop = new MatOfPoint();
        mop.fromList(pts);
        Imgproc.fillPoly(m, List.of(mop), fg);
        Imgproc.polylines(m, List.of(mop), true, fg, 2);
    }

    /** Crescent / moon shape: large filled circle minus a smaller offset circle. */
    private static void drawMoon(Mat m, Scalar fg) {
        // Draw filled large circle
        Imgproc.circle(m, centre(), SIZE / 2 - 10, fg, -1);
        // Erase with background colour by drawing offset filled circle in black
        Imgproc.circle(m, new Point(SIZE * 0.62, SIZE / 2.0), SIZE / 2 - 14,
                new Scalar(0, 0, 0), -1);
    }

    /** Pac-Man: filled circle with a triangular wedge removed. */
    private static void drawPacMan(Mat m, Scalar fg) {
        // Full filled circle
        Imgproc.circle(m, centre(), SIZE / 2 - 10, fg, -1);
        // Erase wedge with black filled triangle (mouth)
        int cx = SIZE / 2, cy = SIZE / 2, r = SIZE / 2 - 6;
        List<Point> mouth = List.of(
            new Point(cx, cy),
            new Point(cx + r, cy - r / 3.0),
            new Point(cx + r, cy + r / 3.0)
        );
        MatOfPoint mop = new MatOfPoint();
        mop.fromList(mouth);
        Imgproc.fillPoly(m, List.of(mop), new Scalar(0, 0, 0));
    }

    /** Irregular quadrilateral — no right angles, no parallel sides. */
    private static void drawIrregularQuad(Mat m, Scalar fg) {
        List<Point> pts = List.of(
            new Point(18,  22),
            new Point(102, 14),
            new Point(112, 96),
            new Point(30,  110)
        );
        MatOfPoint mop = new MatOfPoint();
        mop.fromList(pts);
        Imgproc.polylines(m, List.of(mop), true, fg, 2);
    }

    /** Irregular asymmetric pentagon — deliberately unequal sides and angles. */
    private static void drawIrregularPenta(Mat m, Scalar fg) {
        List<Point> pts = List.of(
            new Point(50,  10),
            new Point(110, 30),
            new Point(115, 90),
            new Point(60,  115),
            new Point(12,  70)
        );
        MatOfPoint mop = new MatOfPoint();
        mop.fromList(pts);
        Imgproc.polylines(m, List.of(mop), true, fg, 2);
    }

    /** Irregular 5-pointed star with unequal tip lengths. */
    private static void drawIrregularStar(Mat m, Scalar fg) {
        // Tip radii vary: outer = 55, 42, 60, 38, 52; inner fixed at 20
        double[] outer = {55, 42, 60, 38, 52};
        double   inner = 20;
        List<Point> pts = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            double outerAngle = Math.toRadians(-90 + 72.0 * i);
            double innerAngle = Math.toRadians(-90 + 72.0 * i + 36);
            pts.add(new Point(SIZE / 2.0 + outer[i] * Math.cos(outerAngle),
                              SIZE / 2.0 + outer[i] * Math.sin(outerAngle)));
            pts.add(new Point(SIZE / 2.0 + inner    * Math.cos(innerAngle),
                              SIZE / 2.0 + inner    * Math.sin(innerAngle)));
        }
        MatOfPoint mop = new MatOfPoint();
        mop.fromList(pts);
        Imgproc.polylines(m, List.of(mop), true, fg, 2);
    }

    // -------------------------------------------------------------------------
    // Rotated rectangle helper
    // -------------------------------------------------------------------------

    /** Draws a rectangle outline rotated by {@code angleDeg} degrees around its centre. */
    private static void drawRotatedRect(Mat m, Scalar fg, double angleDeg) {
        // half-dimensions
        double hw = 48, hh = 28;
        double rad = Math.toRadians(angleDeg);
        double cos = Math.cos(rad), sin = Math.sin(rad);
        double cx = SIZE / 2.0, cy = SIZE / 2.0;
        double[][] corners = {{-hw, -hh}, {hw, -hh}, {hw, hh}, {-hw, hh}};
        List<Point> pts = new ArrayList<>();
        for (double[] c : corners) {
            pts.add(new Point(cx + c[0] * cos - c[1] * sin,
                              cy + c[0] * sin + c[1] * cos));
        }
        MatOfPoint mop = new MatOfPoint();
        mop.fromList(pts);
        Imgproc.polylines(m, List.of(mop), true, fg, 2);
    }

    // -------------------------------------------------------------------------
    // Dashed / dotted line helpers
    // -------------------------------------------------------------------------

    /**
     * Draws a dashed or dotted line from {@code p1} to {@code p2}.
     *
     * @param dashLen  length of each drawn segment in pixels
     * @param gapLen   length of each gap in pixels
     * @param fill     unused (reserved for future filled-dash variant)
     */
    private static void drawDashedLine(Mat m, Scalar fg,
                                       Point p1, Point p2,
                                       int dashLen, int gapLen,
                                       boolean fill) {
        double dx    = p2.x - p1.x;
        double dy    = p2.y - p1.y;
        double total = Math.sqrt(dx * dx + dy * dy);
        double ux    = dx / total;
        double uy    = dy / total;
        double pos   = 0;
        boolean drawing = true;
        while (pos < total) {
            double segLen = drawing ? dashLen : gapLen;
            double end    = Math.min(pos + segLen, total);
            if (drawing) {
                Point a = new Point(p1.x + ux * pos, p1.y + uy * pos);
                Point b = new Point(p1.x + ux * end, p1.y + uy * end);
                Imgproc.line(m, a, b, fg, 2);
            }
            pos    += segLen;
            drawing = !drawing;
        }
    }

    /**
     * Draws a dashed rectangle outline by applying {@link #drawDashedLine}
     * to each of the four edges.
     */
    private static void drawDashedRect(Mat m, Scalar fg,
                                       Point tl, Point br,
                                       int dashLen, int gapLen) {
        Point tr = new Point(br.x, tl.y);
        Point bl = new Point(tl.x, br.y);
        drawDashedLine(m, fg, tl, tr, dashLen, gapLen, false);
        drawDashedLine(m, fg, tr, br, dashLen, gapLen, false);
        drawDashedLine(m, fg, br, bl, dashLen, gapLen, false);
        drawDashedLine(m, fg, bl, tl, dashLen, gapLen, false);
    }

    // -------------------------------------------------------------------------
    // Compound shape helpers
    // -------------------------------------------------------------------------

    /** Inverted equilateral triangle (apex pointing down). */
    private static void drawInvertedTriangle(Mat m, Scalar fg) {
        int pad = 10;
        List<Point> pts = List.of(
            new Point(pad,        pad),
            new Point(SIZE - pad, pad),
            new Point(SIZE / 2.0, SIZE - pad)
        );
        MatOfPoint mop = new MatOfPoint();
        mop.fromList(pts);
        Imgproc.polylines(m, List.of(mop), true, fg, 2);
    }

    // -------------------------------------------------------------------------
    // Multi-colour shape drawing helpers (Milestone 21)
    // -------------------------------------------------------------------------

    /** Circle outline in RED, filled interior in GREEN. */
    private static void drawBicolourCircleRing(Mat m) {
        int r = SIZE / 2 - 12;
        // Filled centre in green
        Imgproc.circle(m, centre(), r - 4, MCOL_GREEN, -1);
        // Circle outline in red (on top)
        Imgproc.circle(m, centre(), r, MCOL_RED, 3);
    }

    /** Rectangle split horizontally: top half CYAN, bottom half MAGENTA. */
    private static void drawBicolourRectHalves(Mat m) {
        int pad = 14;
        int mid = SIZE / 2;
        // Top half — cyan filled
        Imgproc.rectangle(m,
                new Point(pad, pad),
                new Point(SIZE - pad, mid),
                MCOL_CYAN, -1);
        // Bottom half — magenta filled
        Imgproc.rectangle(m,
                new Point(pad, mid),
                new Point(SIZE - pad, SIZE - pad),
                MCOL_MAGENTA, -1);
    }

    /**
     * Equilateral triangle divided into three 120°-hue regions:
     * top vertex sector in RED, bottom-left in GREEN, bottom-right in BLUE.
     */
    private static void drawTricolourTriangle(Mat m) {
        int pad = 10;
        int cx  = SIZE / 2, cy = SIZE / 2;
        // Outer triangle vertices
        Point top = new Point(cx, pad);
        Point bl  = new Point(pad, SIZE - pad);
        Point br  = new Point(SIZE - pad, SIZE - pad);
        Point centreP = centre();
        // Three sub-triangles share the centroid
        fillTriangle(m, top, bl, centreP, MCOL_GREEN);
        fillTriangle(m, bl,  br, centreP, MCOL_BLUE);
        fillTriangle(m, br, top, centreP, MCOL_RED);
        // Outline in white to make edges visible
        MatOfPoint mop = new MatOfPoint(top, bl, br);
        Imgproc.polylines(m, List.of(mop), true, new Scalar(220, 220, 220), 1);
    }

    /**
     * Crosshair lines in YELLOW, surrounding circle ring in CYAN.
     *
     * <p>Drawing order matters:
     * <ol>
     *   <li>YELLOW crosshair is drawn first with arms that stop {@code r-6} pixels
     *       from the image centre — well short of the CYAN ring's inner edge.</li>
     *   <li>CYAN ring is drawn last so it is never broken by the crosshair lines.</li>
     * </ol>
     *
     * <p>This ensures the colour extractor sees:
     * <ul>
     *   <li>CYAN: <b>one</b> unbroken circular arc contour (type=CIRCLE, high circularity)</li>
     *   <li>YELLOW: <b>one</b> connected + polygon (type=CLOSED_CONCAVE_POLY, ~12 vertices)</li>
     * </ul>
     * Both produce stable, scale-invariant {@link org.example.matchers.VectorSignature}
     * descriptors that match reliably at any scene scale.
     */
    private static void drawBicolourCrosshairRing(Mat m) {
        int r   = SIZE / 2 - 10;  // ring radius = 54 for SIZE=128
        int arm = r - 6;           // half-arm length = 48; gap to ring inner edge ≥ 4 px

        // Draw crosshair FIRST — arms stay strictly inside the ring
        Imgproc.line(m, new Point(SIZE / 2 - arm, SIZE / 2),
                        new Point(SIZE / 2 + arm, SIZE / 2), MCOL_YELLOW, 3);
        Imgproc.line(m, new Point(SIZE / 2, SIZE / 2 - arm),
                        new Point(SIZE / 2, SIZE / 2 + arm), MCOL_YELLOW, 3);

        // Draw ring LAST — paints over any stray pixels, stays one unbroken circle
        Imgproc.circle(m, centre(), r, MCOL_CYAN, 5);
    }

    /** Chevron outline in ORANGE, interior filled in MAGENTA. */
    private static void drawBicolourChevronFilled(Mat m) {
        int pad = 14, mid = SIZE / 2, tip = SIZE - pad;
        List<Point> pts = List.of(
                new Point(pad,       pad),
                new Point(mid,       mid - 8),
                new Point(SIZE - pad,pad),
                new Point(SIZE - pad,pad + 20),
                new Point(mid,       mid + 12),
                new Point(pad,       pad + 20)
        );
        MatOfPoint mop = new MatOfPoint();
        mop.fromList(pts);
        // Fill interior with magenta
        Imgproc.fillPoly(m, List.of(mop), MCOL_MAGENTA);
        // Outline in orange
        Imgproc.polylines(m, List.of(mop), true, MCOL_ORANGE, 2);
    }

    /** Fills a triangle defined by three Points with the given colour. */
    private static void fillTriangle(Mat m, Point a, Point b, Point c, Scalar colour) {
        MatOfPoint mop = new MatOfPoint(a, b, c);
        Imgproc.fillPoly(m, List.of(mop), colour);
    }

    // -------------------------------------------------------------------------
    // Utility
    // -------------------------------------------------------------------------

    private static Point centre() {
        return new Point(SIZE / 2.0, SIZE / 2.0);
    }

    private static Scalar blend(Scalar a, Scalar b, double t) {
        return new Scalar(
            blend1(a.val[0], b.val[0], t),
            blend1(a.val[1], b.val[1], t),
            blend1(a.val[2], b.val[2], t)
        );
    }

    private static double blend1(double a, double b, double t) {
        return Math.max(0, Math.min(255, a + (b - a) * t));
    }
}






