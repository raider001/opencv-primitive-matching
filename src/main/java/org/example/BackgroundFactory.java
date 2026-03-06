package org.example;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.EnumMap;
import java.util.Map;
import java.util.Random;

/**
 * Generates synthetic scene background {@link Mat} images for a given {@link BackgroundId}.
 *
 * <p>All backgrounds are produced in full BGR colour at the requested width × height.
 * Results are cached by (id, width, height) so repeated calls within the same JVM are free.
 *
 * <p>Every random background uses a fixed seed so output is deterministic across runs.
 */
public final class BackgroundFactory {

    /** Shared cache: rebuilt per unique size. The test suite only uses one size (640×480). */
    private static final Map<BackgroundId, Mat> CACHE = new EnumMap<>(BackgroundId.class);
    private static int cachedW = -1;
    private static int cachedH = -1;

    private BackgroundFactory() {}

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------

    /**
     * Returns a BGR {@link Mat} of the requested size for the given background ID.
     * The result is cached — callers must NOT release it; clone if mutation is needed.
     */
    public static synchronized Mat get(BackgroundId id, int w, int h) {
        if (w != cachedW || h != cachedH) {
            CACHE.clear();
            cachedW = w;
            cachedH = h;
        }
        return CACHE.computeIfAbsent(id, k -> build(k, w, h));
    }

    /** Builds a fresh (uncached) copy — use when the caller needs to modify the Mat. */
    public static Mat build(BackgroundId id, int w, int h) {
        return switch (id) {
            // Tier 1 — solid fills
            case BG_SOLID_BLACK  -> solid(w, h, 0,   0,   0);
            case BG_SOLID_WHITE  -> solid(w, h, 255, 255, 255);
            case BG_SOLID_GREY   -> solid(w, h, 128, 128, 128);

            // Tier 2 — gradients & light noise
            case BG_GRADIENT_H_GREY      -> gradientH(w, h,
                    new Scalar(0, 0, 0), new Scalar(255, 255, 255));
            case BG_GRADIENT_V_GREY      -> gradientV(w, h,
                    new Scalar(0, 0, 0), new Scalar(255, 255, 255));
            case BG_GRADIENT_H_COLOUR    -> gradientH(w, h,
                    new Scalar(150, 0, 0), new Scalar(0, 0, 150));   // deep blue → deep red
            case BG_GRADIENT_V_COLOUR    -> gradientV(w, h,
                    new Scalar(0, 100, 0), new Scalar(80, 0, 80));   // dark green → dark purple
            case BG_GRADIENT_RADIAL_GREY -> gradientRadial(w, h,
                    new Scalar(220, 220, 220), new Scalar(20, 20, 20));
            case BG_GRADIENT_RADIAL_COLOUR -> gradientRadial(w, h,
                    new Scalar(0, 140, 255), new Scalar(128, 128, 0)); // orange centre → teal edge
            case BG_NOISE_LIGHT  -> noise(w, h, 128, 8,  42L);

            // Tier 3 — structured colour
            case BG_NOISE_HEAVY    -> noise(w, h, 100, 35, 43L);
            case BG_GRID_FINE      -> grid(w, h, 16, new Scalar(0, 150, 150), new Scalar(0, 0, 0));
            case BG_GRID_COARSE    -> grid(w, h, 64, new Scalar(120, 0, 0),  new Scalar(40, 40, 40));
            case BG_HATCHING       -> hatching(w, h, new Scalar(0, 120, 0),  new Scalar(40, 40, 40));
            case BG_RANDOM_LINES   -> randomLines(w, h, 30, 44L);
            case BG_RANDOM_CIRCLES -> randomCircles(w, h, 15, 45L);

            // Tier 4 — rich colour clutter
            case BG_RANDOM_MIXED   -> randomMixed(w, h, 40, 46L);
            case BG_DENSE_TEXT     -> denseText(w, h, 47L);
            case BG_CIRCUIT_LIKE   -> circuitLike(w, h, 48L);
            case BG_ORGANIC        -> organic(w, h, 49L);
            case BG_COLOURED_NOISE -> colouredNoise(w, h, 50L);
        };
    }

    // -------------------------------------------------------------------------
    // Tier 1 — solid
    // -------------------------------------------------------------------------

    private static Mat solid(int w, int h, int b, int g, int r) {
        return new Mat(h, w, CvType.CV_8UC3, new Scalar(b, g, r));
    }

    // -------------------------------------------------------------------------
    // Tier 2 — gradients
    // -------------------------------------------------------------------------

    private static Mat gradientH(int w, int h, Scalar left, Scalar right) {
        Mat m = Mat.zeros(h, w, CvType.CV_8UC3);
        byte[] row = new byte[w * 3];
        for (int x = 0; x < w; x++) {
            double t = (double) x / Math.max(1, w - 1);
            row[x * 3]     = (byte)(int) blend1(left.val[0], right.val[0], t);
            row[x * 3 + 1] = (byte)(int) blend1(left.val[1], right.val[1], t);
            row[x * 3 + 2] = (byte)(int) blend1(left.val[2], right.val[2], t);
        }
        for (int y = 0; y < h; y++) {
            m.put(y, 0, row);
        }
        return m;
    }

    private static Mat gradientV(int w, int h, Scalar top, Scalar bot) {
        Mat m = Mat.zeros(h, w, CvType.CV_8UC3);
        byte[] row = new byte[w * 3];
        for (int y = 0; y < h; y++) {
            double t = (double) y / Math.max(1, h - 1);
            byte b = (byte)(int) blend1(top.val[0], bot.val[0], t);
            byte g = (byte)(int) blend1(top.val[1], bot.val[1], t);
            byte r = (byte)(int) blend1(top.val[2], bot.val[2], t);
            for (int x = 0; x < w; x++) {
                row[x * 3]     = b;
                row[x * 3 + 1] = g;
                row[x * 3 + 2] = r;
            }
            m.put(y, 0, row);
        }
        return m;
    }

    private static Mat gradientRadial(int w, int h, Scalar centre, Scalar edge) {
        Mat m = Mat.zeros(h, w, CvType.CV_8UC3);
        double maxDist = Math.sqrt(w * w + h * h) / 2.0;
        double cx = w / 2.0, cy = h / 2.0;
        byte[] row = new byte[w * 3];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double dist = Math.sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
                double t = Math.min(dist / maxDist, 1.0);
                row[x * 3]     = (byte)(int) blend1(centre.val[0], edge.val[0], t);
                row[x * 3 + 1] = (byte)(int) blend1(centre.val[1], edge.val[1], t);
                row[x * 3 + 2] = (byte)(int) blend1(centre.val[2], edge.val[2], t);
            }
            m.put(y, 0, row);
        }
        return m;
    }

    // -------------------------------------------------------------------------
    // Noise helpers
    // -------------------------------------------------------------------------

    private static Mat noise(int w, int h, int baseLuma, int sigma, long seed) {
        Mat m = new Mat(h, w, CvType.CV_8UC3, new Scalar(baseLuma, baseLuma, baseLuma));
        Mat n = new Mat(h, w, CvType.CV_8SC3);
        // Deterministic noise: fill with seeded random values
        Random rng = new Random(seed);
        byte[] data = new byte[(int)(n.total() * n.channels())];
        rng.nextBytes(data);
        // Scale bytes to ±sigma range
        for (int i = 0; i < data.length; i++) {
            data[i] = (byte) ((data[i] * sigma) >> 7);
        }
        n.put(0, 0, data);
        Mat mFloat = new Mat();
        m.convertTo(mFloat, CvType.CV_16SC3);
        Mat nFloat = new Mat();
        n.convertTo(nFloat, CvType.CV_16SC3);
        Core.add(mFloat, nFloat, mFloat);
        mFloat.convertTo(m, CvType.CV_8UC3);
        mFloat.release();
        nFloat.release();
        n.release();
        return m;
    }

    private static Mat colouredNoise(int w, int h, long seed) {
        Mat m = Mat.zeros(h, w, CvType.CV_8UC3);
        Random rng = new Random(seed);
        byte[] data = new byte[(int)(m.total() * m.channels())];
        rng.nextBytes(data);
        m.put(0, 0, data);
        return m;
    }

    // -------------------------------------------------------------------------
    // Tier 3 — structured patterns
    // -------------------------------------------------------------------------

    private static Mat grid(int w, int h, int step, Scalar lineColour, Scalar bgColour) {
        Mat m = new Mat(h, w, CvType.CV_8UC3, bgColour);
        for (int x = 0; x < w; x += step)
            Imgproc.line(m, new Point(x, 0), new Point(x, h - 1), lineColour, 1);
        for (int y = 0; y < h; y += step)
            Imgproc.line(m, new Point(0, y), new Point(w - 1, y), lineColour, 1);
        return m;
    }

    private static Mat hatching(int w, int h, Scalar lineColour, Scalar bgColour) {
        Mat m = new Mat(h, w, CvType.CV_8UC3, bgColour);
        int step = 16;
        // Diagonal lines at 45° — iterate across top edge and left edge
        for (int start = -(h); start < w; start += step) {
            Point p1 = new Point(start, 0);
            Point p2 = new Point(start + h, h);
            Imgproc.line(m, p1, p2, lineColour, 1);
        }
        return m;
    }

    private static Mat randomLines(int w, int h, int count, long seed) {
        Mat m = new Mat(h, w, CvType.CV_8UC3, new Scalar(20, 20, 20));
        Random rng = new Random(seed);
        for (int i = 0; i < count; i++) {
            Point p1 = new Point(rng.nextInt(w), rng.nextInt(h));
            Point p2 = new Point(rng.nextInt(w), rng.nextInt(h));
            Scalar col = randomColour(rng);
            Imgproc.line(m, p1, p2, col, 1 + rng.nextInt(2));
        }
        return m;
    }

    private static Mat randomCircles(int w, int h, int count, long seed) {
        Mat m = new Mat(h, w, CvType.CV_8UC3, new Scalar(20, 20, 20));
        Random rng = new Random(seed);
        for (int i = 0; i < count; i++) {
            Point centre = new Point(rng.nextInt(w), rng.nextInt(h));
            int radius = 10 + rng.nextInt(Math.min(w, h) / 4);
            Scalar col = randomColour(rng);
            Imgproc.circle(m, centre, radius, col, 1 + rng.nextInt(2));
        }
        return m;
    }

    // -------------------------------------------------------------------------
    // Tier 4 — high complexity
    // -------------------------------------------------------------------------

    private static Mat randomMixed(int w, int h, int count, long seed) {
        Mat m = new Mat(h, w, CvType.CV_8UC3, new Scalar(15, 15, 15));
        Random rng = new Random(seed);
        for (int i = 0; i < count; i++) {
            Scalar col = randomColour(rng);
            int type = rng.nextInt(3);
            switch (type) {
                case 0 -> { // line
                    Imgproc.line(m,
                        new Point(rng.nextInt(w), rng.nextInt(h)),
                        new Point(rng.nextInt(w), rng.nextInt(h)),
                        col, 1 + rng.nextInt(2));
                }
                case 1 -> { // circle outline
                    Imgproc.circle(m,
                        new Point(rng.nextInt(w), rng.nextInt(h)),
                        8 + rng.nextInt(40), col, 1 + rng.nextInt(2));
                }
                case 2 -> { // rectangle outline
                    int x1 = rng.nextInt(w), y1 = rng.nextInt(h);
                    int x2 = rng.nextInt(w), y2 = rng.nextInt(h);
                    Imgproc.rectangle(m,
                        new Point(Math.min(x1, x2), Math.min(y1, y2)),
                        new Point(Math.max(x1, x2), Math.max(y1, y2)),
                        col, 1 + rng.nextInt(2));
                }
            }
        }
        return m;
    }

    private static Mat denseText(int w, int h, long seed) {
        Mat m = new Mat(h, w, CvType.CV_8UC3, new Scalar(15, 15, 15));
        Random rng = new Random(seed);
        String chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%";
        int fontSize = 12;
        int colStep  = fontSize + 2;
        int rowStep  = fontSize + 6;
        for (int y = rowStep; y < h; y += rowStep) {
            for (int x = 2; x < w; x += colStep) {
                char c = chars.charAt(rng.nextInt(chars.length()));
                Imgproc.putText(m, String.valueOf(c), new Point(x, y),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.35, randomColour(rng), 1);
            }
        }
        return m;
    }

    private static Mat circuitLike(int w, int h, long seed) {
        Mat m = new Mat(h, w, CvType.CV_8UC3, new Scalar(20, 30, 10)); // dark brown
        Random rng = new Random(seed);
        Scalar traceGreen = new Scalar(0, 180, 0);
        Scalar padGold    = new Scalar(0, 180, 220);

        // Horizontal and vertical trace segments on a grid
        int gridStep = 20;
        for (int y = gridStep; y < h; y += gridStep) {
            for (int x = gridStep; x < w; x += gridStep) {
                if (rng.nextFloat() > 0.4f) {
                    int len = gridStep * (1 + rng.nextInt(4));
                    boolean horiz = rng.nextBoolean();
                    Point p1 = new Point(x, y);
                    Point p2 = horiz ? new Point(Math.min(x + len, w - 1), y)
                                     : new Point(x, Math.min(y + len, h - 1));
                    Imgproc.line(m, p1, p2, traceGreen, 2);
                }
                // Pads at grid intersections
                if (rng.nextFloat() > 0.65f) {
                    Imgproc.circle(m, new Point(x, y), 3, padGold, -1);
                }
            }
        }
        return m;
    }

    private static Mat organic(int w, int h, long seed) {
        Mat m = new Mat(h, w, CvType.CV_8UC3, new Scalar(20, 20, 40));
        Random rng = new Random(seed);

        // Warm-toned blobs
        for (int i = 0; i < 12; i++) {
            Point centre = new Point(rng.nextInt(w), rng.nextInt(h));
            Size axes = new Size(20 + rng.nextInt(60), 15 + rng.nextInt(40));
            double angle = rng.nextDouble() * 180;
            // Warm colours: reds, oranges, yellows in BGR
            Scalar warm = new Scalar(
                rng.nextInt(60),
                80 + rng.nextInt(120),
                140 + rng.nextInt(115)
            );
            Imgproc.ellipse(m, centre, axes, angle, 0, 360, warm, -1);
        }
        // Cool-toned curved lines over the top
        for (int i = 0; i < 10; i++) {
            Point centre = new Point(rng.nextInt(w), rng.nextInt(h));
            Size axes = new Size(15 + rng.nextInt(50), 10 + rng.nextInt(35));
            double angle = rng.nextDouble() * 180;
            double startA = rng.nextDouble() * 180;
            // Cool colours: blues, teals, purples in BGR
            Scalar cool = new Scalar(
                120 + rng.nextInt(135),
                60 + rng.nextInt(80),
                rng.nextInt(60)
            );
            Imgproc.ellipse(m, centre, axes, angle, startA, startA + 200, cool, 2);
        }
        return m;
    }

    // -------------------------------------------------------------------------
    // Utility
    // -------------------------------------------------------------------------

    private static Scalar randomColour(Random rng) {
        // Avoid very dark colours (< 60 per channel) so shapes are visible
        return new Scalar(
            60 + rng.nextInt(196),
            60 + rng.nextInt(196),
            60 + rng.nextInt(196)
        );
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





