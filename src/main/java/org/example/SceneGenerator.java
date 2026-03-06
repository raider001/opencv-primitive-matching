package org.example;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Generates all four scene categories for the {@link SceneCatalogue}.
 *
 * <p>Every scene is 640×480 BGR colour. Each non-negative scene records a
 * {@link SceneShapePlacement} that describes exactly where the reference shape landed
 * and what transforms were applied — this is the ground truth used by the HTML report.
 */
public final class SceneGenerator {

    // Scene canvas dimensions
    public static final int W = 640;
    public static final int H = 480;

    // Reference thumbnail size (as built by ReferenceImageFactory)
    private static final int REF_SIZE = 128;

    private SceneGenerator() {}

    // -------------------------------------------------------------------------
    // Public entry points
    // -------------------------------------------------------------------------

    /** Generates all Category A scenes (clean, centred, no transforms). */
    public static List<SceneEntry> buildCategoryA() {
        BackgroundId[] backgrounds = {
            BackgroundId.BG_SOLID_BLACK,
            BackgroundId.BG_GRADIENT_H_COLOUR,
            BackgroundId.BG_NOISE_LIGHT,
            BackgroundId.BG_RANDOM_MIXED
        };
        List<SceneEntry> scenes = new ArrayList<>();
        for (ReferenceId ref : ReferenceId.values()) {
            Mat refMat = ReferenceImageFactory.build(ref);
            for (BackgroundId bg : backgrounds) {
                Mat canvas = BackgroundFactory.build(bg, W, H);
                Rect placed = placeAtCentre(canvas, refMat, 1.0, 0.0, 0, 0);
                SceneShapePlacement meta = SceneShapePlacement.clean(ref, placed);
                scenes.add(new SceneEntry(ref, SceneCategory.A_CLEAN,
                        "clean_" + bg.name().toLowerCase(), bg,
                        List.of(meta), canvas));
            }
            refMat.release();
        }
        return scenes;
    }

    /** Generates all Category B scenes (scaled / rotated / repositioned). */
    public static List<SceneEntry> buildCategoryB() {
        record Variant(String label, double scale, double rot, int offX, int offY) {}
        List<Variant> variants = List.of(
            new Variant("scale_0.50",       0.50, 0,   0,   0  ),
            new Variant("scale_0.75",       0.75, 0,   0,   0  ),
            new Variant("scale_1.25",       1.25, 0,   0,   0  ),
            new Variant("scale_1.50",       1.50, 0,   0,   0  ),
            new Variant("scale_2.00",       2.00, 0,   0,   0  ),
            new Variant("rot_15",           1.0,  15,  0,   0  ),
            new Variant("rot_30",           1.0,  30,  0,   0  ),
            new Variant("rot_45",           1.0,  45,  0,   0  ),
            new Variant("rot_90",           1.0,  90,  0,   0  ),
            new Variant("rot_180",          1.0,  180, 0,   0  ),
            new Variant("scale1.5_rot45",   1.50, 45,  0,   0  ),
            new Variant("scale0.75_rot30",  0.75, 30,  0,   0  ),
            new Variant("offset_topleft",   1.0,  0,  -120, -90),
            new Variant("offset_botright",  1.0,  0,   120,  90),
            new Variant("offset_random42",  1.0,  0,  -60,  40 )
        );
        BackgroundId[] backgrounds = {
            BackgroundId.BG_RANDOM_MIXED,
            BackgroundId.BG_CIRCUIT_LIKE,
            BackgroundId.BG_GRADIENT_RADIAL_COLOUR
        };
        List<SceneEntry> scenes = new ArrayList<>();
        int vi = 0;
        for (ReferenceId ref : ReferenceId.values()) {
            Mat refMat = ReferenceImageFactory.build(ref);
            for (Variant v : variants) {
                BackgroundId bg = backgrounds[vi % backgrounds.length];
                vi++;
                Mat canvas = BackgroundFactory.build(bg, W, H);
                Rect placed = placeAtCentre(canvas, refMat, v.scale(), v.rot(), v.offX(), v.offY());
                SceneShapePlacement meta = SceneShapePlacement.transformed(
                        ref, placed, v.scale(), v.rot(), v.offX(), v.offY());
                scenes.add(new SceneEntry(ref, SceneCategory.B_TRANSFORMED,
                        v.label(), bg, List.of(meta), canvas));
            }
            refMat.release();
        }
        return scenes;
    }

    /** Generates all Category C scenes (reference present but degraded). */
    public static List<SceneEntry> buildCategoryC() {
        BackgroundId[] backgrounds = {
            BackgroundId.BG_DENSE_TEXT,
            BackgroundId.BG_CIRCUIT_LIKE,
            BackgroundId.BG_ORGANIC,
            BackgroundId.BG_RANDOM_MIXED,
            BackgroundId.BG_COLOURED_NOISE
        };
        List<SceneEntry> scenes = new ArrayList<>();
        int vi = 0;
        for (ReferenceId ref : ReferenceId.values()) {
            Mat refMat = ReferenceImageFactory.build(ref);

            // --- Variant 1: Gaussian noise σ=10 on full scene
            {
                BackgroundId bg = backgrounds[vi++ % backgrounds.length];
                Mat canvas = BackgroundFactory.build(bg, W, H);
                Rect placed = placeAtCentre(canvas, refMat, 1.0, 0, 0, 0);
                addGaussianNoise(canvas, 10, 51L + ref.ordinal());
                scenes.add(scene(ref, SceneCategory.C_DEGRADED, "noise_s10", bg,
                        SceneShapePlacement.clean(ref, placed), canvas));
            }
            // --- Variant 2: Gaussian noise σ=25 on full scene
            {
                BackgroundId bg = backgrounds[vi++ % backgrounds.length];
                Mat canvas = BackgroundFactory.build(bg, W, H);
                Rect placed = placeAtCentre(canvas, refMat, 1.0, 0, 0, 0);
                addGaussianNoise(canvas, 25, 52L + ref.ordinal());
                scenes.add(scene(ref, SceneCategory.C_DEGRADED, "noise_s25", bg,
                        SceneShapePlacement.clean(ref, placed), canvas));
            }
            // --- Variant 3: Reference at 40% contrast
            {
                BackgroundId bg = backgrounds[vi++ % backgrounds.length];
                Mat canvas = BackgroundFactory.build(bg, W, H);
                Mat fadedRef = applyContrast(refMat, 0.40);
                Rect placed = placeAtCentre(canvas, fadedRef, 1.0, 0, 0, 0);
                fadedRef.release();
                scenes.add(scene(ref, SceneCategory.C_DEGRADED, "contrast_40pct", bg,
                        SceneShapePlacement.clean(ref, placed), canvas));
            }
            // --- Variant 4: 25% occlusion (top-left quarter of shape masked)
            {
                BackgroundId bg = backgrounds[vi++ % backgrounds.length];
                Mat canvas = BackgroundFactory.build(bg, W, H);
                Rect placed = placeAtCentre(canvas, refMat, 1.0, 0, 0, 0);
                occlude(canvas, placed, 0.25);
                SceneShapePlacement meta = new SceneShapePlacement(ref, placed,
                        1.0, 0, 0, 0, false, true, 0.25);
                scenes.add(new SceneEntry(ref, SceneCategory.C_DEGRADED, "occ_25pct", bg,
                        List.of(meta), canvas));
            }
            // --- Variant 5: 50% occlusion (left half of shape masked)
            {
                BackgroundId bg = backgrounds[vi++ % backgrounds.length];
                Mat canvas = BackgroundFactory.build(bg, W, H);
                Rect placed = placeAtCentre(canvas, refMat, 1.0, 0, 0, 0);
                occlude(canvas, placed, 0.50);
                SceneShapePlacement meta = new SceneShapePlacement(ref, placed,
                        1.0, 0, 0, 0, false, true, 0.50);
                scenes.add(new SceneEntry(ref, SceneCategory.C_DEGRADED, "occ_50pct", bg,
                        List.of(meta), canvas));
            }
            // --- Variant 6: Reference blurred with Gaussian 5×5
            {
                BackgroundId bg = backgrounds[vi++ % backgrounds.length];
                Mat canvas = BackgroundFactory.build(bg, W, H);
                Mat blurredRef = new Mat();
                Imgproc.GaussianBlur(refMat, blurredRef, new Size(5, 5), 0);
                Rect placed = placeAtCentre(canvas, blurredRef, 1.0, 0, 0, 0);
                blurredRef.release();
                scenes.add(scene(ref, SceneCategory.C_DEGRADED, "blur_5x5", bg,
                        SceneShapePlacement.clean(ref, placed), canvas));
            }
            // --- Variant 7: Colour shift +40° hue on reference
            {
                BackgroundId bg = backgrounds[vi++ % backgrounds.length];
                Mat canvas = BackgroundFactory.build(bg, W, H);
                Mat shiftedRef = hueShift(refMat, 40);
                Rect placed = placeAtCentre(canvas, shiftedRef, 1.0, 0, 0, 0);
                shiftedRef.release();
                SceneShapePlacement meta = new SceneShapePlacement(ref, placed,
                        1.0, 0, 0, 0, true, false, 0.0);
                scenes.add(new SceneEntry(ref, SceneCategory.C_DEGRADED, "hue_shift_40", bg,
                        List.of(meta), canvas));
            }

            refMat.release();
        }
        return scenes;
    }

    /** Generates all Category D scenes (no reference — false-positive tests). */
    public static List<SceneEntry> buildCategoryD() {
        List<SceneEntry> scenes = new ArrayList<>();

        // One scene per background ID (21 scenes)
        for (BackgroundId bg : BackgroundId.values()) {
            Mat canvas = BackgroundFactory.build(bg, W, H);
            scenes.add(new SceneEntry(null, SceneCategory.D_NEGATIVE,
                    "negative_" + bg.name().toLowerCase(), bg,
                    List.of(), canvas));
        }

        // 5 heavy colour noise scenes with different seeds
        for (int s = 0; s < 5; s++) {
            Mat canvas = Mat.zeros(H, W, CvType.CV_8UC3);
            Random rng = new Random(100L + s);
            byte[] data = new byte[H * W * 3];
            rng.nextBytes(data);
            canvas.put(0, 0, data);
            scenes.add(new SceneEntry(null, SceneCategory.D_NEGATIVE,
                    "negative_colour_noise_" + s, BackgroundId.BG_COLOURED_NOISE,
                    List.of(), canvas));
        }

        // 10 dense random coloured shapes
        for (int s = 0; s < 10; s++) {
            Mat canvas = BackgroundFactory.build(BackgroundId.BG_RANDOM_MIXED, W, H);
            Random rng = new Random(200L + s);
            for (int i = 0; i < 50; i++) {
                Scalar col = new Scalar(60 + rng.nextInt(196), 60 + rng.nextInt(196), 60 + rng.nextInt(196));
                switch (rng.nextInt(3)) {
                    case 0 -> Imgproc.circle(canvas,
                            new Point(rng.nextInt(W), rng.nextInt(H)),
                            10 + rng.nextInt(40), col, 2);
                    case 1 -> Imgproc.rectangle(canvas,
                            new Point(rng.nextInt(W - 60), rng.nextInt(H - 60)),
                            new Point(rng.nextInt(W), rng.nextInt(H)), col, 2);
                    case 2 -> Imgproc.line(canvas,
                            new Point(rng.nextInt(W), rng.nextInt(H)),
                            new Point(rng.nextInt(W), rng.nextInt(H)), col, 2);
                }
            }
            scenes.add(new SceneEntry(null, SceneCategory.D_NEGATIVE,
                    "negative_random_shapes_" + s, BackgroundId.BG_RANDOM_MIXED,
                    List.of(), canvas));
        }

        // 5 circuit-like decoy scenes
        for (int s = 0; s < 5; s++) {
            Mat canvas = BackgroundFactory.build(BackgroundId.BG_CIRCUIT_LIKE, W, H);
            Random rng = new Random(300L + s);
            // Add some extra coloured decoy shapes that look shape-like but are not references
            for (int i = 0; i < 8; i++) {
                Scalar col = new Scalar(60 + rng.nextInt(196), 60 + rng.nextInt(196), 60 + rng.nextInt(196));
                int cx = 40 + rng.nextInt(W - 80);
                int cy = 40 + rng.nextInt(H - 80);
                int r  = 15 + rng.nextInt(25);
                Imgproc.circle(canvas, new Point(cx, cy), r, col, 2);
            }
            scenes.add(new SceneEntry(null, SceneCategory.D_NEGATIVE,
                    "negative_circuit_decoy_" + s, BackgroundId.BG_CIRCUIT_LIKE,
                    List.of(), canvas));
        }

        // 3 dense coloured text scenes at different sizes
        int[] fontScales = {8, 16, 24};
        for (int fs : fontScales) {
            Mat canvas = BackgroundFactory.build(BackgroundId.BG_DENSE_TEXT, W, H);
            scenes.add(new SceneEntry(null, SceneCategory.D_NEGATIVE,
                    "negative_dense_text_sz" + fs, BackgroundId.BG_DENSE_TEXT,
                    List.of(), canvas));
        }

        return scenes;
    }

    // -------------------------------------------------------------------------
    // Core placement helper
    // -------------------------------------------------------------------------

    /**
     * Scales and rotates {@code refMat}, composites it onto {@code canvas} at
     * (centreX + offX, centreY + offY), and returns the bounding {@link Rect}
     * of the placed patch in scene coordinates.
     */
    static Rect placeAtCentre(Mat canvas, Mat refMat,
                              double scale, double rotDeg,
                              int offX, int offY) {
        // 1. Scale
        Mat scaled = new Mat();
        if (scale != 1.0) {
            int newW = Math.max(1, (int) Math.round(refMat.cols() * scale));
            int newH = Math.max(1, (int) Math.round(refMat.rows() * scale));
            Imgproc.resize(refMat, scaled, new Size(newW, newH), 0, 0,
                    scale < 1.0 ? Imgproc.INTER_AREA : Imgproc.INTER_LINEAR);
        } else {
            scaled = refMat.clone();
        }

        // 2. Rotate (about the patch centre)
        Mat rotated = new Mat();
        if (rotDeg != 0.0) {
            Point patchCentre = new Point(scaled.cols() / 2.0, scaled.rows() / 2.0);
            Mat rot = Imgproc.getRotationMatrix2D(patchCentre, -rotDeg, 1.0);
            // Compute new bounding size after rotation
            double rad = Math.toRadians(rotDeg);
            double cos = Math.abs(Math.cos(rad)), sin = Math.abs(Math.sin(rad));
            int newW = (int) Math.ceil(scaled.cols() * cos + scaled.rows() * sin);
            int newH = (int) Math.ceil(scaled.cols() * sin + scaled.rows() * cos);
            // Shift the rotation matrix to account for the new size
            double[] t = rot.get(0, 2);
            rot.put(0, 2, new double[]{ t[0] + (newW - scaled.cols()) / 2.0 });
            t = rot.get(1, 2);
            rot.put(1, 2, new double[]{ t[0] + (newH - scaled.rows()) / 2.0 });
            Imgproc.warpAffine(scaled, rotated, rot, new Size(newW, newH),
                    Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, new Scalar(0, 0, 0));
            rot.release();
        } else {
            rotated = scaled.clone();
        }
        scaled.release();

        int pw = rotated.cols();
        int ph = rotated.rows();

        // 3. Compute top-left corner in scene coordinates
        int cx = canvas.cols() / 2 + offX;
        int cy = canvas.rows() / 2 + offY;
        int x = cx - pw / 2;
        int y = cy - ph / 2;

        // 4. Clip to canvas bounds
        int srcX = Math.max(0, -x);
        int srcY = Math.max(0, -y);
        int dstX = Math.max(0, x);
        int dstY = Math.max(0, y);
        int copyW = Math.min(pw - srcX, canvas.cols() - dstX);
        int copyH = Math.min(ph - srcY, canvas.rows() - dstY);

        if (copyW > 0 && copyH > 0) {
            Mat src = rotated.submat(new Rect(srcX, srcY, copyW, copyH));
            Mat dst = canvas.submat(new Rect(dstX, dstY, copyW, copyH));
            // Alpha-composite: non-black pixels from patch overwrite background
            Mat mask = new Mat();
            Mat grey = new Mat();
            Imgproc.cvtColor(src, grey, Imgproc.COLOR_BGR2GRAY);
            Imgproc.threshold(grey, mask, 5, 255, Imgproc.THRESH_BINARY);
            src.copyTo(dst, mask);
            grey.release();
            mask.release();
            src.release();
            dst.release();
        }
        rotated.release();

        // Return the bounding rect (clipped to canvas)
        return new Rect(
                Math.max(0, x),
                Math.max(0, y),
                Math.min(pw, canvas.cols() - Math.max(0, x)),
                Math.min(ph, canvas.rows() - Math.max(0, y))
        );
    }

    // -------------------------------------------------------------------------
    // Degradation helpers
    // -------------------------------------------------------------------------

    private static void addGaussianNoise(Mat m, int sigma, long seed) {
        Random rng = new Random(seed);
        byte[] data = new byte[(int)(m.total() * m.channels())];
        m.get(0, 0, data);
        for (int i = 0; i < data.length; i++) {
            int v = (data[i] & 0xFF) + (int)(rng.nextGaussian() * sigma);
            data[i] = (byte) Math.max(0, Math.min(255, v));
        }
        m.put(0, 0, data);
    }

    private static Mat applyContrast(Mat src, double factor) {
        Mat dst = new Mat();
        src.convertTo(dst, CvType.CV_8UC3, factor);
        return dst;
    }

    /** Blacks out the top-left {@code fraction} of the placed rect on the canvas. */
    private static void occlude(Mat canvas, Rect placed, double fraction) {
        int occW = (int) Math.round(placed.width  * Math.sqrt(fraction));
        int occH = (int) Math.round(placed.height * Math.sqrt(fraction));
        occW = Math.min(occW, canvas.cols() - placed.x);
        occH = Math.min(occH, canvas.rows() - placed.y);
        if (occW > 0 && occH > 0) {
            Mat region = canvas.submat(new Rect(placed.x, placed.y, occW, occH));
            region.setTo(new Scalar(0, 0, 0));
            region.release();
        }
    }

    /** Applies a +{@code hueDelta} degree HSV hue rotation to a BGR image. */
    static Mat hueShift(Mat bgr, int hueDelta) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(bgr, hsv, Imgproc.COLOR_BGR2HSV);
        // OpenCV HSV: H is 0-179, so half-degree scale
        int delta = (hueDelta / 2) % 180;
        byte[] data = new byte[(int)(hsv.total() * hsv.channels())];
        hsv.get(0, 0, data);
        for (int i = 0; i < data.length; i += 3) {
            int h = (data[i] & 0xFF) + delta;
            if (h >= 180) h -= 180;
            data[i] = (byte) h;
        }
        hsv.put(0, 0, data);
        Mat result = new Mat();
        Imgproc.cvtColor(hsv, result, Imgproc.COLOR_HSV2BGR);
        hsv.release();
        return result;
    }

    // -------------------------------------------------------------------------
    // Private scene factory helper
    // -------------------------------------------------------------------------

    private static SceneEntry scene(ReferenceId ref, SceneCategory cat, String label,
                                    BackgroundId bg, SceneShapePlacement placement, Mat mat) {
        return new SceneEntry(ref, cat, label, bg, List.of(placement), mat);
    }

    // -------------------------------------------------------------------------
    // Slim catalogue builders  (for initial / fast runs — see SceneCatalogue)
    // -------------------------------------------------------------------------

    /**
     * Category A slim: 1 clean scene (solid black background, centred, no transform).
     * Gives 1 scene × 88 refs = 88 ref-scene pairs.
     */
    public static List<SceneEntry> buildCategoryASlim() {
        List<SceneEntry> scenes = new ArrayList<>();
        for (ReferenceId ref : ReferenceId.values()) {
            Mat refMat = ReferenceImageFactory.build(ref);
            Mat canvas = BackgroundFactory.build(BackgroundId.BG_SOLID_BLACK, W, H);
            Rect placed = placeAtCentre(canvas, refMat, 1.0, 0.0, 0, 0);
            scenes.add(new SceneEntry(ref, SceneCategory.A_CLEAN,
                    "clean_solid_black", BackgroundId.BG_SOLID_BLACK,
                    List.of(SceneShapePlacement.clean(ref, placed)), canvas));
            refMat.release();
        }
        return scenes;
    }

    /**
     * Category B slim: 1 transformed scene per reference (scale 0.75 + rotate 45°).
     * Gives 1 scene × 88 refs = 88 ref-scene pairs.
     */
    public static List<SceneEntry> buildCategoryBSlim() {
        List<SceneEntry> scenes = new ArrayList<>();
        for (ReferenceId ref : ReferenceId.values()) {
            Mat refMat = ReferenceImageFactory.build(ref);
            Mat canvas = BackgroundFactory.build(BackgroundId.BG_RANDOM_MIXED, W, H);
            Rect placed = placeAtCentre(canvas, refMat, 0.75, 45.0, 0, 0);
            scenes.add(new SceneEntry(ref, SceneCategory.B_TRANSFORMED,
                    "scale0.75_rot45", BackgroundId.BG_RANDOM_MIXED,
                    List.of(SceneShapePlacement.transformed(ref, placed, 0.75, 45.0, 0, 0)),
                    canvas));
            refMat.release();
        }
        return scenes;
    }

    /**
     * Category C slim: 1 degraded scene per reference (Gaussian noise σ=25).
     * Gives 1 scene × 88 refs = 88 ref-scene pairs.
     */
    public static List<SceneEntry> buildCategoryCSlim() {
        List<SceneEntry> scenes = new ArrayList<>();
        int vi = 0;
        BackgroundId[] bgs = BackgroundId.values();
        for (ReferenceId ref : ReferenceId.values()) {
            Mat refMat = ReferenceImageFactory.build(ref);
            BackgroundId bg = bgs[vi++ % bgs.length];
            Mat canvas = BackgroundFactory.build(bg, W, H);
            Rect placed = placeAtCentre(canvas, refMat, 1.0, 0, 0, 0);
            addGaussianNoise(canvas, 25, 52L + ref.ordinal());
            scenes.add(new SceneEntry(ref, SceneCategory.C_DEGRADED, "noise_s25", bg,
                    List.of(SceneShapePlacement.clean(ref, placed)), canvas));
            refMat.release();
        }
        return scenes;
    }

    /**
     * Category D slim: 1 negative scene (solid black, no reference present).
     */
    public static List<SceneEntry> buildCategoryDSlim() {
        Mat canvas = BackgroundFactory.build(BackgroundId.BG_SOLID_BLACK, W, H);
        return List.of(new SceneEntry(null, SceneCategory.D_NEGATIVE,
                "negative_solid_black", BackgroundId.BG_SOLID_BLACK, List.of(), canvas));
    }

    /**
     * Debug catalogue: 3 scenes for a single reference (A_CLEAN + B_TRANSFORMED + D_NEGATIVE).
     * Total = 3 scenes × 18 variants = 54 matcher calls.
     */
    public static List<SceneEntry> buildDebugScenes(ReferenceId ref) {
        List<SceneEntry> scenes = new ArrayList<>();
        Mat refMat = ReferenceImageFactory.build(ref);

        // A_CLEAN — solid black, centred
        Mat canvasA = BackgroundFactory.build(BackgroundId.BG_SOLID_BLACK, W, H);
        Rect placedA = placeAtCentre(canvasA, refMat, 1.0, 0.0, 0, 0);
        scenes.add(new SceneEntry(ref, SceneCategory.A_CLEAN,
                "debug_clean", BackgroundId.BG_SOLID_BLACK,
                List.of(SceneShapePlacement.clean(ref, placedA)), canvasA));

        // B_TRANSFORMED — random mixed bg, scale 0.75 + rot 45°
        Mat canvasB = BackgroundFactory.build(BackgroundId.BG_RANDOM_MIXED, W, H);
        Rect placedB = placeAtCentre(canvasB, refMat, 0.75, 45.0, 0, 0);
        scenes.add(new SceneEntry(ref, SceneCategory.B_TRANSFORMED,
                "debug_scale0.75_rot45", BackgroundId.BG_RANDOM_MIXED,
                List.of(SceneShapePlacement.transformed(ref, placedB, 0.75, 45.0, 0, 0)),
                canvasB));

        // D_NEGATIVE — solid black, no reference
        Mat canvasD = BackgroundFactory.build(BackgroundId.BG_SOLID_BLACK, W, H);
        scenes.add(new SceneEntry(null, SceneCategory.D_NEGATIVE,
                "debug_negative", BackgroundId.BG_SOLID_BLACK, List.of(), canvasD));

        refMat.release();
        return scenes;
    }

    // -------------------------------------------------------------------------
    // Multi-shape scenes
    // -------------------------------------------------------------------------

    /**
     * Builds a set of demonstration scenes that each contain 2–4 different reference shapes
     * arranged in non-overlapping quadrant positions, with per-shape transforms.
     *
     * <p>These are <em>not</em> part of the main A/B/C/D categories — they exist purely as
     * sample scenes for human inspection and to validate multi-placement metadata.
     *
     * <p>Every scene carries one {@link SceneShapePlacement} per placed reference,
     * recording each shape's position and transform individually.
     *
     * @return list of multi-shape {@link SceneEntry} objects
     */
    public static List<SceneEntry> buildMultiShape() {
        // Quadrant offsets from scene centre (640x480 scene, shapes are ~128px)
        // Each quadrant centre is offset so 128px shapes don't overlap
        record Slot(int offX, int offY) {}
        Slot[] slots = {
            new Slot(-160, -100),  // top-left quadrant
            new Slot( 160, -100),  // top-right quadrant
            new Slot(-160,  100),  // bottom-left quadrant
            new Slot( 160,  100),  // bottom-right quadrant
        };

        // Define several hand-picked multi-shape scene descriptors
        record ShapeSpec(ReferenceId ref, double scale, double rot) {}
        record SceneSpec(String label, BackgroundId bg, ShapeSpec[] shapes) {}

        SceneSpec[] specs = {
            // 2 shapes: horizontal line + filled circle, clean
            new SceneSpec("multi_2_clean", BackgroundId.BG_SOLID_BLACK, new ShapeSpec[]{
                new ShapeSpec(ReferenceId.LINE_H,        1.0,   0),
                new ShapeSpec(ReferenceId.CIRCLE_FILLED, 1.0,   0),
            }),
            // 2 shapes: rotated rect + triangle
            new SceneSpec("multi_2_rotated", BackgroundId.BG_GRADIENT_H_COLOUR, new ShapeSpec[]{
                new ShapeSpec(ReferenceId.RECT_OUTLINE,     1.0,  45),
                new ShapeSpec(ReferenceId.TRIANGLE_FILLED,  1.0,  30),
            }),
            // 3 shapes: circle outline + pentagon + text A, mixed scales
            new SceneSpec("multi_3_scaled", BackgroundId.BG_NOISE_LIGHT, new ShapeSpec[]{
                new ShapeSpec(ReferenceId.CIRCLE_OUTLINE,   0.75,  0),
                new ShapeSpec(ReferenceId.PENTAGON_OUTLINE, 1.25,  0),
                new ShapeSpec(ReferenceId.TEXT_A,           1.0,   0),
            }),
            // 3 shapes: star + hexagon + crosshair, all rotated differently
            new SceneSpec("multi_3_rotated", BackgroundId.BG_RANDOM_MIXED, new ShapeSpec[]{
                new ShapeSpec(ReferenceId.STAR_5_OUTLINE,  1.0,  15),
                new ShapeSpec(ReferenceId.HEXAGON_OUTLINE, 1.0,  60),
                new ShapeSpec(ReferenceId.CROSSHAIR,       1.0,  45),
            }),
            // 4 shapes: one per quadrant, clean placement
            new SceneSpec("multi_4_clean", BackgroundId.BG_SOLID_BLACK, new ShapeSpec[]{
                new ShapeSpec(ReferenceId.RECT_FILLED,     1.0,   0),
                new ShapeSpec(ReferenceId.CIRCLE_OUTLINE,  1.0,   0),
                new ShapeSpec(ReferenceId.TRIANGLE_OUTLINE,1.0,   0),
                new ShapeSpec(ReferenceId.STAR_5_FILLED,   1.0,   0),
            }),
            // 4 shapes: all rotated + scaled
            new SceneSpec("multi_4_transformed", BackgroundId.BG_CIRCUIT_LIKE, new ShapeSpec[]{
                new ShapeSpec(ReferenceId.RECT_OUTLINE,    1.25,  30),
                new ShapeSpec(ReferenceId.ELLIPSE_H,       0.75,  90),
                new ShapeSpec(ReferenceId.HEXAGON_FILLED,  1.0,   45),
                new ShapeSpec(ReferenceId.POLYLINE_ZIGZAG_H, 1.0, 180),
            }),
            // 2 shapes on complex background: line + polygon, rotated
            new SceneSpec("multi_2_complex_bg", BackgroundId.BG_ORGANIC, new ShapeSpec[]{
                new ShapeSpec(ReferenceId.LINE_DIAG_45,    1.0,  90),
                new ShapeSpec(ReferenceId.OCTAGON_OUTLINE, 1.0,  22),
            }),
            // 4 shapes: grid pattern, dashed, compound shapes
            new SceneSpec("multi_4_compound", BackgroundId.BG_GRADIENT_RADIAL_COLOUR, new ShapeSpec[]{
                new ShapeSpec(ReferenceId.GRID_4X4,                  1.0,   0),
                new ShapeSpec(ReferenceId.LINE_DASHED_CROSS,         1.0,  45),
                new ShapeSpec(ReferenceId.COMPOUND_TRIANGLE_IN_CIRCLE, 0.9, 0),
                new ShapeSpec(ReferenceId.CHECKERBOARD_2X2,          0.9,   0),
            }),
        };

        List<SceneEntry> scenes = new ArrayList<>();

        for (SceneSpec spec : specs) {
            Mat canvas = BackgroundFactory.build(spec.bg(), W, H);
            List<SceneShapePlacement> placements = new ArrayList<>();

            for (int i = 0; i < spec.shapes().length; i++) {
                ShapeSpec ss = spec.shapes()[i];
                Slot slot = slots[i % slots.length];

                Mat refMat = ReferenceImageFactory.build(ss.ref());
                Rect placed = placeAtCentre(canvas, refMat, ss.scale(), ss.rot(),
                        slot.offX(), slot.offY());
                refMat.release();

                placements.add(new SceneShapePlacement(
                        ss.ref(), placed, ss.scale(), ss.rot(),
                        slot.offX(), slot.offY(), false, false, 0.0));
            }

            // primaryReferenceId = first shape placed
            ReferenceId primary = spec.shapes()[0].ref();
            scenes.add(new SceneEntry(primary, SceneCategory.A_CLEAN,
                    spec.label(), spec.bg(), placements, canvas));
        }

        return scenes;
    }
}


