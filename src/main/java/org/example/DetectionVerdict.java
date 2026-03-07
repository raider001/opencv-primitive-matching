package org.example;

import org.opencv.core.Rect;

/**
 * Ground-truth verdict for a single {@link AnalysisResult} evaluated against
 * the {@link SceneShapePlacement} ground truth stored in the scene's JSON sidecar.
 *
 * <h2>Classification rules</h2>
 * <pre>
 *  queried ref in scene?   score ≥ threshold?   centre-in-GT?   verdict
 *  ─────────────────────   ──────────────────   ─────────────   ──────────────────
 *  yes (matching ref)      yes                  yes             CORRECT
 *  yes (matching ref)      yes                  no              WRONG_LOCATION
 *  yes (matching ref)      no                   —               MISSED
 *  no  (Cat D or diff ref) yes                  —               FALSE_ALARM
 *  no  (Cat D or diff ref) no                   —               CORRECTLY_REJECTED
 * </pre>
 *
 * <h2>Localisation check</h2>
 * <p>Rather than IoU (which breaks when the matcher returns a fixed-size bbox that
 * differs from the scaled/rotated placed shape), we check whether the <em>centre</em>
 * of the predicted bounding box falls within the ground-truth {@code placedRect}
 * expanded by {@value #CENTRE_TOLERANCE_PX} pixels on each side.  This is robust
 * to scale changes and rotation — if the matcher found the right region, its bbox
 * centre will land on the shape regardless of size.
 *
 * <p>Default score threshold: ≥ {@value #DEFAULT_SCORE_THRESHOLD}%.
 */
public enum DetectionVerdict {

    /** Reference present, detected, and bbox centre is inside the ground-truth region. */
    CORRECT,
    /** Reference present and detected, but bbox centre is outside the ground-truth region. */
    WRONG_LOCATION,
    /** Reference present but score below threshold — shape was not found. */
    MISSED,
    /** Reference absent but score exceeds threshold — shape was hallucinated. */
    FALSE_ALARM,
    /** Reference absent and score below threshold — correctly rejected. */
    CORRECTLY_REJECTED;

    // -------------------------------------------------------------------------
    // Thresholds
    // -------------------------------------------------------------------------

    /** Score percent at or above which a detection is considered "triggered". */
    public static final double DEFAULT_SCORE_THRESHOLD = 50.0;

    /**
     * Pixels of expansion applied to the ground-truth rect when checking whether
     * the predicted bbox centre falls "inside" it.  Accounts for small alignment
     * errors and the fact that the placed rect may be the exact shape bounds while
     * the matcher bbox is template-sized.
     */
    public static final int CENTRE_TOLERANCE_PX = 24;

    // -------------------------------------------------------------------------
    // Factory
    // -------------------------------------------------------------------------

    /** Evaluates using default thresholds. */
    public static DetectionVerdict evaluate(AnalysisResult result, SceneEntry scene) {
        return evaluate(result, scene, DEFAULT_SCORE_THRESHOLD, CENTRE_TOLERANCE_PX);
    }

    /**
     * Evaluates a single result against its scene's ground truth.
     *
     * @param result             the analysis result to evaluate
     * @param scene              the scene that was analysed
     * @param scoreThreshold     minimum score% to consider the detector "fired"
     * @param centreTolerance    pixels of padding added to the GT rect for centre check
     */
    public static DetectionVerdict evaluate(AnalysisResult result, SceneEntry scene,
                                            double scoreThreshold, int centreTolerance) {
        boolean detected = !result.isError() && result.matchScorePercent() >= scoreThreshold;

        // Does this scene actually contain the reference being queried?
        // A scene may have a reference, but if it's a *different* reference it is
        // effectively a negative scene for this particular query (the queried shape
        // is absent), so we apply the same logic as Category D.
        boolean hasThisRef = scene.hasReference()
                          && result.referenceId() != null
                          && scene.primaryReferenceId() == result.referenceId();

        if (!hasThisRef) {
            // The queried shape is not in this scene — firing is a false alarm
            return detected ? FALSE_ALARM : CORRECTLY_REJECTED;
        }

        if (!detected) {
            return MISSED;
        }

        // Detected — check whether the predicted bbox centre lands on the shape
        Rect predicted   = result.boundingRect();
        Rect groundTruth = scene.placements().stream()
                .filter(p -> p.referenceId() == result.referenceId())
                .map(SceneShapePlacement::placedRect)
                .findFirst().orElse(null);

        if (predicted == null || groundTruth == null) {
            return CORRECT;
        }

        return centreInRect(predicted, groundTruth, centreTolerance) ? CORRECT : WRONG_LOCATION;
    }

    // -------------------------------------------------------------------------
    // Geometry helpers
    // -------------------------------------------------------------------------

    /**
     * Returns true if the centre of {@code predicted} falls within {@code gt}
     * expanded by {@code tolerancePx} on every side.
     */
    public static boolean centreInRect(Rect predicted, Rect gt, int tolerancePx) {
        double cx = predicted.x + predicted.width  / 2.0;
        double cy = predicted.y + predicted.height / 2.0;
        int x1 = gt.x - tolerancePx;
        int y1 = gt.y - tolerancePx;
        int x2 = gt.x + gt.width  + tolerancePx;
        int y2 = gt.y + gt.height + tolerancePx;
        return cx >= x1 && cx <= x2 && cy >= y1 && cy <= y2;
    }

    /**
     * Computes Intersection-over-Union for two axis-aligned rectangles.
     * Retained as a utility even though the primary check now uses centre-in-rect.
     */
    public static double iou(Rect a, Rect b) {
        int ix = Math.max(a.x, b.x);
        int iy = Math.max(a.y, b.y);
        int iw = Math.min(a.x + a.width,  b.x + b.width)  - ix;
        int ih = Math.min(a.y + a.height, b.y + b.height) - iy;
        if (iw <= 0 || ih <= 0) return 0.0;
        double intersection = (double) iw * ih;
        double union        = (double) a.width * a.height
                            + (double) b.width * b.height
                            - intersection;
        return union <= 0 ? 0.0 : intersection / union;
    }

    // -------------------------------------------------------------------------
    // Display helpers
    // -------------------------------------------------------------------------

    public String emoji() {
        return switch (this) {
            case CORRECT             -> "✅";
            case WRONG_LOCATION      -> "📍";
            case MISSED              -> "❌";
            case FALSE_ALARM         -> "⚠️";
            case CORRECTLY_REJECTED  -> "✓";
        };
    }

    /** Full human-readable label shown in the HTML report and console. */
    public String label() {
        return switch (this) {
            case CORRECT             -> "Correct";
            case WRONG_LOCATION      -> "Wrong Location";
            case MISSED              -> "Missed";
            case FALSE_ALARM         -> "False Alarm";
            case CORRECTLY_REJECTED  -> "Correctly Rejected";
        };
    }

    /** Short label for compact table cells. */
    public String shortLabel() {
        return switch (this) {
            case CORRECT             -> "TP";
            case WRONG_LOCATION      -> "FP (loc)";
            case MISSED              -> "FN";
            case FALSE_ALARM         -> "FP (alarm)";
            case CORRECTLY_REJECTED  -> "TN";
        };
    }

    public String cssClass() {
        return switch (this) {
            case CORRECT             -> "tp";
            case WRONG_LOCATION      -> "fp";
            case MISSED              -> "fn";
            case FALSE_ALARM         -> "fp";
            case CORRECTLY_REJECTED  -> "tn";
        };
    }
}



