package org.example.utilities;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Documents the expected outcome of a VectorMatcher diagnostic or robustness test.
 *
 * <p>This annotation is informational — it does not affect JUnit execution.
 * It records the design-time expectation alongside the reason so that future
 * readers understand immediately whether a "failing" result is a genuine
 * regression or an accepted geometric limitation.
 *
 * <h2>Values</h2>
 * <ul>
 *   <li>{@link Result#PASS}       — all shapes expected to meet the ≥ 90 % target</li>
 *   <li>{@link Result#PARTIAL}    — known subset of shapes or conditions will fall
 *       below target for documented geometric or architectural reasons</li>
 *   <li>{@link Result#FAIL}       — the scenario is expected to score below target;
 *       exists to document a known hard case, not a regression</li>
 *   <li>{@link Result#DIAGNOSTIC} — no pass/fail criterion; test prints structural
 *       information for human inspection only</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * @Test
 * @ExpectedOutcome(value = Result.PARTIAL,
 *                 reason = "HEXAGON_OUTLINE and STAR_5_FILLED score ~86-87% across all " +
 *                          "backgrounds — below the 90% target due to contour approximation " +
 *                          "variance at this scale.")
 * void runDiagnostics() { ... }
 * }</pre>
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface ExpectedOutcome {

    /** Coarse classification of what this test is expected to produce. */
    Result value();

    /**
     * Human-readable explanation of why this outcome is expected.
     * Should reference specific shapes, backgrounds, or architectural constraints.
     */
    String reason();

    enum Result {
        /** All shapes / conditions expected to meet the ≥ 90 % accuracy target. */
        PASS,
        /**
         * A documented subset of shapes or rotation angles will fall below target
         * for known geometric or architectural reasons — not a regression.
         */
        PARTIAL,
        /**
         * The scenario is expected to score below target overall.
         * Exists to document a known hard case explicitly.
         */
        FAIL,
        /**
         * No pass/fail criterion — the test prints structural diagnostics
         * (signatures, cluster layouts, etc.) for human inspection only.
         */
        DIAGNOSTIC
    }
}

