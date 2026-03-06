package org.example;

/**
 * Top-level category for a scene entry in the catalogue.
 *
 * <ul>
 *   <li>{@link #A_CLEAN}      — reference placed cleanly at centre, no transforms</li>
 *   <li>{@link #B_TRANSFORMED}— reference scaled and/or rotated and/or repositioned</li>
 *   <li>{@link #C_DEGRADED}   — reference present but scene is heavily degraded</li>
 *   <li>{@link #D_NEGATIVE}   — no reference present (false-positive test scenes)</li>
 * </ul>
 */
public enum SceneCategory {
    A_CLEAN,
    B_TRANSFORMED,
    C_DEGRADED,
    D_NEGATIVE
}

