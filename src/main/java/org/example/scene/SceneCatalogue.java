package org.example.scene;

import org.example.factories.ReferenceId;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Singleton that holds the full in-memory scene catalogue.
 *
 * <p>Call {@link #build()} once (typically in {@code @BeforeAll}).
 * The resulting list is immutable and shared across all technique tests.
 *
 * <p>Full catalogue expected counts: A=352, B=1144, D=44 → total=1540
 *
 * <p>Slim catalogue (for initial runs): 4 scenes total (1 A + 1 B + 1 C + 1 D per ref/bg)
 * → 352 ref-scene pairs × 18 variants ≈ 6,336 matcher calls.
 */
public final class SceneCatalogue {

    private static volatile List<SceneEntry> INSTANCE;
    private static volatile List<SceneEntry> SLIM_INSTANCE;

    private SceneCatalogue() {}

    /**
     * Builds and caches the full catalogue. Thread-safe; idempotent.
     *
     * @return unmodifiable list of all {@link SceneEntry} objects
     */
    public static List<SceneEntry> build() {
        if (INSTANCE != null) return INSTANCE;
        synchronized (SceneCatalogue.class) {
            if (INSTANCE != null) return INSTANCE;
            List<SceneEntry> all = new ArrayList<>();
            all.addAll(SceneGenerator.buildCategoryA());
            all.addAll(SceneGenerator.buildCategoryB());
            all.addAll(SceneGenerator.buildCategoryD());
            INSTANCE = Collections.unmodifiableList(all);
        }
        return INSTANCE;
    }

    /**
     * Builds and caches a slim catalogue suitable for initial technique runs.
     *
     * <p>Scene counts:
     * <ul>
     *   <li>A_CLEAN       — 1 scene per reference (88 scenes, solid black bg)</li>
     *   <li>B_TRANSFORMED — 1 scene per reference (88 scenes, scale 0.75 + rot 45°)</li>
     *   <li>D_NEGATIVE    — 1 fixed negative scene</li>
     * </ul>
     * Total: 177 scenes.
     *
     * @return unmodifiable slim scene list
     */
    public static List<SceneEntry> buildSlim() {
        if (SLIM_INSTANCE != null) return SLIM_INSTANCE;
        synchronized (SceneCatalogue.class) {
            if (SLIM_INSTANCE != null) return SLIM_INSTANCE;
            List<SceneEntry> all = new ArrayList<>();
            all.addAll(SceneGenerator.buildCategoryASlim());
            all.addAll(SceneGenerator.buildCategoryBSlim());
            all.addAll(SceneGenerator.buildCategoryDSlim());
            SLIM_INSTANCE = Collections.unmodifiableList(all);
        }
        return SLIM_INSTANCE;
    }

    /** Returns the cached full catalogue, or throws if {@link #build()} has not been called. */
    public static List<SceneEntry> get() {
        if (INSTANCE == null) throw new IllegalStateException("SceneCatalogue.build() not called yet");
        return INSTANCE;
    }

    /**
     * Builds a minimal debug catalogue for a single reference — 3 scenes total:
     * 1× A_CLEAN (solid black), 1× B_TRANSFORMED (scale 0.75 + rot 45°), 1× D_NEGATIVE.
     * Total = 3 scenes × 18 variants = 54 matcher calls. Near-instant to construct.
     *
     * @param ref the single reference to build scenes for
     * @return unmodifiable list of 3 {@link SceneEntry} objects
     */
    public static List<SceneEntry> buildDebug(ReferenceId ref) {
        return SceneGenerator.buildDebugScenes(ref);
    }

    /** Counts entries in the given category. */
    public static long countCategory(SceneCategory cat) {
        return build().stream().filter(e -> e.category() == cat).count();
    }
}
