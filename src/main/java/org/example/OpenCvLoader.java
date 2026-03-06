package org.example;

/**
 * Singleton wrapper around the OpenCV native library loader.
 *
 * Call {@link #load()} once in {@code @BeforeAll} for every test class.
 * Subsequent calls are no-ops thanks to the {@code loaded} guard.
 *
 * Uses openpnp's bundled native library so no manual {@code -Djava.library.path}
 * configuration is required.
 */
public final class OpenCvLoader {

    private static volatile boolean loaded = false;

    private OpenCvLoader() {}

    /**
     * Loads the OpenCV native library if it has not already been loaded.
     * Thread-safe via double-checked locking.
     */
    public static void load() {
        if (!loaded) {
            synchronized (OpenCvLoader.class) {
                if (!loaded) {
                    nu.pattern.OpenCV.loadShared();
                    loaded = true;
                    System.out.println("[OpenCvLoader] Native library loaded: "
                            + org.opencv.core.Core.VERSION);
                }
            }
        }
    }
}

