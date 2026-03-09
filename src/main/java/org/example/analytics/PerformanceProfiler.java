package org.example.analytics;

import java.util.*;

/**
 * Computes {@link PerformanceProfile} from a list of {@link AnalysisResult} objects.
 *
 * <p>Call {@link #profile(List, String)} after all results are collected (in {@code @AfterAll})
 * so that profiling does not interfere with timing accuracy during the test loop.
 */
public final class PerformanceProfiler {

    // Test base resolution
    private static final double BASE_MP = (640.0 * 480.0) / 1_000_000.0; // 0.3072 MP

    // Target resolutions: label -> megapixels
    private static final LinkedHashMap<String, Double> RESOLUTIONS = new LinkedHashMap<>();
    static {
        RESOLUTIONS.put("640x480", BASE_MP);
        RESOLUTIONS.put("720p",    (1280.0 * 720.0)  / 1_000_000.0);
        RESOLUTIONS.put("1080p",   (1920.0 * 1080.0) / 1_000_000.0);
        RESOLUTIONS.put("1440p",   (2560.0 * 1440.0) / 1_000_000.0);
        RESOLUTIONS.put("4K",      (3840.0 * 2160.0) / 1_000_000.0);
    }

    // Per-algorithm memory multipliers (scene Mat bytes × multiplier = working heap estimate)
    private static final Map<String, Double> MEMORY_MULTIPLIERS = new HashMap<>();
    static {
        MEMORY_MULTIPLIERS.put("TemplateMatcher",      1.3);
        MEMORY_MULTIPLIERS.put("FeatureMatcher",       3.0);
        MEMORY_MULTIPLIERS.put("ContourShapeMatcher",  1.1);
        MEMORY_MULTIPLIERS.put("HoughDetector",        1.5);
        MEMORY_MULTIPLIERS.put("GeneralizedHough",     4.0);
        MEMORY_MULTIPLIERS.put("HistogramMatcher",     1.2);
        MEMORY_MULTIPLIERS.put("PhaseCorrelation",     4.0);
        MEMORY_MULTIPLIERS.put("MorphologyAnalyzer",   1.2);
        MEMORY_MULTIPLIERS.put("PixelDiffMatcher",     2.0);
        MEMORY_MULTIPLIERS.put("DummyMatcher",         1.0); // used in infrastructure test
    }

    private PerformanceProfiler() {}

    /**
     * Aggregates timing statistics from {@code results} filtered to {@code methodVariant}
     * and projects performance to higher resolutions.
     *
     * @param results       all results collected during the test run
     * @param methodVariant the variant name to filter on (must match {@link AnalysisResult#methodName()})
     * @return a {@link PerformanceProfile} for that variant
     */
    public static PerformanceProfile profile(List<AnalysisResult> results, String methodVariant) {
        List<Long> times = results.stream()
                .filter(r -> r.methodName().equals(methodVariant) && !r.isError())
                .map(AnalysisResult::elapsedMs)
                .sorted()
                .toList();

        if (times.isEmpty()) {
            return new PerformanceProfile(methodVariant, 0, 0, 0, 0, 0,
                    emptyProjections(), emptyHeap());
        }

        long minMs = times.get(0);
        long maxMs = times.get(times.size() - 1);
        double avgMs = times.stream().mapToLong(Long::longValue).average().orElse(0);
        long p95Ms  = times.get((int) Math.min(times.size() - 1,
                Math.ceil(times.size() * 0.95)));
        double msPerMp = avgMs / BASE_MP;

        // Projection per resolution
        Map<String, double[]> projectedMs = new LinkedHashMap<>();
        Map<String, Double>   heapMb      = new LinkedHashMap<>();
        double multiplier = resolveMultiplier(methodVariant);

        for (Map.Entry<String, Double> res : RESOLUTIONS.entrySet()) {
            String label = res.getKey();
            double targetMp = res.getValue();
            double ratio = targetMp / BASE_MP;
            double linear    = avgMs * ratio;
            double quadratic = avgMs * ratio * ratio;
            projectedMs.put(label, new double[]{ linear, quadratic });

            // Scene Mat: width×height×3 bytes → MB
            double[] dims = resolutionDims(label);
            double sceneBytes = dims[0] * dims[1] * 3.0;
            double sceneMb = sceneBytes / (1024.0 * 1024.0);
            heapMb.put(label, sceneMb * multiplier);
        }

        return new PerformanceProfile(methodVariant, minMs, maxMs, avgMs, p95Ms,
                msPerMp, projectedMs, heapMb);
    }

    /**
     * Generates a human-readable interpretation note for the profile.
     * e.g. "TM_CCOEFF_NORMED averages 4 ms at 640×480. Projects to ~27 ms at 1080p
     * (~37 fps) and ~109 ms at 4K (~9 fps). Suitable for real-time use up to 1080p."
     */
    public static String interpretationNote(PerformanceProfile p) {
        if (p.avgMs() == 0) return p.methodVariant() + ": no timing data available.";

        double[] p1080 = p.projectedMs().get("1080p");
        double[] p4k   = p.projectedMs().get("4K");
        double avg1080 = p1080 != null ? (p1080[0] + p1080[1]) / 2.0 : 0;
        double avg4k   = p4k   != null ? (p4k[0]   + p4k[1])   / 2.0 : 0;

        long fps1080 = avg1080 > 0 ? Math.round(1000.0 / avg1080) : 0;
        long fps4k   = avg4k   > 0 ? Math.round(1000.0 / avg4k)   : 0;

        String suitability;
        if (avg4k > 0 && fps4k >= 24) {
            suitability = "Suitable for real-time use at 4K.";
        } else if (fps1080 >= 24) {
            suitability = "Suitable for real-time use up to 1080p.";
        } else if (p.avgMs() <= 50) {
            suitability = "Suitable for real-time use up to 720p.";
        } else {
            suitability = "Not suitable for real-time use without hardware acceleration.";
        }

        return String.format(
            "%s averages %.0f ms at 640×480. Projects to ~%.0f ms at 1080p (~%d fps) " +
            "and ~%.0f ms at 4K (~%d fps). %s",
            p.methodVariant(), p.avgMs(), avg1080, fps1080, avg4k, fps4k, suitability);
    }

    /**
     * Profiles all distinct method variants found in {@code results}, printing
     * progress every 5 variants (or whenever done).
     *
     * @param results all collected analysis results
     * @return one {@link PerformanceProfile} per distinct method variant, sorted by name
     */
    public static List<PerformanceProfile> profileAll(List<AnalysisResult> results) {
        List<String> variants = results.stream()
                .map(AnalysisResult::methodName)
                .distinct()
                .sorted()
                .toList();

        int total = variants.size(), done = 0;
        long t0 = System.currentTimeMillis();
        System.out.printf("[PERF] Profiling %d variants...%n", total);

        List<PerformanceProfile> profiles = new ArrayList<>(total);
        for (String v : variants) {
            profiles.add(profile(results, v));
            done++;
            if (done % 5 == 0 || done == total) {
                System.out.printf("[PERF] %d/%d variants profiled  (%.1fs)%n",
                        done, total, (System.currentTimeMillis() - t0) / 1000.0);
            }
        }
        System.out.printf("[PERF] Profiling complete in %.1fs%n",
                (System.currentTimeMillis() - t0) / 1000.0);
        return profiles;
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private static double resolveMultiplier(String methodVariant) {
        for (Map.Entry<String, Double> e : MEMORY_MULTIPLIERS.entrySet()) {
            if (methodVariant.contains(e.getKey())) return e.getValue();
        }
        return 1.5; // sensible default
    }

    private static double[] resolutionDims(String label) {
        return switch (label) {
            case "640x480" -> new double[]{ 640,  480  };
            case "720p"    -> new double[]{ 1280, 720  };
            case "1080p"   -> new double[]{ 1920, 1080 };
            case "1440p"   -> new double[]{ 2560, 1440 };
            case "4K"      -> new double[]{ 3840, 2160 };
            default        -> new double[]{ 640,  480  };
        };
    }

    private static Map<String, double[]> emptyProjections() {
        Map<String, double[]> m = new LinkedHashMap<>();
        RESOLUTIONS.keySet().forEach(k -> m.put(k, new double[]{0, 0}));
        return m;
    }

    private static Map<String, Double> emptyHeap() {
        Map<String, Double> m = new LinkedHashMap<>();
        RESOLUTIONS.keySet().forEach(k -> m.put(k, 0.0));
        return m;
    }
}

