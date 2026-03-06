package org.example;

import java.util.Map;

/**
 * Aggregated performance statistics for one matcher variant.
 *
 * @param methodVariant   e.g. "TM_CCOEFF_NORMED"
 * @param minMs           minimum elapsed ms across all scenes
 * @param maxMs           maximum elapsed ms
 * @param avgMs           mean elapsed ms
 * @param p95Ms           95th-percentile elapsed ms
 * @param msPerMp         normalised throughput — avg ms per megapixel at test resolution
 * @param projectedMs     map from resolution label → [linear_ms, quadratic_ms]
 *                        keys: "720p", "1080p", "1440p", "4K"
 * @param estimatedHeapMb map from resolution label → estimated working heap MB
 *                        keys: "640x480", "720p", "1080p", "1440p", "4K"
 */
public record PerformanceProfile(
        String              methodVariant,
        long                minMs,
        long                maxMs,
        double              avgMs,
        long                p95Ms,
        double              msPerMp,
        Map<String,double[]> projectedMs,
        Map<String,Double>   estimatedHeapMb
) {}

