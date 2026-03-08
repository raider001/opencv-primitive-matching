package org.example;

import org.opencv.core.Mat;

import java.nio.file.Path;
import java.util.List;
import java.util.Set;

/**
 * Common interface for all pattern matchers, exposing metadata and a uniform
 * invocation point.
 *
 * <p>Each matcher (Template Matching, Feature Matching, etc.) registers one
 * {@code MatcherDescriptor} with {@code MatcherRegistry}.  The benchmark
 * launcher uses this interface to:
 * <ul>
 *   <li>Populate the UI's matcher / variant selection lists.</li>
 *   <li>Invoke the matcher without reflection or hard-coded dispatch.</li>
 *   <li>Know where the output report will be written.</li>
 * </ul>
 */
public interface MatcherDescriptor {

    /** Short uppercase tag used in log lines and progress display, e.g. {@code "TM"}. */
    String tag();

    /** Human-readable technique name for UI labels and HTML report titles. */
    String displayName();

    /**
     * Root output directory for this technique's {@code report.html} and annotated images.
     * Relative paths are resolved against the working directory at runtime.
     */
    Path outputDir();

    /**
     * All variant names this matcher can produce, in the order they should
     * appear in the UI.  Each name is the exact string stored in
     * {@link AnalysisResult#methodName()}.
     */
    List<String> variantNames();

    /**
     * Runs this matcher for one (reference, scene) pair and returns one
     * {@link AnalysisResult} per active variant.
     *
     * @param refId        the reference being searched for
     * @param refMat       128×128 BGR reference Mat — caller retains ownership
     * @param scene        the scene to search in
     * @param saveVariants variant names whose annotated PNG should be saved
     * @param outputDir    root output directory (may differ from {@link #outputDir()}
     *                     when a custom run path is chosen)
     * @return list of results — one per active variant
     */
    List<AnalysisResult> run(ReferenceId refId, Mat refMat,
                             SceneEntry scene,
                             Set<String> saveVariants,
                             Path outputDir);
}

