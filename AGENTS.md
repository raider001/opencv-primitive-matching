# AGENTS.md

## Overall Objective
- Build and maintain a pattern-matching utility that identifies shape/structure patterns in arbitrary scenes while remaining colour agnostic (colour may assist cluster discovery, but final matching/scoring should be geometry/structure driven).

## Scope (strict)
- This guide is for `VectorMatcher` execution work driven by `src/test/java/org/example/vectormatcher/VectorMatchingTest.java` and `src/test/java/org/example/vectormatcher/CharacterMatchingTest.java`.
- Include all files that participate in runtime execution (direct + transitive dependencies), not only direct imports.
- Treat `test_output/vector_matching/` as the canonical output area for `VectorMatchingTest` (`report.html`, `diagnostics.json`, debug PNGs, `annotated/`).
- Treat `test_output/character_matching/` as the canonical output area for `CharacterMatchingTest` (`report.html`, `diagnostics.json`, `sections/`).
- In scope: matcher pipeline, scene/cluster extraction, signature/scoring, scene/reference/background factories, scene wrappers, analytics records, and reporting/diagnostic helpers used by this suite.
- Out of scope: benchmark pipelines, unrelated matcher families, `AnalyticalTestBase` subclass tests, and non-`vector_matching`/non-`character_matching` output folders.

## Execution-Relevant Files (read first)
- `src/test/java/org/example/vectormatcher/VectorMatchingTest.java`
- `src/test/java/org/example/vectormatcher/CharacterMatchingTest.java` (character-level VectorMatcher tests: a–z, A–Z, 0–9, punctuation)
- `src/test/java/org/example/vectormatcher/CrossRejectDiagnosticTest.java` (focused cross-rejection diagnostics)
- `src/main/java/org/example/matchers/vectormatcher/VectorMatcher.java`
- `src/main/java/org/example/matchers/vectormatcher/README.md` (package architecture & design — **keep in sync**, see Maintenance Rules)
- `src/main/java/org/example/matchers/vectormatcher/components/RegionScorer.java`, `RegionScore.java`
- `src/main/java/org/example/matchers/vectormatcher/components/CandidateFilter.java`
- `src/main/java/org/example/matchers/vectormatcher/components/AnchorMatcher.java`
- `src/main/java/org/example/matchers/vectormatcher/components/BboxExpander.java`
- `src/main/java/org/example/matchers/vectormatcher/components/RefCluster.java`
- `src/main/java/org/example/matchers/vectormatcher/components/GeometryUtils.java`
- `src/main/java/org/example/matchers/SceneDescriptor.java`
- `src/main/java/org/example/matchers/SceneContourEntry.java` (record: one scene contour + cluster metadata + cached `VectorSignature`)
- `src/main/java/org/example/matchers/VectorSignature.java`
- `src/main/java/org/example/matchers/SegmentDescriptor.java` and `src/main/java/org/example/matchers/ContourTopology.java`
- `src/main/java/org/example/colour/SceneColourExtractor.java` (strategy interface — implemented by both `SceneColourClusters` and `ExperimentalSceneColourClusters`)
- `src/main/java/org/example/colour/SceneColourClusters.java`, `src/main/java/org/example/colour/ExperimentalSceneColourClusters.java`, `src/main/java/org/example/colour/ColourCluster.java`
- `src/main/java/org/example/factories/ReferenceImageFactory.java`, `src/main/java/org/example/factories/BackgroundFactory.java`, `src/main/java/org/example/factories/ReferenceId.java`, `src/main/java/org/example/factories/BackgroundId.java`
- `src/main/java/org/example/scene/SceneEntry.java`, `src/main/java/org/example/scene/SceneCategory.java`, `src/main/java/org/example/scene/SceneShapePlacement.java`
- `src/main/java/org/example/OpenCvLoader.java`, `src/main/java/org/example/analytics/AnalysisResult.java`
- `src/test/java/org/example/utilities/MatchReportLibrary.java`, `src/test/java/org/example/utilities/MatchDiagnosticLibrary.java`, `src/test/java/org/example/utilities/ExpectedOutcome.java`
- `pom.xml` (Surefire/JDK 25/OpenCV runtime flags for tests)

## Big Picture Flow
- `VectorMatchingTest` initializes OpenCV + output directory, runs scene builds and matcher calls, then writes consolidated report artifacts.
- `SceneEntry` prebuilds `SceneDescriptor`; `SceneDescriptor.build(...)` uses `ExperimentalSceneColourClusters.extractFromBorderPixels(bgrScene, chromaticOut)` (2-arg overload) for scene clustering; the `chromaticOut` byte array is populated in-place during cluster extraction and used to build `SceneDescriptor.combinedChromaticMask` with a single `Mat.put()` instead of N `bitwise_or` calls.
- `VectorMatcher` builds reference clusters via `ExperimentalSceneColourClusters.INSTANCE.extractFromBorderPixels(...)` (same extractor as scene-side — unified pipeline), scores candidates in 3 layers (Layer 1: boundary count, W=0.15; Layer 2: structural coherence, W=0.25; Layer 3: primary geometry via `VectorSignature.similarity`, W=0.60), and emits `VECTOR_NORMAL` result.
- **Critical invariant:** both ref and scene MUST use the same `SceneColourExtractor` implementation (`ExperimentalSceneColourClusters.INSTANCE`). Using different extractors causes systematic cluster-decomposition mismatches (e.g. the non-experimental extractor misses chromatic halves on `BICOLOUR_RECT_HALVES`).
- `MatchReportLibrary` and `MatchDiagnosticLibrary` record from the same matcher run and write `report.html` + `diagnostics.json`. Report is now section-based: individual stages write per-stage HTML fragments into `test_output/vector_matching/sections/`.
- Annotated images are written under `test_output/vector_matching/annotated/<variant>/` as `<query>_vs_<sceneRef>_<sceneLabel>.png`.
- `VectorMatcher.buildRefSignature()` and `buildRefSignatures()` are public helpers used by diagnostic probes in `VectorMatchingTest` to inspect per-contour similarity outside the main match pipeline.

## Project Conventions
- Java 25 source/target (set via `maven.compiler.source=25` in `pom.xml`); use modern Java features (records, pattern matching, etc.).
- Call `OpenCvLoader.load()` before any OpenCV `Mat` work in tests.
- Preserve native-memory hygiene: mirror existing `Mat.release()` and `SceneEntry.release()` patterns.
- `ExpectedOutcome` annotations are documentation surfaced in reports; JUnit assertions still define pass/fail.
- IoU in this suite is coverage-scaled (`MatchDiagnosticLibrary.iou`), and several checks combine IoU with score thresholds.
- `VectorVariant` declares strict/normal/loose, but core matcher execution path currently runs `VECTOR_NORMAL`.
- Detection pass: `MatchReportLibrary.isDetectionPass(score, iou)` — score > 70 % AND 0.90 < IoU ≤ 1.30. Rejection pass: `MatchReportLibrary.isRejectionPass(score)` — score < 60 %.
- Cross-reject tests use `@Tag("cross-reject")`; Surefire currently includes them. To re-exclude: add `<excludedGroups>cross-reject</excludedGroups>` to the Surefire config.
- `baseline/` at project root holds a frozen `report.html` and `sections/` snapshot for visual regression comparison.

## Dev Workflow (VectorMatcher suite)
- Run only this class: `mvn -Dtest=org.example.vectormatcher.VectorMatchingTest test`
- Run character matching: `mvn -Dtest=org.example.vectormatcher.CharacterMatchingTest test`
- Re-enable excluded groups when needed: `mvn test -DexcludedGroups=""`
- Keep Surefire native access args intact in `pom.xml` (`--enable-native-access=ALL-UNNAMED`).
- Run the test suite after any source or test file change to verify the build: `mvn -Dtest=org.example.vectormatcher.VectorMatchingTest test 2>&1 | tee test_run_latest.log`. If no files have changed since the last run, check `test_run_latest.log` (tail ~50 lines) or `target/surefire-reports/` first to avoid redundant ~3-minute runs.
- Tests run in parallel at the **class** level (configured in `src/test/resources/junit-platform.properties`). `VectorMatchingTest` methods also run in parallel via `@Execution(CONCURRENT)` — shared report/diagnostic state (`MatchReportLibrary`, `MatchDiagnosticLibrary`) uses `CopyOnWriteArrayList` for thread safety.

## Output Expectations
- `test_output/vector_matching/report.html`: visual consolidated table and expected-outcome notes.
- `test_output/vector_matching/diagnostics.json`: summary + per-row payload.
- `test_output/vector_matching/sections/`: per-stage HTML fragments (e.g. `self_match.html`, `rot45deg_black.html`, `cross_ref_rejection.html`) assembled into the consolidated report.
- `test_output/vector_matching/debug_scene_*.png`: focused probe scene dumps.
- `test_output/vector_matching/annotated/VECTOR_NORMAL/`: primary annotated outputs currently produced by matcher execution.
- `test_output/vector_matching/annotated/VECTOR_STRICT/` and `test_output/vector_matching/annotated/VECTOR_LOOSE/` may exist from earlier runs.
- `test_output/character_matching/report.html`: character-level visual consolidated report.
- `test_output/character_matching/diagnostics.json`: character-level diagnostic payload.
- `test_output/character_matching/sections/`: per-stage HTML fragments (e.g. `self_match_lowercase.html`, `cr_uppercase.html`, `alphabet_scene_digit.html`).

## Definition Of Done (Objective Gates)
- Preserve core rule: colour may help cluster discovery, but final matching/scoring remains geometry/structure driven.
- For matcher/scoring changes, validate against existing suite outputs in `test_output/vector_matching/`.
- Required checks after changes:
  - `diagnostics.json`: verify score + IoU behavior does not regress unexpectedly for documented scenarios.
  - `report.html`: verify row outcomes and pass/fail distribution are directionally consistent with expectations in `VectorMatchingTest` annotations/assertions.
- Treat unexplained drops in score/IoU or newly flipped expected outcomes as regressions until explicitly approved.

## Colour-Agnostic Guardrails (Hard Rules)
- Allowed: hue/saturation usage for cluster discovery/grouping in `SceneDescriptor` / colour-cluster extraction paths.
- Forbidden: adding colour-dependent terms to final score computation in `VectorMatcher` or signature similarity in `VectorSignature`.
- Forbidden: selecting final winning candidate primarily by hue/colour identity instead of structural similarity.
- If a score term uses colour metadata, document why geometry-only alternatives are insufficient and get explicit approval before merging.

## Maintenance Rules
- **`vectormatcher/README.md` must stay current.** After any change to files in `src/main/java/org/example/matchers/vectormatcher/` or its `components/` sub-package — including adding/removing/renaming classes, changing constants, altering pipeline stages, modifying scoring formulas, or changing data model fields — update `src/main/java/org/example/matchers/vectormatcher/README.md` to reflect the new state. Sections to check: Package Map, Pipeline Architecture, Three-Layer Scoring, Data Model, Constants Reference, Performance Optimisations, and any affected prose.
- Treat a stale README as a documentation regression: verify the README matches the code before marking a task as done.

## Regression Targets / Baselines
- Use expectations already documented in `VectorMatchingTest` as the baseline contract for this suite.
- Self-match baseline: single-colour and multi-colour self scenarios should remain aligned with existing threshold/assert behavior (including documented PARTIAL cases).
- Diagnostic matrix baseline: the full shape × background matrix (`runDiagnosticMatrix`) was removed; every `assertBgMatch`/`assertSelfMatch`/`recordBgMatch` call now records into both `report` and `diag` directly — individual tests ARE the diagnostic source. Remain near the documented result band and known false-positive profile across those individual tests.
- Rotation robustness baseline: remain near the documented pass band and known AR-sensitive failure profile in `runRotationRobustness()`.
- Any result outside these documented bands/profiles is a regression unless explicitly approved.
