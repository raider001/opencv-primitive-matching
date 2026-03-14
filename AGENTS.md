# AGENTS.md — Pattern Matching Benchmark

## Architecture Overview

This is a **multi-technique computer vision benchmark** built on OpenCV 4.7.0 (via `org.openpnp:opencv`), compiled with **Java 25** (`maven.compiler.source=25` in `pom.xml`). `MatcherRegistry.ALL` registers **13 matchers**; `BenchmarkReportRunner.KNOWN_REPORTS` adds `CF1` and `MCF1` (Colour-First region-proposal) entries for a total of 15 benchmark entries. Per-technique HTML reports are unified into a single benchmark dashboard.

**Core data flow:**
```
ReferenceId enum → ReferenceImageFactory (128×128 BGR Mat)
    → MatcherDescriptor.run(refMat, SceneEntry[+SceneDescriptor], ...) → List<AnalysisResult>
        → DetectionVerdict.evaluate() + PerformanceProfiler.profileAll()
        → ResultMetadataStore (JSON sidecars, incremental resume) → HtmlReportWriter (report.html)
            → BenchmarkReportRunner (unified test_output/benchmark/report.html)
```

> `SceneEntry` eagerly builds a `SceneDescriptor` (colour-cluster contours via `SceneColourClusters`) on construction. Matchers that exploit scene structure read it via `scene.descriptor()`. `SceneDescriptor` owns a native `combinedChromaticMask` Mat — always free via `SceneEntry.release()` (or `SceneDescriptor.release()` directly).

> `SceneEntry.placements()` returns a `List<SceneShapePlacement>` — one ground-truth record per placed shape, empty for Cat D. `SceneShapePlacement` (`src/main/java/org/example/scene/SceneShapePlacement.java`) carries `referenceId`, `placedRect`, `scaleFactor`, `rotationDeg`, `offsetX/Y`, `colourShifted`, `occluded`, and `occlusionFraction`. `SceneEntry.primaryReferenceId()` returns the `ReferenceId` of the first placed shape (null for Cat D) — the primary key used by `AnalyticalTestBase` and `SceneCatalogueLoader` for reference filtering. `SceneEntry.groundTruthRect()` is shorthand for `placements().get(0).placedRect()`. `SceneEntry.stub(refId, category, bgId, variantLabel, groundTruthRect)` constructs a mat-free entry used when reloading results from JSON sidecars (incremental mode).

**Two entry points share the same runtime logic:**
- `BenchmarkLauncher` — Swing GUI wizard (`src/main/java/.../ui/BenchmarkLauncher.java`)
- `AnalyticalTestBase` — JUnit 5 base class (`src/test/java/.../utilities/AnalyticalTestBase.java`)

**Swing UI class structure (`org.example.ui`):**

| Class | Role |
|---|---|
| `RunConfiguration` | Single selection model; all 5 wizard panels read from and write to this one object. Also holds two run options: `clearPrevious` (wipe output dir on start) and `includeNegatives` (include Cat D scenes) — exposed as checkboxes in the launcher's top bar. |
| `WizardContext` | Shared helpers: catalogue file scanning, expected-total arithmetic, matcher list |
| `SelectionTable` | Reusable dark-theme checkbox table (used in every wizard panel) |
| `Palette` | Colour and font constants (dark-mode GitHub-style palette) |
| `Widgets` | Static factory for consistently styled Swing components |

Wizard panels (`org.example.ui.panels`), in order: `MatchersPanel` → `VariantsPanel` → `ReferencesPanel` → `BackgroundsPanel` → `ScenesPanel`, plus read-only `RunSummaryPanel`. `ProgressPanel` (also in `org.example.ui.panels`) is the full-screen live run monitor — per-matcher progress bars, ETA, cancel button — swapped in via card layout in `BenchmarkLauncher` to replace `RunSummaryPanel` while a run is active. To add a new step: create a panel extending `WizardStepPanel`, add it in `BenchmarkLauncher`'s constructor, and update `RunConfiguration` with matching selection state.

## Adding a New Matcher

1. Create `src/main/java/org/example/matchers/XyzMatcher.java` + `XyzVariant.java` (enum implementing `MatcherVariant`).
2. Register in `MatcherRegistry.ALL` (`src/main/java/org/example/MatcherRegistry.java`).
3. Add a `TechniqueReport` entry to `BenchmarkReportRunner.KNOWN_REPORTS`.
4. Create a test class extending `AnalyticalTestBase` implementing the **seven** abstract methods (`tag`, `techniqueName`, `outputDir`, `debugMode`, `debugRef`, `saveVariants`, `runMatcher`).

Variant names **must** follow the convention `<BASE>_CF_LOOSE` / `<BASE>_CF_TIGHT` for colour-filtered variants — this suffix is parsed at runtime by `cfTierFilter()`.

> **Annotated images**: matchers are responsible for writing annotated PNGs themselves (see `TemplateMatcher` for the `Imgcodecs.imwrite` pattern). `AnalysisOutputWriter.saveAnnotatedImages()` is a no-op kept for API compatibility — it does nothing. Images go to `<outputDir>/annotated/<variantName>/`.

> **No `AnalyticalTestBase` subclasses exist yet** for any of the 13 registered matchers. The first focused matcher test (`vectormatcher/VectorMatchingTest` and `VectorMatcherDiagnosticTest`) instead uses the standalone `MatchReportLibrary` / `MatchDiagnosticLibrary` utilities (`src/test/java/org/example/utilities/`) which produce their own `report.html` and `diagnostics.json`. Use `AnalyticalTestBase` for full-catalogue benchmark runs; use `MatchReportLibrary`/`MatchDiagnosticLibrary` for focused per-shape diagnostic tests.

> **Exception — `VectorMatcher`**: Although `VectorVariant` declares `VECTOR_STRICT`, `VECTOR_NORMAL`, and `VECTOR_LOOSE`, the matcher always executes only `VECTOR_NORMAL` (ε = 4 % of perimeter). `MatcherRegistry.ALL` still registers all three names via `MatcherVariant.allNamesOf(VectorVariant.class)`, but `VectorMatcher.match()` returns a single result keyed to `VECTOR_NORMAL`.
> The matcher uses **three-layer scoring**: Layer 1 — Boundary count match (W=0.15, exponential decay); Layer 2 — Structural coherence / spatial proximity (W=0.25); Layer 3 — `VectorSignature.similarity()` on the primary boundary (W=0.60). Bright/dark achromatic cluster pairs that trace the same physical edge are deduplicated before scoring (IoU > 0.50 AND area ratio > 0.90 triggers dedup). `SegmentDescriptor` (traverses raw contour points into STRAIGHT/CURVED segments with scale-invariant length + radius ratios) is the active geometry descriptor inside `VectorSignature`; `ContourTopology` (polygon edge turn-angle descriptor) exists but is no longer the primary descriptor.
> **`VectorMatcher` diagnostic API**: `VectorMatcher.buildRefSignature(Mat refBgr, double epsilonFactor)` returns the `VectorSignature` for the primary (largest) boundary of the reference. `VectorMatcher.buildRefSignatures(Mat refBgr, double epsilonFactor)` returns one `VectorSignature` per colour cluster. Both are public static helpers used by `MatchReportLibrary` and `VectorMatcherDiagnosticTest` to introspect signature matching outside of a full scene run.

> **`@ExpectedOutcome` annotation** (`src/test/java/org/example/utilities/ExpectedOutcome.java`): informational annotation for `@Test` methods in focused diagnostic tests. Values: `PASS`, `PARTIAL`, `FAIL`, `DIAGNOSTIC`. Scanned by `MatchReportLibrary.scanTestAnnotations()` and rendered as a *Test Scenarios* table in the HTML report. Does not affect JUnit pass/fail.

> **Exception — `MorphologyAnalyzer`**: uses string constants `VAR_POLY` / `VAR_CIRC` / `VAR_COMBINED` instead of a variant enum; `MatcherRegistry` calls `morphVariantNames()` to build the variant set programmatically.

> **Exception — `CF1` / `MCF1`**: these Colour-First region-proposal entries are **not** backed by a `MatcherDescriptor` in `MatcherRegistry.ALL`. Add only a `TechniqueReport` entry to `BenchmarkReportRunner.KNOWN_REPORTS` for dashboard collation; do **not** add to `MatcherRegistry.ALL`.

> **Exception — `TemplateMatcher`**: registers **20 variants** — 6 base OpenCV methods × 3 CF tiers (18 variants) plus `TM_CCOEFF_NORMED_CF1_LOOSE` and `TM_CCOEFF_NORMED_CF1_TIGHT`. These two CF1 variants use `ColourFirstLocator` to restrict the search to a colour-proposed candidate window before running `TM_CCOEFF_NORMED` (distinct from the ordinary `_CF_LOOSE`/`_CF_TIGHT` pre-filter variants). `TmVariant.isCf1()` identifies them. The `TemplateMatcher.CF1_LOOSE` / `CF1_TIGHT` string constants are deprecated — prefer `TmVariant` enum values.

## Colour Pre-Filter (CF) System

Every matcher is expected to run three CF modes declared in `CfMode`:
- `NONE` — raw scene/reference, no colour processing
- `LOOSE` — ±15° hue window via `ColourPreFilter.LOOSE`
- `TIGHT` — ±8° hue window via `ColourPreFilter.TIGHT`

Red/orange hues (near H=0/179 in OpenCV) are wrapped automatically inside `ColourPreFilter.apply()`.

`SceneColourClusters` decomposes a BGR scene into up to `MAX_CLUSTERS=12` chromatic hue bands plus `BRIGHT_ACHROMATIC` and `DARK_ACHROMATIC` buckets; used to build the per-entry `SceneDescriptor` (contours grouped by colour cluster). `SceneDescriptor.build()` calls `SceneColourClusters.extractFromBorderPixels()` (not `extract()`) — this restricts the hue histogram to morphological-gradient border pixels, avoiding large filled interiors dominating the histogram. `HUE_TOLERANCE=14.0°`, `MIN_CONTOUR_AREA=64 px²`.

`ColourFirstLocator` proposes candidate search windows via HSV thresholding — **single-colour** (Milestone 15, CF1 report) and **multi-colour** (Milestone 21, MCF1 report, uses multiple per-channel passes then merges). Falls back to a full-scene rect when no candidates survive filtering.

## Scene Catalogue

Scenes live at `test_output/catalogue_samples/` as `<name>.png` + `<name>.json` pairs loaded by `SceneCatalogueLoader`. Four categories:

| Category | Description |
|---|---|
| `A_CLEAN` | Reference centred, no transform |
| `B_TRANSFORMED` | Scaled and/or rotated, repositioned |
| `C_DEGRADED` | Noise, blur, occlusion, hue shift — **defined in `SceneCategory` but not generated**; `SceneGenerator` has no `buildCategoryC()` and `SceneCatalogue.build()` includes only A, B, D |
| `D_NEGATIVE` | No reference present (false-positive probe) |

Full catalogue: ~1346 scenes (A=93, B=1209, D=44). `buildCategoryA()` generates 1 scene per reference (solid-black background, centred, no transform); `buildCategoryB()` generates 13 variants per reference (`scale_0.50`, `scale_0.75`, `scale_1.25`, `scale_1.50`, `scale_2.00`, `rot_15`, `rot_30`, `rot_45`, `rot_90`, `rot_180`, `offset_topleft`, `offset_botright`, `offset_random42`) cycling across 3 backgrounds (`BG_RANDOM_MIXED`, `BG_CIRCUIT_LIKE`, `BG_GRADIENT_RADIAL_COLOUR`). Slim catalogue: 187 scenes (93×A + 93×B + 1×D). Debug catalogue: 3 scenes (instant).
Reference images: 93 synthetic 128×128 px BGR shapes in `test_output/references/` — 88 single-colour shapes plus 5 multi-colour shapes (`BICOLOUR_*`, `TRICOLOUR_*`) added in Milestone 21 for MCF1 validation.

`BackgroundId` enum (`src/main/java/org/example/factories/BackgroundId.java`) defines **21 background types** across 4 complexity tiers (Tier 1: solid fills; Tier 2: gradients & light noise; Tier 3: structured patterns — grids/hatching/random lines; Tier 4: rich colour clutter — dense text, circuit-like, organic blobs). Backgrounds are generated at any resolution by `BackgroundFactory`. The `BackgroundId` is stored in each `SceneEntry` and persisted in `ResultMetadataStore` JSON sidecars.

`SceneCatalogueLoader` (`src/main/java/org/example/scene/SceneCatalogueLoader.java`) loads PNG+JSON pairs from `test_output/catalogue_samples/` in parallel. Falls back to `SceneCatalogue.build()` if the catalogue directory does not exist or yields no valid entries. Two overloads: `load(ReferenceId... filterRefs)` (reference filter only) and `load(ReferenceId[] filterRefs, SceneCategory[] filterCats)` (combined reference+category filter). Cat D negatives always pass the reference filter regardless.

`SceneGenerator.buildMultiShape()` produces 8 hand-crafted demo scenes containing 2–4 reference shapes each in non-overlapping quadrant positions. These scenes are **not** part of the A/B/C/D catalogue — they exist for human inspection and multi-placement metadata validation only.

`SceneVariant` enum (`src/main/java/org/example/scene/SceneVariant.java`) provides typed constants for all fixed variant labels (e.g. `SceneVariant.SCALE_0_75`, `SceneVariant.ROT_45`). Prefer `scene.variantLabel().equals(SceneVariant.ROT_45.label())` over raw strings inside `sceneFilter()` overrides.

## Developer Workflows

```bash
# Compile
mvn clean compile

# Run all tests (Surefire already sets --enable-native-access=ALL-UNNAMED)
mvn test

# Run one test class (use an existing class, e.g. VectorMatchingTest or SceneCatalogueTest)
mvn test -Dtest=VectorMatchingTest

# Run cross-reference rejection tests (tagged cross-reject; excluded by default)
mvn test -Dtest=VectorMatchingTest -DexcludedGroups=""

# Full shape × background diagnostic for VectorMatcher (writes diagnostics.json + report.html)
mvn test -Dtest=VectorMatcherDiagnosticTest

# Collate all per-technique report.html files into the unified benchmark dashboard
mvn test -Dtest=MatchingBenchmarkTest

# Build shaded fat-JAR → target/PatternMatching-all.jar
mvn package

# Launch the Swing GUI (actual entry point — there is no Main.java)
mvn exec:java -Dexec.mainClass="org.example.ui.BenchmarkLauncher"
```

`VectorMatcherDiagnosticTest` prints a one-line summary per run. Key classification codes:
- **FP** — score ≥ 40 % but IoU < 0.3 (wrong shape or wrong location)
- **LOW** — score ≥ 40 %, IoU ≥ 0.3, but score < 75 % (correct but low confidence)
- **BIOU** — score ≥ 40 %, IoU 0.3–0.5 (approx right area, bbox off)
- **MISS** — score < 40 % (shape not found)
- **correct** — score ≥ 75 % AND IoU ≥ 0.5

> **OpenCV native library**: always call `OpenCvLoader.load()` in `@BeforeAll`. Uses `nu.pattern.OpenCV.loadShared()` — no `-Djava.library.path` needed.

## Test Package Structure

| Package | Classes | Purpose |
|---|---|---|
| `org.example.setups` | `SmokeTest`, `ReferenceImageLibraryTest`, `BackgroundFactoryTest`, `SceneCatalogueTest`, `AnalysisInfrastructureTest`, `ColourPreFilterTest`, `ColourPreFilterVisualTest`, `ColourClusterVisualTest` | Milestone 1–6 infrastructure & visual-sanity tests |
| `org.example.vectormatcher` | `VectorMatchingTest`, `VectorMatcherDiagnosticTest` | Focused VectorMatcher tests using `MatchReportLibrary` / `MatchDiagnosticLibrary` |
| `org.example.utilities` | `AnalyticalTestBase`, `MatchReportLibrary`, `MatchDiagnosticLibrary`, `MatchingBenchmarkRunner`, `ProgressDisplay`, `ExpectedOutcome` | Shared test utilities; `MatchingBenchmarkRunner` is a thin wrapper delegating to main-scope `BenchmarkReportRunner` |
| `org.example` | `MatchingBenchmarkTest` | Collates per-technique reports into the unified benchmark dashboard |

## Test Base Class Hooks (`AnalyticalTestBase`)

Override these to control a test run without modifying core logic:

| Method | Default | Purpose |
|---|---|---|
| `debugMode()` | — (abstract) | `true` = 3-scene in-memory catalogue, single ref |
| `debugRef()` | — (abstract) | Reference used in debug mode |
| `referenceFilter()` | `[]` (all) | Restrict to specific `ReferenceId`s |
| `sceneFilter(scene)` | `true` (all) | Exclude individual scenes |
| `cfTierFilter()` | all three | Run only BASE / LOOSE / TIGHT variants |
| `incrementalMode()` | `false` | Resume a partially completed run |
| `saveVariants()` | — (abstract) | Which variants produce annotated PNGs on disk |

> The `analyseAllScenesAgainstAllReferences` test flattens all `refs × scenes` work items and runs them in a `ForkJoinPool` sized `min(32, availableProcessors − 1)`. Each (ref, scene) pair is independent — matchers must be thread-safe and must not write shared mutable state.

> `ProgressDisplay` (`src/test/java/org/example/utilities/ProgressDisplay.java`) opens a Swing window showing a live progress bar, ETA, and status line during the parallel double-loop. Falls back to plain stderr line output in headless environments (e.g. CI).

## Output Structure

```
test_output/
  references/          ← 93 reference PNGs (generated once, reused)
  catalogue_samples/   ← PNG + JSON scene pairs (source of truth for tests)
  <technique>/         ← per-matcher report.html
    annotated/
      <variant>/       ← annotated PNGs + JSON sidecars (ResultMetadataStore)
  benchmark/           ← unified report.html (Milestone 20)
  colour_prefilter/    ← ColourPreFilterTest panels (Milestone 6)
  colour_prefilter_visual/ ← ColourPreFilterVisualTest per-category panels
  cluster_visual/      ← ColourClusterVisualTest cluster overlay images
  infrastructure_test/ ← AnalysisInfrastructureTest dummy-matcher output
  backgrounds/         ← BackgroundFactoryTest generated background PNGs
  colour_first/        ← CF1 (single-colour ColourFirstLocator) output
  multi_colour_first/  ← MCF1 (multi-colour ColourFirstLocator) output
```

## Score Tiers

`AnalysisResult.matchScoreEmoji()`: 🟢 ≥ 70 % · 🟡 ≥ 40 % · 🔴 < 40 % · 💥 error.

`DetectionVerdict.evaluate()` classifies each result against ground-truth (threshold ≥ 50 %):

| Verdict | Meaning |
|---|---|
| `CORRECT` | Ref present, detected, bbox centre inside GT rect (±24 px tolerance) |
| `WRONG_LOCATION` | Ref present and score ≥ threshold but bbox centre outside GT rect |
| `MISSED` | Ref present but score below threshold |
| `FALSE_ALARM` | Ref absent (Cat D or different ref) but score ≥ threshold |
| `CORRECTLY_REJECTED` | Ref absent and score below threshold |

Localisation uses **centre-in-GT** rather than IoU — robust to scale/rotation since only the predicted bbox centre must land inside the ground-truth rect (expanded by `CENTRE_TOLERANCE_PX=24`).

