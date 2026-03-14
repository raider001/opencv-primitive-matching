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

> `SceneEntry` eagerly builds a `SceneDescriptor` (colour-cluster contours via `SceneColourClusters`) on construction. Matchers that exploit scene structure read it via `scene.descriptor()`.

**Two entry points share the same runtime logic:**
- `BenchmarkLauncher` — Swing GUI wizard (`src/main/java/.../ui/BenchmarkLauncher.java`)
- `AnalyticalTestBase` — JUnit 5 base class (`src/test/java/.../utilities/AnalyticalTestBase.java`)

## Adding a New Matcher

1. Create `src/main/java/org/example/matchers/XyzMatcher.java` + `XyzVariant.java` (enum implementing `MatcherVariant`).
2. Register in `MatcherRegistry.ALL` (`src/main/java/org/example/MatcherRegistry.java`).
3. Add a `TechniqueReport` entry to `BenchmarkReportRunner.KNOWN_REPORTS`.
4. Create a test class extending `AnalyticalTestBase` implementing the **seven** abstract methods (`tag`, `techniqueName`, `outputDir`, `debugMode`, `debugRef`, `saveVariants`, `runMatcher`).

Variant names **must** follow the convention `<BASE>_CF_LOOSE` / `<BASE>_CF_TIGHT` for colour-filtered variants — this suffix is parsed at runtime by `cfTierFilter()`.

> **No `AnalyticalTestBase` subclasses exist yet** for any of the 13 registered matchers. The first focused matcher test (`vectormatcher/VectorMatchingTest` and `VectorMatcherDiagnosticTest`) instead uses the standalone `MatchReportLibrary` / `MatchDiagnosticLibrary` utilities (`src/test/java/org/example/utilities/`) which produce their own `report.html` and `diagnostics.json`. Use `AnalyticalTestBase` for full-catalogue benchmark runs; use `MatchReportLibrary`/`MatchDiagnosticLibrary` for focused per-shape diagnostic tests.

> **Exception — `MorphologyAnalyzer`**: uses string constants `VAR_POLY` / `VAR_CIRC` / `VAR_COMBINED` instead of a variant enum; `MatcherRegistry` calls `morphVariantNames()` to build the variant set programmatically.

> **Exception — `CF1` / `MCF1`**: these Colour-First region-proposal entries are **not** backed by a `MatcherDescriptor` in `MatcherRegistry.ALL`. Add only a `TechniqueReport` entry to `BenchmarkReportRunner.KNOWN_REPORTS` for dashboard collation; do **not** add to `MatcherRegistry.ALL`.

## Colour Pre-Filter (CF) System

Every matcher is expected to run three CF modes declared in `CfMode`:
- `NONE` — raw scene/reference, no colour processing
- `LOOSE` — ±15° hue window via `ColourPreFilter.LOOSE`
- `TIGHT` — ±8° hue window via `ColourPreFilter.TIGHT`

Red/orange hues (near H=0/179 in OpenCV) are wrapped automatically inside `ColourPreFilter.apply()`.

`SceneColourClusters` decomposes a BGR scene into up to `MAX_CLUSTERS=12` chromatic hue bands plus `BRIGHT_ACHROMATIC` and `DARK_ACHROMATIC` buckets; used to build the per-entry `SceneDescriptor` (contours grouped by colour cluster).

`ColourFirstLocator` proposes candidate search windows via HSV thresholding — **single-colour** (Milestone 15, CF1 report) and **multi-colour** (Milestone 21, MCF1 report, uses multiple per-channel passes then merges). Falls back to a full-scene rect when no candidates survive filtering.

## Scene Catalogue

Scenes live at `test_output/catalogue_samples/` as `<name>.png` + `<name>.json` pairs loaded by `SceneCatalogueLoader`. Four categories:

| Category | Description |
|---|---|
| `A_CLEAN` | Reference centred, no transform |
| `B_TRANSFORMED` | Scaled and/or rotated, repositioned |
| `C_DEGRADED` | Noise, blur, occlusion, hue shift — **defined in `SceneCategory` but not generated**; `SceneGenerator` has no `buildCategoryC()` and `SceneCatalogue.build()` includes only A, B, D |
| `D_NEGATIVE` | No reference present (false-positive probe) |

Full catalogue: ~1540 scenes (A=352, B=1144, D=44). Slim catalogue: 187 scenes (93×A + 93×B + 1×D; increased from 177 when Milestone 21 added 5 multi-colour refs). Debug catalogue: 3 scenes (instant).
Reference images: 93 synthetic 128×128 px BGR shapes in `test_output/references/` — 88 single-colour shapes plus 5 multi-colour shapes (`BICOLOUR_*`, `TRICOLOUR_*`) added in Milestone 21 for MCF1 validation.

`SceneGenerator.buildMultiShape()` produces 8 hand-crafted demo scenes containing 2–4 reference shapes each in non-overlapping quadrant positions. These scenes are **not** part of the A/B/C/D catalogue — they exist for human inspection and multi-placement metadata validation only.

## Developer Workflows

```bash
# Compile
mvn clean compile

# Run all tests (Surefire already sets --enable-native-access=ALL-UNNAMED)
mvn test

# Run one test class (use an existing class, e.g. VectorMatchingTest or SceneCatalogueTest)
mvn test -Dtest=VectorMatchingTest

# Build shaded fat-JAR → target/PatternMatching-all.jar
mvn package

# Launch the Swing GUI
mvn exec:java -Dexec.mainClass="org.example.Main"
```

> **OpenCV native library**: always call `OpenCvLoader.load()` in `@BeforeAll`. Uses `nu.pattern.OpenCV.loadShared()` — no `-Djava.library.path` needed.

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

