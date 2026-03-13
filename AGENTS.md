# AGENTS.md — Pattern Matching Benchmark

## Architecture Overview

This is a **multi-technique computer vision benchmark** built on OpenCV 4.7.0 (via `org.openpnp:opencv`). It evaluates 15 matching algorithms against a synthetic image catalogue and produces per-technique HTML reports unified into a single benchmark dashboard.

**Core data flow:**
```
ReferenceId enum → ReferenceImageFactory (128×128 BGR Mat)
    → MatcherDescriptor.run(refMat, SceneEntry, ...) → List<AnalysisResult>
        → AnalysisOutputWriter (JSON sidecars) → HtmlReportWriter (report.html)
            → BenchmarkReportRunner (unified test_output/benchmark/report.html)
```

**Two entry points share the same runtime logic:**
- `BenchmarkLauncher` — Swing GUI wizard (`src/main/java/.../ui/BenchmarkLauncher.java`)
- `AnalyticalTestBase` — JUnit 5 base class (`src/test/java/.../utilities/AnalyticalTestBase.java`)

## Adding a New Matcher

1. Create `src/main/java/org/example/matchers/XyzMatcher.java` + `XyzVariant.java` (enum implementing `MatcherVariant`).
2. Register in `MatcherRegistry.ALL` (`src/main/java/org/example/MatcherRegistry.java`).
3. Add a `TechniqueReport` entry to `BenchmarkReportRunner.KNOWN_REPORTS`.
4. Create a test class extending `AnalyticalTestBase` implementing the five abstract methods (`tag`, `techniqueName`, `outputDir`, `debugMode`, `debugRef`, `saveVariants`, `runMatcher`).

Variant names **must** follow the convention `<BASE>_CF_LOOSE` / `<BASE>_CF_TIGHT` for colour-filtered variants — this suffix is parsed at runtime by `cfTierFilter()`.

## Colour Pre-Filter (CF) System

Every matcher is expected to run three CF modes declared in `CfMode`:
- `NONE` — raw scene/reference, no colour processing
- `LOOSE` — ±15° hue window via `ColourPreFilter.LOOSE`
- `TIGHT` — ±8° hue window via `ColourPreFilter.TIGHT`

Red/orange hues (near H=0/179 in OpenCV) are wrapped automatically inside `ColourPreFilter.apply()`.

## Scene Catalogue

Scenes live at `test_output/catalogue_samples/` as `<name>.png` + `<name>.json` pairs loaded by `SceneCatalogueLoader`. Four categories:

| Category | Description |
|---|---|
| `A_CLEAN` | Reference centred, no transform |
| `B_TRANSFORMED` | Scaled and/or rotated, repositioned |
| `C_DEGRADED` | Noise, blur, occlusion, hue shift |
| `D_NEGATIVE` | No reference present (false-positive probe) |

Full catalogue: ~1540 scenes. Slim catalogue: 177 scenes. Debug catalogue: 3 scenes (instant).
Reference images: 47 synthetic 128×128 px BGR shapes in `test_output/references/`.

## Developer Workflows

```bash
# Compile
mvn clean compile

# Run all tests (Surefire already sets --enable-native-access=ALL-UNNAMED)
mvn test

# Run one test class
mvn test -Dtest=TemplateMatchingTest

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
  references/          ← 47 reference PNGs (generated once, reused)
  catalogue_samples/   ← PNG + JSON scene pairs (source of truth for tests)
  <technique>/         ← per-matcher annotated images + report.html
  benchmark/           ← unified report.html (Milestone 20)
```

## Score Tiers

`AnalysisResult.matchScoreEmoji()`: 🟢 ≥ 70 % · 🟡 ≥ 40 % · 🔴 < 40 % · 💥 error.

