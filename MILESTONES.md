# OpenCV Pattern Matching — Development Milestones

Incremental, staged delivery. Each milestone produces working, runnable code that builds
on the previous one. No milestone starts until the previous one compiles and runs cleanly.

---

## Milestone 1 — Project Scaffold & OpenCV Smoke Test ✅ COMPLETE

**Goal:** Prove that OpenCV loads correctly, the project compiles, and `mvn test` runs.

### Deliverables
- `pom.xml` — confirmed correct: Java 25, openpnp 4.7.0-0, JUnit 5, Surefire 3.1.2
- `src/main/java/org/example/OpenCvLoader.java` — single static `load()` method wrapping
  `nu.pattern.OpenCV.loadShared()`, called once across all tests
- `src/test/java/org/example/SmokeTest.java` — one `@Test` that calls `OpenCvLoader.load()`,
  creates a 10×10 black `Mat`, and prints its size to confirm the native library is working
- `test_output/` directory created at project root (gitignored)

### Done when
- `mvn test` passes with the smoke test printing `Mat: 10x10` to console
- No native library errors on startup

---

## Milestone 2 — Reference Image Library ✅ COMPLETE

**Goal:** Generate all 88 colour reference images and visually verify them.

### Deliverables
- `src/main/java/org/example/ReferenceId.java` — `enum` with all 88 IDs,
  covering Lines (8), Circles/Ellipses (8), Rectangles (5), Regular Polygons (14),
  Polylines (12), Arcs & Partial Curves (6), Concave/Irregular Polygons (6),
  Rotated Rectangles (4), Dashed & Dotted Lines (5), Compound/Nested Shapes (7),
  Grids/Patterns (7), Text (6)
- `src/main/java/org/example/ReferenceImageFactory.java` — static `Mat build(ReferenceId)` producing
  128×128 **BGR colour** images:
  - **Foreground colours** cycle through an 8-colour palette (white, red, green, blue, yellow,
    cyan, magenta, orange) by `referenceIndex % 8`
  - **Canvas backgrounds** cycle through 4 fills by `referenceIndex % 4`:
    solid black · solid dark grey · horizontal colour gradient (dark blue→dark red) · radial gradient
  - **Gradient-fill shapes** (`CIRCLE_FILLED_GRADIENT`, `RECT_FILLED_GRADIENT`,
    `TRIANGLE_FILLED_GRADIENT`): shape interior painted with a linear colour gradient from the
    foreground colour to a lighter complementary tone using scanline `Imgproc.line` over the mask
- `src/test/java/org/example/ReferenceImageLibraryTest.java` — one `@Test` that:
  - Iterates all 47 `ReferenceId` values
  - Calls `ReferenceImageFactory.build()` for each
  - Saves each to `test_output/references/<ID>.png` for visual inspection
  - Prints a table: ID · foreground colour · background fill · width · height · non-zero pixel count
- No assertions — test always passes

### Done when
- `mvn test` produces 88 PNG files in `test_output/references/` ✅
- All 88 images are visually correct, clearly coloured, and visually distinct when opened ✅

> ⚠️ **Known complication — gradient canvas backgrounds:**
> Reference images assigned background slots 2 (H colour gradient) and 3 (radial gradient)
> carry a non-uniform background as part of their pixel data. This is **intentional and kept**
> so the test suite captures the real effect. However it is expected to cause:
> - Artificially low scores for pixel-sensitive techniques (Template Matching, Pixel Diff,
>   Histogram) when the scene background does not match the reference canvas gradient
> - Wider colour masks for `_CF` variants if the gradient hue overlaps the foreground colour
> - Spurious edge keypoints at the canvas border for Feature Matching / Hough
>
> When interpreting results, treat references with gradient backgrounds as a harder variant.
> The HTML report should flag these references with a ⚠️ in the summary table thumbnail column.

---

## Milestone 3 — Background Factory ✅ COMPLETE

**Goal:** Generate all 21 colour background types and visually verify them.

### Deliverables
- `src/main/java/org/example/BackgroundId.java` — `enum` with all 22 IDs grouped by tier (see PLAN.md):
  - Tier 1 (3): `BG_SOLID_BLACK`, `BG_SOLID_WHITE`, `BG_SOLID_GREY`
  - Tier 2 (7): greyscale H/V gradients, colour H/V gradients, greyscale radial, colour radial, light noise
  - Tier 3 (6): heavy noise, fine grid (cyan), coarse grid (blue), hatching (green), random coloured lines,
    random coloured circles
  - Tier 4 (5): random mixed coloured shapes, dense coloured text, PCB circuit (green/gold),
    organic blobs (warm/cool bands), full-frame RGB noise
- `src/main/java/org/example/BackgroundFactory.java` — static `Mat build(BackgroundId, int w, int h)`
  producing full **BGR colour** backgrounds at any requested size:
  - Solid fills via `Mat.zeros()` + `Mat.setTo()`
  - Colour gradients via row-buffered `Mat.put()` pixel assignment (H, V, and radial)
  - Coloured grid/hatching lines via `Imgproc.line` with explicit colour
  - Random coloured lines/circles/shapes via seeded `Random` + random BGR colours
  - Dense text via `Imgproc.putText` tiled with random chars in random colours
  - PCB circuit via orthogonal line segments in green + small yellow pad dots
  - Organic blobs via `Imgproc.ellipse` in warm/cool colour bands using a seeded layout
  - Full RGB noise via seeded `Random.nextBytes()` written row-by-row
- `src/test/java/org/example/BackgroundFactoryTest.java` — one `@Test` that:
  - Iterates all 21 `BackgroundId` values
  - Saves each to `test_output/backgrounds/<ID>.png`
  - Prints a table: ID · tier · complexity label · mean pixel intensity per channel (B/G/R)
- No assertions

### Done when
- `mvn test` produces 21 PNG files in `test_output/backgrounds/` ✅
- All 21 backgrounds are visually distinct, clearly coloured, and recognisable by tier ✅

---

## Milestone 4 — Scene Generator & Full Catalogue ✅ COMPLETE

**Goal:** Combine 88 colour references + 21 colour backgrounds into the full scene catalogue,
with per-scene metadata recording exactly which reference shapes are present and where.

### Deliverables
- `src/main/java/org/example/SceneCategory.java` — `enum { A_CLEAN, B_TRANSFORMED, C_DEGRADED, D_NEGATIVE }`
- `src/main/java/org/example/SceneShapePlacement.java` — record describing one shape placed into a scene:
  `{ ReferenceId referenceId, Rect placedRect, double scaleFactor, double rotationDeg,
     int offsetX, int offsetY, boolean colourShifted, boolean occluded, double occlusionFraction }`
  - `placedRect` — the bounding box of the placed (and transformed) shape within the 640×480 scene
  - Ground truth used by the HTML report to score spatial precision of match results
- `src/main/java/org/example/SceneEntry.java` — record:
  `{ ReferenceId primaryReferenceId, SceneCategory category, String variantLabel,
     BackgroundId backgroundId, List<SceneShapePlacement> placements, Mat sceneMat }`
  - `primaryReferenceId` is `null` for Category D (negative) scenes
  - `placements` is an empty list for Category D scenes
  - Every Category A/B/C scene has exactly one entry in `placements` describing where the
    reference was placed and what transform was applied
- `src/main/java/org/example/SceneGenerator.java` — builds all four categories:
  - **Cat A** (352): 88 refs × 4 backgrounds — one from each complexity tier
    (`BG_SOLID_BLACK`, `BG_GRADIENT_H_COLOUR`, `BG_NOISE_LIGHT`, `BG_RANDOM_MIXED`)
  - **Cat B** (1320): 88 refs × 15 transform variants on colour backgrounds cycling
    (`BG_RANDOM_MIXED`, `BG_CIRCUIT_LIKE`, `BG_GRADIENT_RADIAL_COLOUR` by `variantIndex % 3`)
  - **Cat C** (616): 88 refs × 7 degradation variants on high-complexity colour backgrounds
    cycling (`BG_DENSE_TEXT`, `BG_CIRCUIT_LIKE`, `BG_ORGANIC`, `BG_RANDOM_MIXED`,
    `BG_COLOURED_NOISE` by `variantIndex % 5`); includes a **colour shift** variant
    (+40 hue rotation applied to reference via HSV manipulation)
  - **Cat D** (45): one scene per BackgroundId (21) + 5 coloured noise seeds + 10 random
    coloured shape scenes + 5 circuit decoys + 3 dense text sizes — no placements, no reference
- `src/main/java/org/example/SceneCatalogue.java` — singleton: `build()` calls all factories once
  and caches a `List<SceneEntry>` of 2,332 entries
- `src/test/java/org/example/SceneCatalogueTest.java` — one `@Test` that:
  - Builds the catalogue and prints counts per category
  - Prints a metadata spot-check table: 12 random scenes, showing variantLabel,
    backgroundId, placement count, placedRect, scaleFactor, rotationDeg
  - Saves 2 sample scenes per category (with placement rect drawn as an overlay) to
    `test_output/catalogue_samples/`
  - Saves a 6×6 colour thumbnail contact sheet to `test_output/catalogue_samples/contact_sheet.png`
- No assertions

### Done when
- `mvn test` prints `Catalogue built: 2332 scenes` and produces sample colour images
- Scene counts match: A=352, B=1320, C=616, D=44
- Metadata spot-check shows all A/B/C scenes have exactly 1 placement with valid rect + transform
- All D scenes show 0 placements
- Contact sheet clearly shows colour variety across backgrounds and reference colours

---

## Milestone 5 — Analysis Infrastructure ✅ COMPLETE

**Goal:** Build the shared result model, output writer, performance profiler, and HTML report
writer used by all technique tests. Verified with a dummy matcher before any real matching
logic exists.

### Deliverables
- `src/main/java/org/example/AnalysisResult.java` — record:
  `{ String methodName, ReferenceId referenceId, String variantLabel, SceneCategory category,
     BackgroundId backgroundId, double scorePercent, Rect boundingRect, long elapsedMs,
     int scenePx, Mat annotatedMat, boolean isError, String errorMessage }`
- `src/main/java/org/example/PerformanceProfile.java` — record:
  `{ String methodVariant, long minMs, long maxMs, double avgMs, long p95Ms, double msPerMp,
     Map<String,double[]> projectedMs,   // key = "720p"/"1080p"/"1440p"/"4K", value=[linear,quadratic]
     Map<String,Double> estimatedHeapMb  // key = resolution label
  }`
- `src/main/java/org/example/PerformanceProfiler.java`:
  - `profile(List<AnalysisResult>, String methodVariant)` → `PerformanceProfile`
    - Aggregates min / max / avg / P95 `elapsedMs` across all scenes for the variant
    - Computes `msPerMp = elapsedMs / (scenePx / 1_000_000)`
    - Projects to 720p, 1080p, 1440p, 4K using linear (`×MP ratio`) and quadratic (`×MP ratio²`) models
    - Estimates working heap MB per resolution using per-algorithm memory multipliers:

      | Algorithm family | Multiplier |
      |------------------|------------|
      | Template Matching | 1.3× |
      | Feature Matching | 3.0× |
      | Hu Moments / Contour | 1.1× |
      | Hough Transforms | 1.5× |
      | Generalized Hough | 4.0× |
      | Histogram | 1.2× |
      | Phase Correlation | 4.0× |
      | Morphology | 1.2× |
      | Pixel Diff | 2.0× |

  - `interpretationNote(PerformanceProfile)` → human-readable string:
    e.g. _"Averages 4.2 ms at 640×480. Projects to ~28 ms at 1080p (~36 fps) and ~112 ms at
    4K (~9 fps). Suitable for real-time use up to 1080p."_
- `src/main/java/org/example/AnalysisOutputWriter.java`:
  - `saveAnnotatedImages(List<AnalysisResult>, Path outputDir)`
  - `saveReferenceGrids(List<AnalysisResult>, Path outputDir)`
  - `printAsciiTable(List<AnalysisResult>)`
  - `printPerformanceTable(List<PerformanceProfile>)` — ASCII table to console
- `src/main/java/org/example/HtmlReportWriter.java`:
  - `write(List<AnalysisResult>, List<PerformanceProfile>, String techniqueName, Path reportPath)`
  - Produces a self-contained `report.html` with:
    - **Results tab**: summary table (avg score per category), colour-coded 🟢🟡🔴,
      expandable per-reference accordion sections with inline scene grids
    - **Reference thumbnails** in the summary table flagged with ⚠️ when the reference ID
      was generated with a gradient canvas background (slots 2 & 3 of the background cycle),
      with a tooltip: _"This reference has a gradient canvas background — scores for
      pixel-sensitive techniques may be lower than the shape difficulty alone would suggest."_
    - **Performance tab**:
      - Timing summary table: Min · Max · Avg · P95 ms · ms/MP per variant
      - Resolution projection table: avg ms at 720p / 1080p / 1440p / 4K (linear + quadratic)
      - Memory estimation table: scene MB + working MB per resolution
      - HTML/CSS bar chart showing projected ms per variant at each resolution
      - Auto-generated interpretation notes (fps estimates, real-time suitability)
    - All images embedded as base64 data URIs — fully self-contained, no server needed
- `src/test/java/org/example/AnalysisInfrastructureTest.java` — one `@Test` using a dummy matcher
  (returns random scores 0–100%, random elapsed ms 1–50) over 5 references × 10 scenes:
  - Exercises `AnalysisOutputWriter`, `PerformanceProfiler`, `HtmlReportWriter`
  - Saves to `test_output/infrastructure_test/`

### Done when
- `mvn test` produces `test_output/infrastructure_test/report.html`
- Report opens in a browser with both Results and Performance tabs rendering correctly
- Performance tab shows projected timings at all 4 resolutions, memory table, bar chart,
  and interpretation notes

---

## Milestone 6 — Colour Pre-Filter Infrastructure ✅ COMPLETE

**Goal:** Build the colour isolation pre-processing layer that wraps every base technique as a
`_CF_LOOSE` and `_CF_TIGHT` variant.  Verified standalone before any matcher uses it.

### Design principle — with AND without colour filter for every technique

Every technique from Milestone 7 onward is run in **three modes** in a single test pass:

| Mode | Suffix | What it does |
|------|--------|--------------|
| **Base** | *(none)* | Runs directly on the full-colour scene and reference |
| **CF_LOOSE** | `_CF_LOOSE` | Applies colour pre-filter (±15° hue tolerance) before matching |
| **CF_TIGHT** | `_CF_TIGHT` | Applies colour pre-filter (±8° hue tolerance) before matching |

This means:
- **Every** `report.html` (Milestones 7–15) contains results for all three modes side-by-side
- The **Base vs CF Comparison** sub-tab in every report shows the delta (CF − base) per scene
  category, making it easy to see exactly where colour filtering helps vs. hurts
- The Performance tab tracks `preFilterElapsedMs` separately for CF variants, so the overhead
  cost is always visible alongside the accuracy benefit
- The **Unified Benchmark** (Milestone 16) aggregates this three-way comparison across all
  nine technique families

### Deliverables
- `src/main/java/org/example/ColourRange.java` — record:
  `{ Scalar hsvLower, Scalar hsvUpper, String label }`
- `src/main/java/org/example/ColourPreFilter.java`:
  - `apply(Mat bgrImage, Scalar hsvLower, Scalar hsvUpper)` → `Mat` (binary mask CV_8UC1)
    - Converts BGR → HSV via `Imgproc.cvtColor`
    - Applies `Core.inRange(hsv, lower, upper, mask)`
    - Returns the binary mask (white = within colour range, black = outside)
  - `applyToScene(Mat bgrScene, Mat bgrReference, ReferenceId id, double hueTolerance)` → `Mat`
    - Convenience method: extracts the colour range from the reference ID then applies it to
      the scene; used directly by matcher `_CF` variants
  - `extractReferenceColourRange(ReferenceId id, double hueTolerance)` → `ColourRange`
    - Calls `ReferenceImageFactory.foregroundColour(id)` to get the known BGR foreground
    - Converts to HSV, builds `hsvLower`/`hsvUpper` with `±hueTolerance` on Hue,
      `±40` on Saturation, `±40` on Value (clamped to valid HSV ranges 0–179, 0–255)
    - Handles red hue wrap-around (H≈0° in OpenCV, wraps at 180) with two `Core.inRange`
      calls merged via `Core.bitwise_or`
  - Two tolerance preset constants:
    - `LOOSE = 15.0` (±15° hue)
    - `TIGHT =  8.0` (±8° hue)
- `src/test/java/org/example/ColourPreFilterTest.java` — one `@Test` that:
  - Iterates all 88 `ReferenceId` values
  - For each, calls `extractReferenceColourRange` at both LOOSE and TIGHT tolerances
  - Applies `apply()` to the reference image itself (should produce a mostly-white mask)
  - Applies `apply()` to a plain black scene (should produce an all-black mask)
  - Saves a 4-panel side-by-side image:
    `[original reference | loose mask | tight mask | black-scene mask (all black)]`
    to `test_output/colour_prefilter/<ID>.png`
  - Prints a table: ID · fg colour · loose white-pixel % · tight white-pixel % · red-wrap?
- No assertions

### Done when
- `mvn test` produces 88 side-by-side PNGs in `test_output/colour_prefilter/`
- Reference images produce high white-pixel % on their own loose/tight masks (shape visible)
- Plain black scene masks are 0% white pixels (no false detections)
- Red/orange references handled correctly via hue wrap-around
- White reference (slot 0: low saturation) produces low mask coverage — expected and noted

---

## Milestone 7 — Technique: Template Matching ✅ COMPLETE

**Goal:** Implement and analyse the first real matching technique, including CF variants.

### Deliverables
- `src/main/java/org/example/matchers/TemplateMatcher.java`:
  - Implements all 6 `TM_*` base method variants
  - Implements `_CF_LOOSE` and `_CF_TIGHT` variants for each (12 total CF variants + 6 base = 18)
  - CF variants call `ColourPreFilter.apply()` on both scene and reference before `matchTemplate`;
    `preFilterElapsedMs` is recorded separately
  - Returns `List<AnalysisResult>` — one result per variant
  - Normalises all scores to 0–100% (inverts distance metrics)
- `src/test/java/org/example/TemplateMatchingTest.java`:
  - `@BeforeAll` — loads OpenCV, builds `SceneCatalogue`
  - `@Test analyseAllScenesAgainstAllReferences()` — double loop: 47 refs × 1,267 scenes
  - `@AfterAll` — calls `PerformanceProfiler.profile()` per variant, then writes:
    - Annotated PNGs + reference grid PNGs via `AnalysisOutputWriter`
    - ASCII results table + ASCII performance table to console
    - `test_output/template_matching/report.html` with Results + Performance tabs
    - Report includes a **Base vs CF comparison sub-tab** showing side-by-side accuracy and
      timing for each base variant against its `_CF_LOOSE` and `_CF_TIGHT` counterparts

> **CF + Performance note (applies to Milestones 7–15):**
> - Every technique test runs both base AND CF variants in the same double loop
> - `preFilterElapsedMs` is tracked and reported separately in the Performance tab
> - The HTML report includes a **Base vs CF comparison sub-tab** per technique
> - The Performance tab shows pre-filter overhead (ms) at each resolution projection

### Done when
- `mvn test` completes, `report.html` shows Results, Performance, and Base vs CF tabs
- CF variants show improved accuracy on coloured shapes against contrasting backgrounds (Cat A/B)
- CF variants show reduced accuracy on colour-shifted scenes (Cat C colour shift variant)
- Performance tab shows pre-filter overhead is < 1 ms at 640×480

---

## Milestone 8 — Technique: Feature Matching (SIFT / ORB / AKAZE / BRISK / KAZE) ✅ COMPLETE

### Deliverables
- `src/main/java/org/example/matchers/FeatureMatcher.java`:
  - 5 base variants (SIFT, ORB, AKAZE, BRISK, KAZE)
  - `_CF_LOOSE` and `_CF_TIGHT` for each (15 CF variants)
  - Uses `BFMatcher` + Lowe's ratio test (0.75) + `findHomography` RANSAC
  - CF variants apply mask before keypoint detection — restricts keypoints to masked region
  - Score = inlier match count normalised to 0–100% relative to reference keypoint count
- `src/test/java/org/example/FeatureMatchingTest.java` — same structure as Milestone 7

### Done when
- Report shows SIFT/AKAZE outperforming ORB on transformed + degraded scenes
- CF variants show reduced false positives on complex coloured backgrounds

---

## Milestone 9 — Technique: Contour Shape Matching (Hu Moments) ✅ COMPLETE

### Deliverables
- `src/main/java/org/example/matchers/ContourShapeMatcher.java`:
  - 3 base variants: `CONTOURS_MATCH_I1`, `I2`, `I3`
  - `_CF_LOOSE` and `_CF_TIGHT` for each (6 CF variants)
  - Full-scene `findContours` + per-contour `matchShapes` against rendered reference binary;
    best contour wins, bbox expanded via `unionNearbyContours` to cover arc fragments
  - Score = `1 / (1 + matchShapesValue)` → 0–100%; `renderContour` Mat overload used for
    numerical stability (avoids Infinity from I1 on smooth contours)
- `src/test/java/org/example/ContourShapeMatchingTest.java`

### Done when
- `mvn test` produces `test_output/contour_shape_matching/report.html` ✅
- Bounding box tightly wraps detected shape (fragment union handles arc splits) ✅
- D_NEGATIVE scenes score 0%; clean circle scores 83%+; CF_LOOSE B_TRANSFORMED scores 96%+ ✅

---

## Milestone 10 — Technique: Hough Transforms ✅ COMPLETE

### Deliverables
- `src/main/java/org/example/matchers/HoughDetector.java`:
  - 2 base variants: `HoughLinesP` + `HoughCircles`
  - `_CF_LOOSE` and `_CF_TIGHT` for each (4 CF variants)
  - CF reduces background line/circle noise significantly — expected to show largest improvement
  - `HoughLinesP` score = line-count ratio (scene vs reference); bbox spans all segments
  - `HoughCircles` score = radius-match ratio (70%) + count ratio (30%); bbox = best-circle rect
- `src/test/java/org/example/HoughDetectorTest.java`

### Done when
- `mvn test` produces `test_output/hough_transforms/report.html` ✅
- `HoughCircles` clean scene: 99.1%; CF_LOOSE B_TRANSFORMED: 83.3% (vs base 69.9%) ✅
- CF reduces `HoughCircles` time from 1230ms → 73ms on complex backgrounds (17× faster) ✅
- D_NEGATIVE scenes score 0% ✅

---

## Milestone 11 — Technique: Generalized Hough Transform ✅ COMPLETE

### Deliverables
- `src/main/java/org/example/matchers/GeneralizedHoughDetector.java`:
  - 2 base variants: `GeneralizedHoughBallard` (dp=2, coarse) + `GeneralizedHoughGuil` (dp=1, fine)
  - `_CF_LOOSE` and `_CF_TIGHT` for each (4 CF variants)
  - Score = peak vote count / template edge pixel count × 100%, capped at 100%
  - Note: True `GeneralizedHoughGuil` (rotation+scale) is impractically slow at 640×480 in
    OpenCV 4.7 Java; `VAR_GUIL` is implemented as a finer-resolution Ballard (dp=1) to provide
    a meaningful contrast while completing in <2s. Documented clearly in class Javadoc.
- `src/test/java/org/example/GeneralizedHoughTest.java`

### Done when
- `mvn test` produces `test_output/generalized_hough/report.html` ✅
- Clean scene: 100% for both Ballard and Guil variants ✅
- D_NEGATIVE: 0.0% for all 6 variants ✅
- CF variants ~8× faster on complex backgrounds (297ms → 34ms) ✅

---

## Milestone 12 — Technique: Histogram Comparison ✅ COMPLETE

### Deliverables
- `src/main/java/org/example/matchers/HistogramMatcher.java`:
  - 4 base variants: `HISTCMP_CORREL`, `HISTCMP_CHISQR`, `HISTCMP_INTERSECT`, `HISTCMP_BHATTACHARYYA`
  - `_CF_LOOSE` and `_CF_TIGHT` for each (8 CF variants = 12 total)
  - 2-D HSV Hue–Saturation histogram (H=50 bins, S=60 bins); V channel ignored
  - Sliding window (stride=8) provides spatial localisation for bbox
  - Each method normalised to 0–100%: CORREL (val+1)/2×100, CHISQR 1/(1+val)×100,
    INTERSECT val/refSum×100, BHATTACHARYYA (1-val)×100
- `src/test/java/org/example/HistogramMatchingTest.java`

### Done when
- `mvn test` produces `test_output/histogram_matching/report.html` ✅
- 36 results (4 methods × 3 modes × 3 scenes) ✅
- Clean scene: 100% for all 12 variants ✅
- D_NEGATIVE: 100% — confirms the expected weakness: pure colour histogram has NO spatial
  awareness; a black scene with white crops scores perfectly because both contain ~0 colour
  pixels. This is the documented limitation and exactly what the report should highlight. ✅

---

## Milestone 13 — Technique: Phase Correlation ✅ COMPLETE

### Deliverables
- `src/main/java/org/example/matchers/PhaseCorrelationMatcher.java`:
  - 2 base variants × base / CF_LOOSE / CF_TIGHT = **6 variants**
  - Manual DFT implementation (`Core.dft` / `Core.mulSpectrums` / `Core.idft` / `Core.minMaxLoc`)
    — `Core.phaseCorrelate` is not exposed in openpnp opencv-4.7.0-0 Java binding
  - Hanning window built manually with separable 2-D von Hann formula
  - Score = IDFT peak value × 100 (∈ [0,1] after normalised cross-power spectrum)
  - Sliding window stride=8 for localisation; sub-pixel shift from DFT peak refines bbox
- `src/test/java/org/example/PhaseCorrelationTest.java`

### Done when
- `mvn test` produces `test_output/phase_correlation/report.html` ✅
- 18 results (2 methods × 3 modes × 3 scenes) ✅
- A_CLEAN: **100%** for all 6 variants ✅
- D_NEGATIVE: **0.0%** for all 6 variants ✅ (correctly rejects wrong shape)
- B_TRANSFORMED: ~9–12% — correctly low; the scaled+rotated scene does NOT produce
  the same FFT peak as the exact reference because phase correlation is translation-only ✅
- Performance: ~3.6s per scene (3600ms) — very slow due to 4225 DFT calls per variant;
  this accurately reflects phase correlation's cost at 640×480 with stride=8

---

## Milestone 14 — Technique: Morphology Analysis ✅ COMPLETE

### Deliverables
- `src/main/java/org/example/matchers/MorphologyAnalyzer.java`:
  - 3 base variants: `approxPolyDP` vertex count, circularity ratio, combined descriptor
  - `_CF_LOOSE` and `_CF_TIGHT` for each (6 CF variants)
  - Score = geometric similarity to reference shape descriptors → 0–100%
- `src/test/java/org/example/MorphologyAnalysisTest.java`

---

## Milestone 15 — Colour-First Region Proposal

**Goal:** Implement a generic colour-first pipeline that can front any of the 9 base
techniques. A colour threshold proposes candidate bounding boxes; the matcher only runs
inside those boxes — not across the full scene.

### Motivation

The current `_CF_LOOSE` / `_CF_TIGHT` variants:
- Mask the **full scene** then run the matcher over the entire masked frame
- Gain colour discrimination but the matcher still searches the whole image

The **colour-first approach** does the opposite:
1. Apply HSV threshold → binary mask
2. `findContours` on the mask → candidate blobs
3. Merge/pad each blob to ≥ reference size, clamp to scene bounds
4. Run the matcher **only inside each candidate window**
5. Return the best-scoring window

This is ~10–50× faster on cluttered scenes because typically only 1–5 small regions
are searched. On a D_NEGATIVE scene where the target colour is absent, zero candidates
are proposed → score = 0% with near-zero compute cost.

### Applies to all 9 base techniques

| Technique | CF1 variant names |
|-----------|------------------|
| Template Matching | `CF1_TM_CCOEFF_NORMED_LOOSE`, `CF1_TM_CCOEFF_NORMED_TIGHT` |
| Feature Matching (SIFT) | `CF1_SIFT_LOOSE`, `CF1_SIFT_TIGHT` |
| Contour Shape Matching | `CF1_CSM_I1_LOOSE`, `CF1_CSM_I1_TIGHT` |
| Hough Circles | `CF1_HOUGH_CIRCLES_LOOSE`, `CF1_HOUGH_CIRCLES_TIGHT` |
| Generalized Hough | `CF1_GHT_BALLARD_LOOSE`, `CF1_GHT_BALLARD_TIGHT` |
| Histogram Comparison | `CF1_HC_CORREL_LOOSE`, `CF1_HC_CORREL_TIGHT` |
| Phase Correlation | `CF1_PC_LOOSE`, `CF1_PC_TIGHT` |
| Morphology Analysis | `CF1_MORPH_LOOSE`, `CF1_MORPH_TIGHT` |
| Pixel Diff | `CF1_PDIFF_LOOSE`, `CF1_PDIFF_TIGHT` |

### Architecture

Rather than wrapper classes per technique, `ColourFirstLocator` is a **generic pre-pass**
that any matcher can call directly inside its own `match()` method as an additional variant:

```java
// Inside TemplateMatcher.match():
List<Rect> windows = ColourFirstLocator.propose(sceneMat, referenceId, LOOSE, MIN_AREA);
if (windows.isEmpty()) windows = List.of(new Rect(0, 0, sceneW, sceneH)); // fallback
Rect best = windows.stream()
    .map(w -> runTemplateMatchInWindow(refMat, sceneMat, w))
    .max(Comparator.comparingDouble(AnalysisResult::matchScorePercent))
    .orElse(...);
```

This means **no new matcher wrapper classes** are needed — each existing matcher simply
gains 2 extra variant paths (`_CF1_LOOSE` / `_CF1_TIGHT`) in its own `match()` method.

### Deliverables

- `src/main/java/org/example/ColourFirstLocator.java`:
  - `List<Rect> propose(Mat scene, ReferenceId refId, double hueTolerance, int minArea)`
    — HSV threshold → `findContours` → union overlapping rects → pad each to ≥ 128×128
    → clamp to scene bounds → return list sorted by area (largest first)
  - Falls back to `[full-scene Rect]` if zero candidates found
  - Static, stateless, thread-safe

- CF1 variants added to each of the 9 existing matchers:
  - `TemplateMatcher.java` — 2 new CF1 variants
  - `FeatureMatcher.java` — 2 new CF1 variants (SIFT only, as representative)
  - `ContourShapeMatcher.java` — 2 new CF1 variants
  - `HoughDetector.java` — 2 new CF1 variants (HoughCircles only)
  - `GeneralizedHoughDetector.java` — 2 new CF1 variants
  - `HistogramMatcher.java` — 2 new CF1 variants (CORREL only)
  - `PhaseCorrelationMatcher.java` — 2 new CF1 variants
  - `MorphologyAnalyzer.java` — 2 new CF1 variants
  - `PixelDiffMatcher.java` — 2 new CF1 variants

- `src/test/java/org/example/ColourFirstMatchingTest.java`:
  - Extends `AnalyticalTestBase`
  - Runs all 18 CF1 variants (9 techniques × 2 tolerances) across the debug catalogue
  - Output: `test_output/colour_first/report.html`

### Done when
- `mvn test` produces `test_output/colour_first/report.html` ✅
- A_CLEAN: CF1 score ≥ base CF score for all 9 techniques ✅
- D_NEGATIVE: CF1 score ≈ 0% for all 9 techniques (zero candidates) ✅
- Performance table shows CF1 variants faster than full-scene CF variants ✅

---

## Milestone 16 — Technique: Pixel Diff (Baseline) ✅ COMPLETE

### Deliverables
- `src/main/java/org/example/matchers/PixelDiffMatcher.java`:
  - 1 base variant: `Core.absdiff()` + `Core.sumElems()` sliding window
  - `_CF_LOOSE` and `_CF_TIGHT` (2 CF variants)
  - Score = `1 - (diffSum / maxPossibleDiff)` → 0–100%
- `src/test/java/org/example/PixelDiffTest.java`

---

## Milestone 17 — Unified Benchmark Report

**Goal:** Aggregate all technique variant families into a single cross-technique
comparison report covering accuracy, CF improvement delta, CF1 speedup, and performance.

### Deliverables
- `src/test/java/org/example/MatchingBenchmarkTest.java`:
  - `@BeforeAll` — loads OpenCV, builds `SceneCatalogue`
  - `@Test runFullBenchmark()` — instantiates all matchers (base + CF + CF1 variants),
    runs the double loop (47 refs × 1,267 scenes) for every technique
  - `@AfterAll` — calls `PerformanceProfiler` per variant, then `HtmlReportWriter`

### Unified HTML report (`test_output/benchmark/report.html`) contains:

**Accuracy tab:**
- Cross-technique accuracy table — one row per scene variant, one column per technique+variant,
  colour-coded; JavaScript-sortable by column

**Base vs CF Comparison tab:**
- For each technique, shows: base score · CF_LOOSE score · CF_TIGHT score · delta (CF − base)
- Delta cells colour-coded: 🟢 CF improved · ⬜ no change · 🔴 CF degraded
- Summary row showing which technique benefits most from colour pre-filtering

**Colour-First (CF1) tab:**
- For techniques with CF1 variants: base score · CF1_LOOSE score · CF1_TIGHT score
- Speedup column: CF1 elapsed / base CF elapsed (expected: 0.05–0.20 on cluttered scenes)
- Candidate count column: how many regions CF1 proposed vs full-scene search

**Performance tab:**
- Cross-technique timing table including `preFilterElapsedMs` as a separate column
- Projected timings at 720p / 1080p / 1440p / 4K for both base and CF variants
- Pre-filter overhead per resolution (expected: < 1 ms at 640×480, < 8 ms at 4K)
- Memory comparison including CF mask overhead per resolution

### `benchmark_summary.csv` columns:
```
technique, variant, isCF, cfTolerance, referenceId, sceneCategory, variantLabel, backgroundId,
scorePercent, boundingX, boundingY, boundingW, boundingH,
elapsedMs, preFilterElapsedMs, scenePx,
projLinear_720p, projLinear_1080p, projLinear_1440p, projLinear_4K,
projQuad_720p, projQuad_1080p, projQuad_1440p, projQuad_4K,
heapMb_720p, heapMb_1080p, heapMb_1440p, heapMb_4K
```

### Done when
- `mvn test` produces the unified report with all three tabs rendering correctly
- Base vs CF tab shows clear patterns (e.g. Hough CF greatly reduces FP on random-line backgrounds)
- `benchmark_summary.csv` contains all projection and pre-filter columns populated

---

## Milestone Summary

| # | Milestone | Key Output |
|---|-----------|------------|
| 1 | Scaffold & smoke test | `mvn test` passes, OpenCV loads |
| 2 | Reference image library | 88 colour PNGs in `test_output/references/` ✅ |
| 3 | Background factory | 21 colour PNGs in `test_output/backgrounds/` across 4 tiers ✅ |
| 4 | Scene catalogue | 2,332 scenes (A=352, B=1320, C=616, D=44) + placement metadata, contact sheet ✅ |
| 5 | Analysis infrastructure | Dummy matcher → `report.html` with Results + Performance tabs ✅ |
| 6 | Colour pre-filter | 88 mask PNGs, red hue wrap verified, 0% false detections on black scenes ✅ |
| 7 | Template Matching | `report.html` — 6 base + 12 CF variants, Base vs CF tab ✅ |
| 8 | Feature Matching | `report.html` — 5 base + 10 CF variants (SIFT/ORB/AKAZE/BRISK/KAZE) ✅ |
| 9 | Contour / Hu Moments | `report.html` — 3 base + 6 CF variants ✅ |
| 10 | Hough Transforms | `report.html` — 2 base + 4 CF variants ✅ |
| 11 | Generalized Hough | `report.html` — 2 base + 4 CF variants ✅ |
| 12 | Histogram Comparison | `report.html` — 4 base + 8 CF variants ✅ |
| 13 | Phase Correlation | `report.html` — 2 base + 4 CF variants ✅ |
| 14 | Morphology Analysis | `report.html` — 3 base + 6 CF variants ✅ |
| 15 | Colour-First Region Proposal | `colour_first/report.html` — CF1 variants for all 9 techniques (18 variants) |
| 16 | Pixel Diff (baseline) | `report.html` — 1 base + 2 CF variants ✅ |
| 17 | Unified Benchmark | `benchmark/report.html` (Accuracy + Base vs CF + CF1 + Performance) + full CSV |











