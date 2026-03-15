# Conversation Notes — VectorMatcher Contour Isolation Problem

## Context
Session date: 2026-03-14. Discussing architectural improvements to `VectorMatcher` and the `SceneColourClusters` pipeline to improve robustness on busy backgrounds.

---

## Topic 1: Single-cluster contour assignment

**Problem raised:** Each contour/vector is assigned to exactly one colour cluster (winner-take-all per pixel). In busy scenes where a physical edge is shared between two colour regions, one cluster may lose that boundary entirely.

**Current mitigation:** The `VectorMatcher` expansion loop seeds from the primary cluster and expands spatially to adjacent clusters — partially reconstructing fragmented contours. The achromatic-dedup logic (IoU > 0.50 AND area ratio > 0.90) handles the most common BRIGHT/DARK cluster pair representing the same physical edge, but does NOT generalise to chromatic-vs-chromatic pairs.

**Proposed idea (discussion only):** Duplicate a vector/contour into all clusters it overlaps with (if ≥ some fraction of its border pixels fall within that cluster's hue tolerance). Rejected for full duplication due to FP risk — colour agreement is the primary discriminator. Alternatives noted:
- Extend dedup to chromatic pairs (higher IoU threshold, e.g. 0.70)
- Contour-first assignment: run `findContours` on the full edge map first, assign each whole contour to its dominant cluster (avoids per-pixel fragmentation at source)
- Soft threshold: duplicate only if ≥ 40 % of border pixels match the target cluster's hue

---

## Topic 2: Background line contamination — the core problem

**Observed failure:** `CIRCLE_FILLED` on `BG_RANDOM_LINES`:
- Clean scene (own): **98.1%** ✅
- On random-lines background: **86.9%** with **wrong bbox** ❌

**Root cause (traced through code):**

1. `BackgroundFactory.randomLines()` draws 30 random-colour lines (channel values 60–255, no saturation constraint) on a dark bg. Some lines are near-grey (saturation < `MIN_SAT=35`) → classified as **BRIGHT_ACHROMATIC**.
2. `CIRCLE_FILLED` from `buildShapeMat()` is pure white → also **BRIGHT_ACHROMATIC**.
3. Both land in the same cluster mask. If a near-grey line exits from the circle's boundary, those adjacent pixels merge with the circle in the cluster mask → one connected region: **circle + line arm**.
4. `contoursFromMask` traces the merged blob → **circle + STRAIGHT arm segments**.
5. `SegmentDescriptor` in Layer 3 sees STRAIGHT segments where a pure circle has only CURVED → **score drops**.
6. With a degraded circle contour, a background arc elsewhere may score **higher** → matcher returns **wrong bbox**.

**Key pipeline details confirmed:**
- `extractFromBorderPixels`: uses morphological gradient for hue-peak *discovery*, but the cluster MASK is the full `hueRangeMask` (all matching pixels, not just border pixels)
- `SceneGenerator.placeAtCentre()` / `MatchDiagnosticLibrary.compositeOnBackground()`: background drawn first, shape pasted on top via non-black mask. Lines INSIDE the circle footprint are erased. Contamination is from lines outside the circle that touch its boundary.
- `SceneDescriptor` does NOT retain raw cluster `Mat` masks after construction — they are released after `contoursFromMask` runs.

---

## Topic 3: Proposed solutions

### 3a. Connected-component filtering on cluster mask (spatial isolation)
After `hueRangeMask` builds the cluster mask, decompose into connected components and **keep only the K largest** (K=2–3 for compound/multi-part shapes). Isolated same-colour background fragments that are not physically connected to the main shape are discarded.

- ✅ Safe for both filled and outline shapes (the main shape is the largest component)
- ✅ No geometry information required
- ❌ Does NOT help when the arm is physically connected to the circle (they form one component)

### 3b. Morphological opening on cluster mask (arm severance — conditional)
Apply erode(N px) + dilate(N px) before `contoursFromMask`. `BG_RANDOM_LINES` uses 1–2 px line thickness — a 2 px opening severs arms cleanly while a filled circle (60 px radius solid) survives.

- ✅ Directly severs thin line arms connected to filled shapes
- ❌ DESTROYS outline shapes (`CIRCLE_OUTLINE`, `LINE_H`, etc.) which are themselves 1–2 px thick
- → **Must be conditional on reference fill density**

### 3c. Reference-adaptive erosion (active proposal)
Determine erosion depth from the **reference shape's fill profile** before applying it to scene cluster masks:

| Reference fill type | solidity / fill_ratio | Erosion depth |
|---|---|---|
| Filled polygon (circle, rect, hexagon) | ≥ 0.70 | 2 px |
| Partial fill (star, arrow, compound) | 0.30–0.70 | 1 px |
| Outline / line | < 0.30 | 0 px (skip) |

`VectorSignature.solidity` (already computed) is the natural signal. Erosion is applied per-cluster in VectorMatcher after the reference is known — NOT in `SceneDescriptor.build()` (which has no reference context).

**Architecture note:** Since `SceneDescriptor` releases cluster masks after construction, reference-adaptive erosion requires either:
- (a) Re-running `SceneColourClusters.extractFromBorderPixels()` on the candidate cluster (targeted, low cost since only the winning cluster is re-extracted)
- (b) Retaining raw cluster masks in `SceneDescriptor` (memory cost)
- (c) Storing two contour lists per cluster: clean + eroded (doubles memory, avoids re-extraction)

Option (a) is the least invasive: VectorMatcher checks reference solidity, and if erosion is warranted, re-extracts just the relevant cluster with appropriate opening before scoring.

### 3d. Expansion loop already partially compensates
The multi-cluster expansion loop aggregates candidates across clusters but does NOT clean individual contours. The contamination (arm) exists WITHIN one cluster's contour. Layer 3 already penalises unexpected STRAIGHT segments on a CIRCLE reference, but not strongly enough to prevent wrong-bbox selection at 87% vs a background arc.

---

## Topic 4: Performance impact assessment

**Re-extraction cost per triggered pair (option a):**
- BGR→HSV + `hueRangeMask` + morphological open + `contoursFromMask` + `VectorSignature`: ~3–5 ms
- Current VectorMatcher per-pair cost: ~5–15 ms → overhead is 25–100% per triggered pair

**At full benchmark scale (143,220 pairs, ~70% triggered, 32 threads): ~12–40 seconds real-time extra — negligible.**

**Timing flaw identified:** Option (a) as originally described re-extracts the *winner* post-score. In the exact failure case the background arc *wins* — cleaning the background arc's contour does nothing. Re-extraction must happen **pre-score**, applied to the top-K candidates by raw contour area (K=3 recommended) before the full scoring loop runs. Extra cost: ~12 ms per triggered pair. Still acceptable at all scales.

**Memory concern (storing cluster masks in SceneDescriptor) — resolved:**
Not a real concern. `SceneDescriptor` only needs to hold the active scene's masks during matching — not all scenes simultaneously. The catalogue is iterated; each scene is processed and released. The memory model is already correct.

**Live environment context:**
In production, the matcher runs against one incoming image every ~100–200 ms. The full re-extraction cost (~4 ms) is less than 4% of the available budget. This is entirely acceptable.

---

## ✅ Proposal approved — reference-adaptive erosion (pre-score, top-K)

**Agreed implementation path:**

1. After `buildRefClusters`, compute reference solidity from `primaryRef.bestSig(EPSILON).solidity`.
2. Determine erosion depth:
   - solidity ≥ 0.70 → 2 px open (filled shapes)
   - solidity 0.30–0.70 → 1 px open (partial/star/arrow)
   - solidity < 0.30 → 0 px (outline/line — skip entirely)
3. If erosion depth > 0: for the top-K=3 scene candidates by raw contour area, re-extract their cluster masks from `scene.sceneMat()` (BGR→HSV + `hueRangeMask` for the known cluster hue + morphological open + `contoursFromMask`), replace their contours in the candidate pool before the expansion/scoring loop runs.
4. Outline shapes (solidity < 0.30) are never touched — no regression risk for `CIRCLE_OUTLINE`, `LINE_H`, etc.

**Combined with connected-component filtering (3a) as first pass** — handles isolated noise blobs that are not connected to the main shape, before erosion handles connected arms. Two-stage, applied inside `VectorMatcher.runMatch()` prior to `collectSceneCandidates`.

---

## ✅ IMPLEMENTED (2026-03-14)

### Stage 1 — Connected-component filter (`VectorMatcher.applyConnectedComponentFilter`)
Per-cluster: contours whose area < 10% of the cluster's largest contour are dropped as isolated noise blobs. Applied to `collectSceneCandidates` output before the scoring loop.

### Stage 2 — Reference-adaptive erosion (`VectorMatcher.reExtractTopCandidates`)
Top-3 candidates by area are re-extracted with `MORPH_OPEN` keyed to `primaryRef.bestSig.solidity`: 2 px (≥0.70), 1 px (0.30–0.70), 0 (skip). For achromatic candidates, a `MORPH_GRADIENT` step follows the opening to restore border-pixel representation. Spatially closest new contour replaces the original entry.

### Hue anti-aliasing — valley-based exclusive assignment (`SceneColourClusters`)
- `computeSmoothedHist` extracted as standalone helper; smoothing radius reduced from 2 → 1 (3-bin window) for sharper peak separation.
- `findPeaks` refactored to accept pre-smoothed histogram.
- `computeValleyBounds` + `findValleyBetween` added: for each pair of adjacent hue peaks the actual histogram minimum is found and used as the hard cluster boundary. Each pixel assigned to exactly one cluster (no ±14° window overlap between adjacent colours).
- `hueRangeMaskByBounds(hsv, lo, hi)` builds masks from valley bounds with circular wrap-around.
- Public `buildHueMask` / `buildAchromaticMask` added for `VectorMatcher` re-extraction.



