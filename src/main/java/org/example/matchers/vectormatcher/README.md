# VectorMatcher Package — Architecture & Design

## Overview

`org.example.matchers.vectormatcher` is a **structural pattern-matching engine** that locates reference shapes in arbitrary scenes using **colour-edge cluster decomposition** and **geometry-driven scoring**.

> **Core principle:** colour is used only during cluster discovery (to separate distinct boundary groups). All candidate selection, scoring, and final matching are purely geometry/structure-driven — no colour term ever enters the final score.

The package is split into two levels:

| Level | Contents |
|---|---|
| `vectormatcher/` | `VectorMatcher.java` — public façade and orchestrator |
| `vectormatcher/components/` | Extracted single-responsibility helpers (filtering, anchor expansion, scoring, bbox refinement, geometry utilities, data carriers) |

---

## Package Map

```
vectormatcher/
├── VectorMatcher.java              # Public API + orchestration
├── README.md                       # ← you are here
└── components/
    ├── RefCluster.java             # Data: one colour-edge cluster from the reference image
    ├── RegionScore.java            # Data: three-layer scoring result (record)
    ├── CandidateFilter.java        # Stage 1–2: multi-stage candidate reduction
    ├── AnchorMatcher.java          # Stage 3: anchor assignment + expansion
    ├── RegionScorer.java           # Stage 4: three-layer scoring (L1/L2/L3)
    ├── BboxExpander.java           # Stage 5: post-score bounding box refinement
    └── GeometryUtils.java          # Shared bbox arithmetic (union, IoU, intersect)
```

---

## Key External Dependencies

| Type | Package | Role |
|---|---|---|
| `SceneDescriptor` | `o.e.matchers` | Pre-computed scene: contours grouped by colour cluster, combined chromatic mask |
| `SceneContourEntry` | `o.e.matchers` | Record: one scene contour + cluster metadata + cached `VectorSignature` |
| `VectorSignature` | `o.e.matchers` | Scale/rotation-invariant structural descriptor (topology, segments, circularity, concavity, angle histogram) |
| `VectorVariant` | `o.e.matchers` | Enum of epsilon levels (STRICT/NORMAL/LOOSE); currently only NORMAL is executed |
| `ExperimentalSceneColourClusters` | `o.e.colour` | Colour-cluster extractor (border-pixel analysis) — shared by both ref and scene paths |
| `ColourCluster` | `o.e.colour` | Raw cluster output: binary mask + hue + achromatic flags |
| `AnalysisResult` | `o.e.analytics` | Unified result record emitted to callers (score, bbox, timing, per-layer breakdown) |
| `SceneEntry` | `o.e.scene` | Scene wrapper: BGR Mat, descriptor, placement metadata |
| `ReferenceId` | `o.e.factories` | Identifies which reference shape is being searched for |

---

## Pipeline Architecture

The matching pipeline executes as a linear sequence of stages. Each stage reduces or annotates the candidate set, culminating in a scored result with a bounding box.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        VectorMatcher.match()                        │
│  Entry point — builds ref clusters, delegates to runMatch()         │
└──────────────────┬───────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  REFERENCE PREPARATION                                              │
│                                                                     │
│  buildRefClusters(refBgr)                                           │
│    1. ExperimentalSceneColourClusters.extractFromBorderPixels()      │
│       → ColourCluster list (hue, mask, achromatic flags)            │
│    2. Convert masks → contours via SceneDescriptor.contoursFromMask │
│    3. Wrap in RefCluster (caches: primaryBbox, maxContourArea,      │
│       contourBboxes[], solidity, cachedRefFraction)                 │
│    4. Fallback: if no border-pixel clusters → full extract()        │
│    5. deduplicateRefClusters() — collapse bright/dark + chrom/dark  │
│       achromatic pairs that trace the same physical edge            │
│                                                                     │
│  Critical: uses SAME extractor as scene-side (SceneDescriptor.build)│
└──────────────────┬───────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  SCENE CANDIDATE COLLECTION                                         │
│  collectSceneCandidates(descriptor)                                 │
│                                                                     │
│  • Iterates descriptor.clusters() (pre-built by SceneDescriptor)    │
│  • Skips envelope clusters                                          │
│  • Skips contours with bbox > 80% of scene area                    │
│  • Emits SceneContourEntry with sig=null (deferred for OPT-R)       │
└──────────────────┬───────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 1: CANDIDATE FILTERING                     [CandidateFilter] │
│                                                                     │
│  1a. Connected-component filter                                     │
│      Per-cluster noise reduction — drops contours < 10% (5% for     │
│      achromatic) of cluster-max area, unless centroid is inside     │
│      the main shape's bbox (compound component rescue)              │
│                                                                     │
│  1b. Global minimum-size filter                                     │
│      Scene-wide floor — drops contours < 8% of the global largest   │
│      contour. Never returns empty (safety fallback).                │
│                                                                     │
│  2.  Reference-adaptive morphological opening  (currently disabled)  │
│      computeErosionDepth() → always returns 0 today.                │
│      When enabled: erode→dilate to sever thin background-line       │
│      connections, then re-extract contours from cleaned mask.        │
└──────────────────┬───────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  OPT-R: DEFERRED SIGNATURE BUILD                                    │
│  buildSignatures(candidates, sceneArea)                             │
│                                                                     │
│  VectorSignature.buildFromContour() is expensive. Building is       │
│  deferred until after filtering so eliminated contours skip it.     │
│  Replaces null sig on each surviving SceneContourEntry.             │
└──────────────────┬───────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 3: ANCHOR LOOP                                               │
│  For each candidate as anchor:                                      │
│                                                                     │
│  3a. Anchor-to-ref assignment              [AnchorMatcher]          │
│      Assigns anchor to the RefCluster whose relative fill fraction  │
│      (maxContourArea / imageArea) best matches the anchor's         │
│      (contourArea / bboxArea).                                      │
│                                                                     │
│  3b. Expansion from anchor                 [AnchorMatcher]          │
│      Grows the matched set by selecting one scene contour per       │
│      remaining RefCluster. Selection uses ONLY:                     │
│        • Spatial proximity (centre-to-centre ≤ 35% scene diag,     │
│          or 50% for >2 ref clusters)                                │
│        • Relative size closeness                                    │
│      No colour. No geometry. Purely spatial + size.                 │
│                                                                     │
│  3c. Achromatic pair deduplication         [VectorMatcher]          │
│      Two-pass collapse of bright/dark achromatic and chrom/dark     │
│      achromatic pairs sharing the same physical boundary            │
│      (IoU ≥ 0.90, area ratio ≥ 0.75).                              │
│                                                                     │
│  3d. OPT-G: Early exit                                              │
│      If bestScore ≥ 70% and anchor's ShapeType is incompatible     │
│      with the primary ref's type → skip scoring entirely.           │
│      Compatibility: same type, either is COMPOUND, or both in the  │
│      {CIRCLE, CLOSED_CONVEX_POLY, CLOSED_CONCAVE_POLY} family.     │
│                                                                     │
│  3e. Three-layer scoring                   [RegionScorer]           │
│      (see "Three-Layer Scoring" section below)                      │
│                                                                     │
│  Track best score + all anchor data for post-loop processing.       │
└──────────────────┬───────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 4: ANCHOR RE-SELECTION              [VectorMatcher]          │
│  (only when bestScore ≥ 60%)                                        │
│                                                                     │
│  Corrects the bbox when a small background element outscored the    │
│  actual shape.                                                      │
│                                                                     │
│  Path A — Non-prominent (bestAnchor bbox < 50% of largest):         │
│    Re-select to the largest-bbox candidate scoring ≥ 70% of best,   │
│    gated by ShapeType compatibility.                                │
│                                                                     │
│  Path B — Prominent (bestAnchor is large):                          │
│    Tight same-size swap only: bbox area ≥ 90%, score ≥ 90%,        │
│    larger contour area. Also gated by ShapeType.                    │
└──────────────────┬───────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 5: BBOX EXPANSION                   [BboxExpander]           │
│  (only when bestScore ≥ 75%)                                        │
│                                                                     │
│  Multi-step refinement to capture the complete shape extent:        │
│                                                                     │
│  A. Compute reference full extent + estimate scene-to-ref scale     │
│     (half-step quantised: 1×, 1.5×, 2×, 2.5×, … 8×)               │
│                                                                     │
│  B. Derive three area caps from ref geometry × scale:               │
│     • anchorTrimArea  (5% margin) — trimming inflated anchor bbox   │
│     • matchedUnionArea (15% margin) — union with scored contours    │
│     • siblingExpArea   (varies) — unverified sibling expansion:     │
│       - Elongated ref (AR > 1.8): 0% margin (exact ref area @ scale)│
│       - Near-square ref: 5% margin (= anchorTrimArea)              │
│                                                                     │
  C. Trim + union matched contours                                   
     Trim anchor bbox if inflated beyond anchorTrimArea.             
     Union with co-located matched contours (within 50% anchor diag).
     **Skip union if refAR > 3.0** — extremely thin shapes (LINE_H,
     LINE_V) already have complete extent; unionizing other matched
     LINE_SEGMENT contours from busy backgrounds causes bbox growth
     perpendicular to the primary axis (e.g. 342×9 → 342×74).
     Area cap is matchedUnionArea for multi-colour refs,             
     anchorTrimArea for single-colour refs.
│                                                                     │
  D. Sibling expansion (4-pass iterative, skipped if refAR > 3.0)    
     Only considers candidates from clusters already represented by   
     matched entries (pre-filtered to avoid unrelated cluster noise). 
     Union same-cluster siblings of matched entries when:            
     • Concentric (centre ≤ 35% of sibling diagonal), OR            
     • Overlapping (rects intersect)                                 
     AND sibling area ≥ 10% of current bbox area.                   
     Hard cap at siblingExpArea, with a tightly-concentric override  
     (centre ≤ 5% anchor diag, area ≤ 3× current) for compound     
     outer rings (e.g. COMPOUND_BULLSEYE).                           
     **Skipped entirely for refAR > 3.0** — extremely thin shapes
     (LINE_H, LINE_V) have complete extent from Step C.
│                                                                     │
│  E. Sanity: reject if expanded bbox > 80% scene area.              │
└──────────────────┬───────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  RESULT EMISSION                                                    │
│                                                                     │
│  • Write annotated PNG (4× upscale, nearest-neighbour) if variant   │
│    is in saveVariants set                                           │
│  • Return AnalysisResult with score %, best bbox, elapsed time,     │
│    per-layer breakdown (ScoringLayers)                              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Three-Layer Scoring (`RegionScorer`)

The scoring model uses three independent layers, each measuring a different structural property. The layers are weighted and summed:

| Layer | Weight | What it measures |
|---|---|---|
| **L1 — Boundary count** | `W_COUNT = 0.15` | Does the candidate region have the same number of distinct colour-edge boundaries as the reference? |
| **L2 — Structural coherence** | `W_MATCH = 0.25` | Are the matched boundaries spatially co-located, size-consistent, and at the right scale? |
| **L3 — Primary geometry** | `W_GEOM = 0.60` | How similar is the best ref contour to the best scene contour (via `VectorSignature.similarity`)? |

**Combined = L1 × 0.15 + L2 × 0.25 + L3 × 0.60**, clamped to [0, 1].

### Layer 1 — Boundary Count

```
diff = matchedCount - refCount

if diff > 0:  score = exp(-0.10 × diff)           // over-segmentation penalty
if diff ≤ 0:  score = max(exp(-0.10 × |diff|), 0.45)  // under-segmentation (gentler)
```

**Fix B — Chromatic contamination:** For achromatic-only references, if the candidate bbox sits on top of heavily chromatic scene content (>70% chromatic pixel fraction), countScore is further penalised via `(1 - excess)^1.5`, floored at 0.30.

**High-geometry rescue:** If L3 ≥ 0.95 and L2 ≥ 0.95, countScore is floored at 0.45 to prevent count-only over-segmentation from suppressing a clear true match.

### Layer 2 — Structural Coherence

For each ref cluster, find the best-matching scene entry (by bbox-normalised fill ratio closeness), then score:

- **Proximity** (40%): `max(0, 1 - dist/(sceneDiag × 0.30))`
- **Coverage** (40%): `1 - |refFrac - sceneFrac| / max(refFrac, 0.01)` — both fractions are bbox-normalised (area / own bbox area), making them scale-invariant
- **Scale** (20%): penalises extreme bbox-dimension ratios (>4.5× → 0.70, >6× → 0.25)

Primary cluster carries weight 1.0; secondary clusters carry weight 0.30.

**LINE_SEGMENT guard:** When either contour has aspect ratio > 4:1, AABB-based coverage is unreliable (a thin line rotated 45° has a much larger AABB than its actual extent). In this case, coverage is set to 1.0 (skipped) and scale uses contour-area ratio directly.

### Layer 3 — Primary Geometry

Computes `VectorSignature.similarity(refSig, sceneSig)` for every (ref contour, scene entry) pair across ALL ref clusters and ALL matched scene entries. The best similarity wins.

This all-pairs approach is necessary for compound shapes — e.g. a circle-outline cluster's `bestSig` might be a co-located triangle (higher solidity), causing incorrect pairing if only per-cluster best signatures were used.

**Outline coherence boost (in `VectorSignature.similarity`):** For outline shapes (both sides have solidity < 0.40) on busy backgrounds, background contours physically merge with the shape edges, causing `segScore` and `topoScore` to fail even when global geometric metrics (type, circularity, solidity, vertex count, AR) agree well. When all global metrics agree above safety thresholds, seg and topo are floored to prevent contamination-induced score collapse. Three tiers: Strong (floor 0.65), Moderate (floor 0.55), Weak (floor 0.45). This is a geometry-only mechanism — it fires on the structural agreement of the shape itself, not on colour.

---

## Data Model

### `RefCluster`

Represents one colour-edge cluster from the 128×128 reference image.

| Field | Type | Description |
|---|---|---|
| `hue` | `double` | Mean hue (OpenCV 0–180 scale) |
| `achromatic` | `boolean` | True if cluster is achromatic (low saturation) |
| `brightAchromatic` | `boolean` | True if cluster is bright achromatic (V > 200) |
| `contours` | `List<MatOfPoint>` | All contours in this cluster |
| `imageArea` | `double` | Total reference image area (typically 16384 = 128×128) |
| `maxContourArea` | `double` | Area of the largest (primary) contour |
| `primaryBbox` | `Rect` | Cached bounding rect of the primary contour |
| `contourBboxes` | `Rect[]` | Cached bounding rects for ALL contours (OPT-E) |
| `cachedRefFraction` | `double` | `maxContourArea / imageArea` (OPT-S) |
| `solidity` | `double` | Primary contour solidity (area / convex hull area) |
| `cachedSig` | `VectorSignature` | Lazily computed signature via `bestSig(eps)` |

### `RegionScore`

Immutable record carrying the scoring result:

```java
record RegionScore(double combined, double countScore, double matchScore, double geomScore)
```

All values in [0, 1]. Also has `toArray()` for legacy callers.

### `SceneContourEntry` (external)

```java
record SceneContourEntry(MatOfPoint contour, int clusterIdx, boolean achromatic,
                          boolean brightAchromatic, double clusterHue,
                          VectorSignature sig, Rect bbox, double area)
```

The `sig` field starts as `null` (OPT-R deferred build) and is populated after filtering.

---

## Deduplication Strategy

Both ref-side and scene-side share the same two-pass deduplication logic:

**Pass 1 — Bright/dark achromatic pair:** A shape on a contrasting background produces two achromatic clusters (bright side, dark side) tracing the same physical edge. Collapsed when bbox IoU ≥ 0.90 AND area ratio ≥ 0.75. The larger-area cluster survives.

**Pass 2 — Chromatic/dark achromatic pair:** A coloured shape on a dark background produces a chromatic cluster (colour side) and a dark achromatic cluster (background side). The chromatic entry survives. Same IoU + area-ratio safety conditions.

**Safety conditions** prevent collapsing genuine multi-boundary shapes (e.g. `HEXAGON_OUTLINE` inner void has area ratio ≈ 0.79 < the 0.90 IoU threshold, so its inner/outer rings are preserved).

---

## Performance Optimisations

| Tag | Description |
|---|---|
| **OPT-R** | Deferred signature build — `VectorSignature.buildFromContour()` is called only after CC + global size filtering, so eliminated contours skip the expensive computation |
| **OPT-G** | Early exit — once `bestScore ≥ 0.70`, skip scoring for anchors whose `ShapeType` is incompatible with the primary ref |
| **OPT-E** | Eager bbox caching — `RefCluster` pre-computes all contour bounding rects in the constructor to avoid per-anchor recomputation |
| **OPT-S** | Cached ref fraction — `maxContourArea / imageArea` is computed once and stored |

---

## Constants Reference

### VectorMatcher

| Constant | Value | Purpose |
|---|---|---|
| `EPSILON` | `0.04` | Polygon approximation: 4% of perimeter (VECTOR_NORMAL) |
| `DEDUP_IOU_MIN` | `0.90` | Min bbox IoU for deduplication — high to preserve inner/outer rings |
| `DEDUP_AREA_RATIO_MIN` | `0.75` | Min area ratio for dedup confirmation — lower to handle anti-aliasing |

### RegionScorer

| Constant | Value | Purpose |
|---|---|---|
| `W_COUNT` | `0.15` | Layer 1 weight |
| `W_MATCH` | `0.25` | Layer 2 weight |
| `W_GEOM` | `0.60` | Layer 3 weight |
| `CLUSTER_PENALTY_K` | `0.10` | Extra-boundary decay rate |
| `CLUSTER_PENALTY_K_MISS` | `0.10` | Missing-boundary decay rate |
| `MIN_COUNT_SCORE_MISS` | `0.45` | Floor for missing-boundary case |
| `SECONDARY_WEIGHT` | `0.30` | Layer 2 weight for non-primary clusters |

### CandidateFilter

| Constant | Value | Purpose |
|---|---|---|
| `CC_AREA_RATIO_MIN` | `0.10` | Per-cluster noise floor (0.05 for achromatic clusters) |
| `MIN_GLOBAL_AREA_RATIO` | `0.08` | Scene-wide minimum contour size |

### AnchorMatcher

| Constant | Value | Purpose |
|---|---|---|
| `PROXIMITY_THRESHOLD` | `0.35` | Max centre-to-centre distance as fraction of scene diagonal |

---

## Colour-Agnostic Guardrails

These are hard rules enforced by the project's AGENTS.md and reflected in the code:

1. **Allowed:** Hue/saturation usage for cluster discovery and grouping in `SceneDescriptor` and colour-cluster extraction paths.
2. **Forbidden:** Colour-dependent terms in `RegionScorer` or `VectorSignature.similarity()`.
3. **Forbidden:** Selecting the final winning candidate by hue/colour identity instead of structural similarity.
4. **Enforced:** `AnchorMatcher.expandFromAnchor()` uses only spatial proximity and relative size — no colour check, no geometry check.
5. **Enforced:** `RegionScorer.scoreGeometry()` calls `VectorSignature.similarity()` which is purely structural (topology, vertex count, circularity, concavity, segment descriptors, angle histograms).

---

## Single-Variant Rationale

Although `VectorVariant` defines three epsilon levels (STRICT=0.02, NORMAL=0.04, LOOSE=0.08), only **VECTOR_NORMAL** is executed. Three variants no longer add value because:

- 87% of the geometry score (via `VectorSignature.similarity`) is epsilon-independent (circularity, concavity ratio, solidity, segment descriptors, component count)
- The expansion loop no longer uses geometry for candidate selection (it was moved to a spatial+size-only model)
- Running three variants tripled execution time for negligible accuracy gain

---

## Debug Flags

| System Property | Effect |
|---|---|
| `-Dvm.debug` | Enables `[VM-ANCHOR]` per-anchor logging in `VectorMatcher.runMatch()` |
| `-Dvm.bbox.debug` | Enables `[BBOX-DEBUG]` and `[STEP-C]` logging in `BboxExpander` |

---

## Thread Safety

`VectorMatcher` is a stateless utility class (private constructor, all-static methods). `RefCluster` instances are created per-call and not shared. The class is safe for concurrent use from parallel test methods.

---

## Native Memory Hygiene

- `RefCluster.release()` releases all `MatOfPoint` contours
- `VectorMatcher.matchWithDescriptor()` releases all `RefCluster` instances after use
- `SceneDescriptor.build()` results are released in the `finally` block when not pre-built
- `BboxExpander` and `RegionScorer` do not allocate native memory (they operate on existing Mats/contours)
- `CandidateFilter.reExtractTopCandidates()` explicitly releases temporary `Mat` objects (mask, kernel, eroded, opened)

