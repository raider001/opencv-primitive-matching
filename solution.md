# VectorMatcher — Solution Plan

> This document describes the correct target architecture.
> Once agreed, the code will be rewritten from scratch based on this plan.

---

## Guiding Principles

1. **Structural pattern matching — not colour matching.**
   Hue is used in exactly one place: to identify and separate distinct boundary
   clusters during cluster discovery. After that, hue plays no further role in
   selection, matching, or scoring.

2. **Ref and scene use identical cluster discovery.**
   Both images go through exactly the same pipeline. Different methods produce
   incomparable representations — matching becomes fundamentally unsound.

3. **Selection and scoring are independent.**
   The expansion loop selects which scene clusters are "relevant". It must not
   use any signal that is also used for scoring — doing so pre-determines the
   score before evaluation happens.

4. **Each layer asks a genuinely different question.**
   The three scoring layers are non-redundant. Each evaluates a different
   aspect of the candidate.

---

## Step 1 — Cluster Discovery

### Shared pipeline — called identically for both ref and scene

```
1. BGR → HSV

2. Full-image edge detection:
   - Build valid-pixel mask (all pixels with val ≥ MIN_VAL)
   - Apply morphological gradient (dilate − erode) to full mask
   - Result: edge pixels wherever ANY colour/luminance transition occurs
     (white↔black, red↔blue, red↔white — all captured in one pass)

3. Classify each edge pixel by colour:
   - sat ≥ MIN_SAT AND val ≥ MIN_VAL  → chromatic → assign to nearest hue peak
   - sat < MIN_SAT AND val ≥ BRIGHT_THRESH → bright achromatic
   - val < BRIGHT_THRESH               → dark achromatic

4. Build one binary mask per cluster from its classified edge pixels

5. findContours on each mask → each contour IS a shape boundary
```

### Role of Hue

Hue (including the bright/dark achromatic split) labels each edge pixel so
it can be grouped into the correct cluster mask. This is its only purpose.

The number of distinct clusters produced IS a meaningful signal — it tells
us how many distinct colour boundaries the shape has. The specific hue values
are not used anywhere after this step.

### Cluster Deduplication

Near-duplicate contours arise from one specific cause: the bright and dark
achromatic masks both trace the **same physical edge** from opposite sides.
At a white/black boundary, the morphological gradient fires on both the bright
mask (from the bright side) and the dark mask (from the dark side), producing
two thin-strip contours at essentially the same location.

**This only happens between a bright achromatic cluster and a dark achromatic
cluster.** It cannot happen between:
- Two chromatic clusters (different hues → different edges → different locations)
- A chromatic and an achromatic cluster (different physical boundaries)
- Two bright achromatic clusters or two dark achromatic clusters (impossible —
  there is at most one bright and one dark achromatic cluster per image)

**Deduplication rule — bright/dark achromatic pair only:**
If the matched set contains both a bright achromatic entry AND a dark achromatic
entry, and their contour bboxes overlap significantly (IoU > 0.5), they represent
the same physical boundary traced from opposite sides. Keep only the one with
the larger contourArea (the outer boundary side) — it carries more structural
information. The other is discarded before scoring.

**Do NOT deduplicate:**
- Chromatic cluster entries — ever
- Two bright or two dark achromatic entries — cannot exist
- A bright achromatic entry whose bbox does NOT significantly overlap a dark
  achromatic entry — they represent different boundaries (e.g. the bright
  shape boundary vs a separate dark region boundary)

**Why IoU > 0.7 alone is NOT safe — worked example:**

HEXAGON_OUTLINE has a 2px stroke. The morphological gradient (3×3 kernel) fires
on both sides of that stroke. The ref image is 128×128, hexagon fits within
~108px outer diameter.

- Bright cluster bbox (outer stroke edge): ≈ 108×108
- Dark cluster bbox (inner stroke edge):   ≈ 96×96, centred

```
intersection area = 96 × 96 = 9216
union area        = 108×108 + 96×96 - 9216 = 11664
IoU               = 9216 / 11664 = 0.79
```

IoU = 0.79 — this **exceeds 0.7**. Using IoU > 0.7 alone would incorrectly
collapse the outer and inner hexagon boundaries into one entry, destroying the
structural signal that distinguishes HEXAGON_OUTLINE from HEXAGON_FILLED.

**The area ratio saves it:**
- Outer hex contourArea ≈ π × 54² ≈ 9161 px²
- Inner void contourArea ≈ π × 48² ≈ 7238 px²
- Ratio = 7238 / 9161 = **0.79** — well below the 0.90 threshold → NOT collapsed ✓

**RECT_FILLED bright/dark pair:**
Both traces are thin strips around the same rectangle. The contour points form
essentially the same enclosed rectangle on both sides.
- Both contourAreas ≈ the same enclosed rect area
- Ratio ≈ **1.0** → exceeds 0.90 → collapsed ✓
- IoU ≈ 1.0 → exceeds 0.5 ✓

**Final deduplication rule — both conditions must hold:**
1. One entry is bright achromatic AND the other is dark achromatic, AND
2. Bbox IoU > 0.5, AND
3. contourArea ratio (min/max) > 0.90

The IoU threshold of 0.5 is deliberately loose — since we only ever apply
this rule to the bright/dark pair (never chromatic), a loose IoU just means
"they are roughly at the same location." The area ratio of 0.90 is the tight
discriminator that separates same-boundary duplicates from inner/outer ring pairs.



---

## Step 2 — Candidate Selection (Expansion Loop)

### Purpose

For each scene contour acting as an **anchor**, find the additional scene
clusters that are spatially co-located with it. Together these form the
**matched set** — the clusters that may collectively represent the ref shape.

### Selection Criteria — Proximity + Relative Size Only

No colour. No geometry similarity. Both belong in scoring, not selection.

**1. Spatial proximity gate:**
The candidate must be near the anchor.
```
dist(anchor_centre, candidate_centre) ≤ scene_diagonal × PROXIMITY_THRESHOLD
```

**2. Relative contour area — pick the best match:**
Among proximate candidates, prefer the one whose contour area fraction best
matches the ref cluster's contour area fraction. This is a structural size
signal — it avoids picking tiny noise fragments when the ref cluster represents
a large boundary.

**Critical — normalise to candidate region, not full image:**
The ref fraction is relative to the ref image area (128×128 = 16384px).
The scene fraction must be relative to the **candidate region bbox area**,
not the full scene (640×480 = 307200px). A shape that fills 50% of the ref
should fill roughly 50% of the candidate bbox in the scene, regardless of
how much of the full scene it occupies.

```
refFraction       = contourArea(rc.bestContour()) / refImageArea
candidateFraction = contourArea(ce.contour)       / candidateBboxArea
best = argmin(|refFraction - candidateFraction|, proximate candidates)
```

### Anchor-to-Ref Cluster Assignment

The anchor covers the ref cluster whose relative contour area (vs ref image)
it most closely matches (vs candidate bbox). That ref cluster is skipped.
All remaining ref clusters are searched for additional scene clusters.

**Edge case — equal-area clusters (e.g. BICOLOUR_RECT_HALVES):**
When multiple ref clusters have similar contour areas, the anchor may match
any of them ambiguously. In this case, assign by the ORDER of the expansion
search — anchor takes the first ref cluster; expansion fills the rest in order.
The scoring layers are symmetric for equal-weight clusters so the assignment
order does not affect the final score.

```
anchorRef = argmin(|refFraction(rc) - anchorFraction(anchor, candidateBbox)|)

for each rc in refClusters where rc ≠ anchorRef:
    best = proximate candidate with closest relative contour area to rc
    if found: add to matchedEntries
```

### BBox Expansion

Only expand `regionBbox` to include a matched entry if:
- Its contour area is < 60% of the full scene area (not background-scale), AND
- It is not an achromatic secondary boundary (these are structural context only)

Background-scale contours contribute to structural scoring but must not inflate
the detection bbox.

---

## Layer 1 — Boundary Count

### Question
**"Does the candidate region contain the right number of distinct structural
boundaries?"**

### Signal

Count of distinct physical boundaries in the deduplicated matched set vs the
ref cluster count. Exponential decay penalty for mismatch:

```
diff  = matchedBoundaryCount - refClusterCount
score = exp(-K × 2 × diff)   if diff > 0  (more than ref — steep penalty)
score = exp(-K × |diff|)      if diff ≤ 0  (fewer than ref)
```

Steeper penalty for extra boundaries — these almost always mean the candidate
is absorbing unrelated background structure.

### Fix B — Chromatic Contamination (achromatic refs only)

For refs whose clusters are all achromatic (white/grey/black shapes), penalise
candidates whose bounding box contains chromatic (coloured) pixels:

```
contamination = chromaticPixelsInBbox / bboxArea
score        *= (1 - contamination)^4
```

`combinedChromaticMask` is built from filled chromatic pixel regions (not edge
pixels) so that pixel-area counting inside the bbox works correctly.

A clean white rect on a dark background: zero contamination → no penalty.
A blob spanning coloured circles: high contamination → heavily penalised.

### What Layer 1 Does NOT Do
- Does not use hue values — boundary count only
- Does not score geometry
- Does not score spatial relationships between boundaries

---

## Layer 2 — Structural Coherence

### Question
**"Do the matched boundaries have the right spatial relationships and relative
sizes with respect to each other and to the candidate region?"**

### Core Principle

Colour is entirely absent. Layer 2 scores the quality of the structural
organisation assembled by the expansion loop.

### 2a — Spatial Proximity Score

Each matched entry must be near the anchor:

```
dist           = euclidean(anchor_centre, entry_centre)
diagonal       = sqrt(sceneWidth² + sceneHeight²)
proximityScore = max(0, 1 - dist / (diagonal × 0.3))
```

### 2b — Relative Size Consistency

The ref cluster's contour encloses a certain fraction of the ref image.
The matched scene contour should enclose a similar fraction of the **candidate
region bbox** (not the full scene). This verifies structural proportions.

Use `contourArea` (enclosed area), NOT pixel count. Edge-first cluster masks
contain only edge pixels — pixel count reflects stroke width, not shape size.

```
refFraction   = contourArea(rc.bestContour()) / refImageArea
sceneFraction = contourArea(entry.contour)    / regionBboxArea
coverageScore = 1 - min(1, |refFraction - sceneFraction| / max(refFraction, 0.01))
```

### 2c — Structural Role Weighting

The primary boundary defines the shape. Secondary boundaries corroborate it.

**Primary** = ref cluster with the largest `contourArea`. Colour-agnostic.

**Important for outline shapes:** for shapes like HEXAGON_OUTLINE, the outer
boundary has the largest contourArea. The inner void boundary is secondary.
Both contribute to Layer 2 — the outer gets weight 1.0, the inner gets 0.30.
This is correct: the outer boundary's proximity and proportions matter most,
while the inner void confirms the outline nature of the shape.

```
weight = 1.0  if rc is primary (largest contourArea in ref)
weight = 0.30 otherwise
```

### Formula

```
for each RefCluster rc:
    entry = matched scene entry for rc

    if entry is null:
        contribution = 0
        weight       = role_weight(rc)    // still in denominator
        continue

    proximityScore = max(0, 1 - dist / (diagonal × 0.3))
    coverageScore  = 1 - min(1, |refFraction - sceneFraction| / max(refFraction, 0.01))
    weight         = 1.0 if primary, else 0.30

    contribution = weight × (proximityScore × 0.5 + coverageScore × 0.5)

clusterMatchScore = sum(contributions) / sum(weights)
```

Missing ref clusters reduce the score proportionally via the denominator.
No re-penalisation of the cluster count — that is Layer 1's job.

### What Layer 2 Does NOT Do
- Does not use hue or colour
- Does not score geometry (shape, vertices, circularity)
- Does not re-penalise cluster count

---

## Layer 3 — Geometry

### Question
**"Does the primary boundary's geometric structure match the reference?"**

### Core Principle

Colour is absent. Only the **primary cluster** (largest `contourArea` in ref)
contributes to the geometry score. Secondary clusters have already contributed
to Layers 1 and 2.

**Important — why only the primary:**
Secondary cluster contours (background context, interior voids) are structurally
valid but their geometry varies significantly based on scene composition —
background shapes, noise, and proximity to other elements all change the
secondary contour. The primary boundary is the most stable and reproducible
geometric signal.

```
primaryRef   = argmax(contourArea(rc.bestContour()), refClusters)
primaryScene = matched entry for primaryRef

geometryScore = 0.0 if primaryScene is null
              else primaryRef.bestSig(eps).similarity(primaryScene.sig)
```

### VectorSignature — Edge-First Does Not Break It

`buildFromContour` renders the contour as a **filled polygon** before computing
metrics. Circularity, solidity, vertex count, aspect ratio, and type are all
computed from the filled shape — not from the raw edge-pixel strip. The
edge-first cluster change has no effect on these metrics.

Both ref (`rc.bestSig`) and scene (`entry.sig`) use `buildFromContour`.
Both sides computed identically — metrics are comparable.

### VectorSignature Sub-Components

| Component | Weight | Notes |
|---|---|---|
| Type (CIRCLE/CONVEX/CONCAVE/LINE) | 0.15 | From filled shape — valid |
| SegmentDescriptor | 0.38 | From raw contour traversal — primary structural signal |
| Topology | 0.10 | Edge traversal structure — valid |
| Circularity | 0.13 | From filled polygon — valid |
| Solidity | 0.11 | From filled polygon hull — valid |
| Vertex count | 0.08 | **Fix: symmetric ratio** (see below) |
| Angle histogram | 0.05 | Rotation-invariant — valid |
| Aspect ratio | ×multiplier | Bbox ratio gate — valid |

### Vertex Count Fix — Symmetric Ratio

Current: `min(vScene, vRef) / vRef`
A 20-vertex noise blob scores 1.0 against a 4-vertex rect — extra vertices
are not penalised at all.

Fix: `min(vScene, vRef) / max(vScene, vRef)`
Symmetric — penalises both extra AND missing vertices equally.

### Filled vs Outline Shape Discrimination

A critical geometry requirement: HEXAGON_OUTLINE must not match HEXAGON_FILLED,
and vice versa. Both have the same outer boundary shape — they are distinguished
only by **solidity** (filled ≈ 1.0 vs outline ring ≈ low) and the presence of
an inner void cluster.

Solidity is already in VectorSignature (weight 0.11). But 0.11 may be
insufficient to hard-separate filled from outline when other metrics agree
strongly. Consider: if two shapes agree on type, circularity, vertices, aspect
ratio, and segments (0.15+0.38+0.10+0.13+0.08+0.05 = 0.89 weight), a solidity
mismatch of 0.8 only contributes `0.11 × 0.8 = 0.088` penalty — not enough.

**Fix: increase solidity weight to 0.20, reduce SegmentDescriptor to 0.30.**
Solidity is the primary filled/outline discriminator. It should not be
outweighed by shape traversal agreement.

### What Layer 3 Does NOT Do
- Does not use colour
- Does not score secondary clusters
- Does not re-check proximity, count, or proportions — Layers 1 and 2

---

## Variants — STRICT / NORMAL / LOOSE

### What Epsilon Actually Controls

The three variants exist to run `approxPolyDP` at different tolerances:
- STRICT: ε = 2% of perimeter — preserves fine detail
- NORMAL: ε = 4% of perimeter
- LOOSE:  ε = 8% of perimeter — tolerates noise

Epsilon only directly affects two things in `VectorSignature`:
- **Vertex count** (weight 0.08) — loose collapses more polygon corners
- **Angle histogram** (weight 0.05) — fewer vertices = fewer angles

Everything else is epsilon-independent:
- SegmentDescriptor (0.38) — built from raw contour BEFORE approxPolyDP
- Circularity (0.13) — computed from filled polygon rendered at fixed strictEps=0.02
- Solidity (0.11) — same
- Type classification (0.15) — same
- Topology (0.10) — built at strictEps internally
- Aspect ratio multiplier — from bbox, unaffected

**Total epsilon-sensitive weight: 0.13 out of 1.0.**

### Why Three Variants Were Needed Before

In the old architecture, the expansion loop used geometry similarity
(`rc.bestSig(eps).similarity(ce.sig)`) for candidate selection. Different
epsilons explored different candidate sets — the variants genuinely diverged.

In the new architecture, selection is purely proximity + relative size.
Epsilon only affects Layer 3 scoring of the already-selected primary cluster.
Three variants would produce scores differing by at most 13% of total weight.
The best-of-three would almost never differ from a single well-chosen epsilon.

### Decision — Single Variant

Drop to a single variant. Use a fixed epsilon of **4% of perimeter** (the
current NORMAL value) — this is the established general-purpose baseline that
works well across the shape catalogue.

The vertex count fix (symmetric min/max ratio) removes the main reason STRICT
was needed (to preserve star points at high epsilon). With symmetric scoring,
extra vertices at loose epsilon don't damage the score either.

This simplifies the matcher output from 3 results per match to 1, which also
simplifies the calling infrastructure and report.

**Note:** The `VectorVariant` enum and the three-result contract are part of
the existing API used by the test harness and report library. Whether to keep
the enum as a single-value enum or remove it entirely is an integration concern
— the matcher itself should be written for a single epsilon.

---



```
finalScore = (boundaryCountScore × W_COUNT)
           + (structuralScore    × W_MATCH)
           + (geometryScore      × W_GEOM)

// Achromatic refs only:
finalScore *= (1 - contamination)^4
```

Initial weight suggestion: `W_COUNT=0.15, W_MATCH=0.25, W_GEOM=0.60`.
To be tuned empirically once all layers are correctly implemented.

---

## Implementation Order

1. **Rewrite cluster discovery** — unified edge-first pipeline. Full-image
   morphological gradient. Classify all edge pixels by colour/luminance.
   Identical method for both ref and scene.

2. **Rewrite expansion loop** — selection by spatial proximity + relative
   contour area (vs candidate bbox, not full scene). No hue. No geometry.
   Anchor assigned to ref cluster by closest relative contour area.

3. **Bright/dark achromatic deduplication** — if matched set contains both a
   bright and dark achromatic entry with IoU > 0.5 AND contourArea ratio > 0.90,
   they trace the same physical boundary. Keep the larger; discard the other.
   Never deduplicate chromatic entries.

4. **Rewrite Layer 1** — boundary count from deduplicated set. Fix B
   chromatic contamination penalty retained.

5. **Rewrite Layer 2** — proximity × relative-size (vs regionBbox). Weighted
   by structural role (primary = largest contourArea). No colour.

6. **Rewrite Layer 3** — primary cluster only. Fix vertex count to symmetric
   min/max ratio. Increase solidity weight to 0.20 to correctly separate
   filled vs outline shapes.

7. **Rebalance weights** — empirically after all layers correct.

---

## Summary

| Step | Purpose | Uses colour? |
|---|---|---|
| Cluster discovery | Identify distinct boundaries by colour at edges | Yes — identification only |
| Expansion loop | Select spatially co-located scene clusters | No |
| Layer 1 | Boundary count matches ref; no contamination | No (count only) |
| Layer 2 | Spatial and proportional structural coherence | No |
| Layer 3 | Primary boundary geometry matches ref | No |

---

## Known Remaining Gaps

- **Gradient-colour FPs** — stripe/large-blob contour filter at candidacy stage
- **Anchor size constraint** — min/max bbox area gate to exclude noise and
  full-image blobs before the anchor loop
- **SegmentDescriptor noise reliability** — traversal matching under noisy
  backgrounds not yet analysed





