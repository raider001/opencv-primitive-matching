# VectorMatcher — Technique Assessment & Gap Analysis

---

## What We Have Built

### Layer 1 — Colour Cluster Count Matching
The reference image is scanned for colour clusters via border-pixel sampling
(`extractFromBorderPixels`). The number of distinct colour clusters in the
**candidate region** is compared against the reference cluster count. An
exponential decay penalty is applied for mismatches, with a steeper penalty
(×2) for having *more* clusters than the reference.

### Layer 2 — Per-Cluster Contour Coverage
For each reference colour cluster, the best-matching scene contour is found by
hue compatibility. A cluster that cannot be matched contributes zero to the
score. Checks "is every colour component of the reference present in the
candidate region?".

### Layer 3 — VectorSignature Geometry
Per-shape geometric signature with 8 sub-components:

| Component | Weight | What it measures |
|---|---|---|
| Type (CIRCLE / CONVEX / CONCAVE / LINE) | 0.15 | Broad shape family, hard gate on incompatible types |
| SegmentDescriptor (traversal) | 0.38 | Primary structural signal — angle/length sequences |
| Topology | 0.10 | Connected-edge graph structure |
| Circularity (4π·area/perimeter²) | 0.13 | How round the shape is |
| Solidity (area/hull area) | 0.11 | Fill ratio / convexity |
| Vertex count | 0.08 | Polygon corner count |
| Angle histogram | 0.05 | Rotation-invariant edge-angle distribution |
| Aspect ratio (multiplicative gate) | ×multiplier | Fires only when mismatch > 15% |

### Supporting Mechanisms
- **Clutter penalty** — per-cluster, decays logarithmically with peer count
- **Ref signature override** — for all-achromatic refs, uses the foreground
  binary threshold sig rather than the background boundary contour sig
- **Contour safety** — border zeroing, frame-spanning filter, deduplication,
  self-intersection sanitisation before `convexityDefects`
- **AR multiplicative gate** — aspect ratio suppresses entire score when
  mismatch > 15%, using `(aspectScore/0.85)^3`

---

## Current Scores
- **Total: 85 | Correct: 42 | FP: 31 | BIOU: 12 | Missed: 0**

---

## Layer-by-Layer Interrogation

---

### Layer 1 — Does It Actually Work for Single-Colour Refs?

#### Initial Framing
The *scene* itself has multiple colour clusters (random circles, lines etc.),
so even for a single-achromatic reference, the cluster count comparison should
be informative — a candidate that absorbs background clusters should be
penalised.

#### Counter-Argument Round 1 (original — partially wrong)
The first counter-argument claimed the expansion loop never fires, so
`sceneClustersInRegion` is always 1 vs `refClusterCount` 1 → Layer 1 saturated.

**This was incorrect about `refClusterCount`.**

#### User's Counter-Argument — Two Ref Clusters
Even a single-colour ref image (e.g. RECT_FILLED: white rect on black
background) produces **two** achromatic ref clusters from `extractFromBorderPixels`:

- **Bright achromatic** (`val >= 100`): the white rect pixels → 1 cluster with
  the white rect's contour
- **Dark achromatic** (`val < 100`): the black background pixels → 1 cluster
  with the background's inner boundary (= the rect's outer edge)

So `refClusterCount = 2`, not 1. The `singleAchromatic` check
(`refClusterCount == 1 && all achromatic`) is therefore **FALSE** for
RECT_FILLED. The standard weights apply (`wGeom = 0.70`, `wCount = 0.10`,
`wMatch = 0.20`) and Layer 1 does produce a non-trivial score via
`diff = sceneClustersInRegion - 2`.

#### Why It Still Doesn't Discriminate — The Bright/Dark Collapse

The expansion loop for an achromatic anchor:

```
for (RefCluster rc : refClusters) {    // 2 iterations — bright-achromatic, dark-achromatic
    if (hueMatch(rc, anchor)) continue; // anchor achromatic → rc[0] achromatic → TRUE → SKIP
                                        //                   → rc[1] achromatic → TRUE → SKIP
}
```

`hueMatch` only checks `rc.achromatic == ce.achromatic`. Both ref clusters are
`achromatic=true`. Both are skipped. `matched = [anchor]` always.

Result: `sceneClustersInRegion = 1`, `diff = 1 - 2 = -1`,
`clusterCountScore = exp(-0.5) ≈ 0.607`.

This 0.607 penalty applies **identically to every anchor** — the clean rect
anchor and the merged blob anchor both produce matched=[anchor] with
sceneClustersInRegion=1. Layer 1 is still constant across all candidates.
It penalises all single-achromatic shapes equally, not selectively.

#### The Deeper Flaw — Bright/Dark Is Meaningful But Ignored

The user's counter-argument reveals a real structural issue: the bright
(white shape) and dark (black background) achromatic clusters are **distinct
things** in the reference, but `hueMatch` treats them as identical because
both have `achromatic=true`. The bright/dark split information carried in
`BRIGHT_VAL_THRESHOLD` is discarded at matching time.

What this means:
- The white rect in the scene belongs to the **bright achromatic** scene cluster
- The black background belongs to the **dark achromatic** scene cluster
- If the expansion loop respected the bright/dark distinction, an anchor from
  the bright cluster would satisfy ref cluster 0 (bright) but fail to expand
  into ref cluster 1 (dark) — unless a dark-cluster entry was pulled in
- `sceneClustersInRegion` would then be either 1 or 2 depending on whether a
  dark-cluster scene contour was included, giving a real Layer 1 signal

**This is not currently implemented. `hueMatch` merges all achromatic into
one equivalence class, discarding the bright/dark distinction.**

#### What Should Be Done

**Fix A — Respect bright/dark in `hueMatch`:**
Extend `hueMatch` to distinguish `brightAchromatic` from `darkAchromatic`.
A bright-cluster anchor would cover ref cluster 0 (bright), skip it, then
try to expand for ref cluster 1 (dark) — finding a dark-achromatic scene
contour and pulling it in. `sceneClustersInRegion = 2 = refClusterCount`,
Layer 1 = 1.0.

**However, Fix A does not discriminate between the blob and the correct rect.**
Both are bright-achromatic anchors. Both would expand to include a dark
background contour. Both end up with `sceneClustersInRegion = 2`. Layer 1
returns 1.0 for both. The signal washes out — Fix A is a structural
improvement to the architecture but does not solve the blob-vs-rect problem.

**Fix B — Bbox colour purity / spatial contamination check (the correct fix):**
For each candidate anchor bbox, compute how much the bbox spatially overlaps
with chromatic scene clusters using the cluster masks already present in
`SceneDescriptor`. If the ref is achromatic and the anchor bbox has significant
chromatic contamination, apply a penalty.

This IS the actual discriminator:
- Clean rect `(230,160,181,161)`: sits on dark background, minimal chromatic
  overlap — low contamination, no penalty
- Blob `(101,160,310,170)`: extends left into a region where coloured circles
  sit — measurable chromatic overlap from those circles — penalised

This check operates on the **actual pixel content of the scene** inside the
anchor bbox, not on the matched-set structure. It directly measures whether
a candidate bbox is absorbing background regardless of cluster matching logic.

**Fix B is the more accurate solution. Fix A is architecturally correct but
insufficient on its own — both are worth implementing, but Fix B is what
actually solves the discrimination problem.**

---

### Are Both Borders of a Single Graphic Used in Scene Matching?

#### The Question
A single achromatic graphic (e.g. white rect on black background) produces
two distinct contour sets in the scene:
- **Border A** — the outline of the white shape (from the bright-achromatic cluster mask)
- **Border B** — the inner boundary of the dark background (from the dark-achromatic cluster mask)

Are both of these being used in the match evaluation?

#### The Answer — Both Are Collected, But Only One Is Ever Used

Both borders ARE present in `collectSceneCandidates`. `SceneDescriptor.build()`
runs `SceneColourClusters.extract()` which produces both a bright-achromatic
cluster and a dark-achromatic cluster. Each cluster's mask goes through
`contoursFromMask()` and both sets of contours become `SceneContourEntry`
records in the candidates list.

However, in the anchor loop, only **one** contour is ever used per scoring:

```
matched = [anchor]   ← always just the anchor
// Expansion loop:
for (RefCluster rc : refClusters) {
    if (hueMatch(rc, anchor)) continue;  // anchor is achromatic
                                         // rc[0] bright-achromatic → hueMatch = TRUE → SKIP
                                         // rc[1] dark-achromatic  → hueMatch = TRUE → SKIP
}
// matched is still just [anchor]. Border B is never pulled in.
```

`hueMatch` returns true for ALL achromatic entries against ALL achromatic ref
clusters, so the expansion body is never entered. The second border — whether
it is the bright rect outline or the dark background boundary — **always sits
unused in the candidates list**.

`scoreRegion` only receives `matched = [anchor]` and evaluates geometry against
ONE border contour. The other border's geometric information is completely
ignored.

#### Why This Matters

The two borders are **redundant representations of the same shape** — they
trace the same geometric boundary from the inside and outside respectively.
Using only one means:
- Half the geometric evidence about the shape is discarded every time
- The choice of which border gets used is determined by which cluster the anchor
  comes from — an implementation accident, not a deliberate decision
- If the anchor happens to be from the dark-background cluster, its contour
  traces the *background* shape (a frame with a hole), which is geometrically
  very different from the foreground rect — producing a poor similarity score
  against the ref sig built from the foreground shape

#### What Fix A Would Actually Do

Fix A (respect bright/dark in `hueMatch`) would allow the expansion loop to
fire: a bright-cluster anchor would cover ref cluster 0 (bright), then expand
to pull in a dark-cluster contour for ref cluster 1 (dark). `matched` would
then contain **both** borders. `scoreRegion` would evaluate geometry against
both, average them, and the second border's agreement would corroborate the
match.

This still doesn't discriminate blob vs rect (as established earlier), but it
would mean **both borders contribute to the geometry score** rather than one
being silently discarded — making the overall score more geometrically accurate.

#### Conclusion

**No — only one of the two borders is ever used in matching.** Fix A would
fix this and use both. Fix B (chromatic contamination) is still the correct
discriminator for the blob-vs-rect problem. Both fixes are complementary and
both should be implemented.

### Gap 1 — Vertex Count Asymmetry
`min(vDet, vRef) / vRef` means a 20-vertex noisy blob scores 1.0 when matching
a 4-vertex rect. Extra vertices from noise are not penalised at all. Should use
a symmetric ratio: `min(vDet, vRef) / max(vDet, vRef)`.

### Gap 2 — Gradient-Colour False Positives (10 of 31 FPs)
Gradient backgrounds produce full-height/full-width stripe contours. These pass
geometry scoring because gradient stripes can have rectangular bboxes and
reasonable circularity. A stripe-shape filter (contour bbox spanning >60% of
scene height AND width asymmetrically) would eliminate these at candidacy time.

### Gap 3 — Anchor Size Has No Constraint
No lower or upper bound on anchor bbox area relative to the scene. Tiny noise
fragments and huge merged blobs are treated identically until geometry scores
them — which is too late if the geometry accidentally matches.

### Gap 4 — `random-circles RECT_FILLED` Persistent Miss (IoU = 0.55)
The matched anchor is a merged achromatic blob `(101,160,310,170)` rather than
the clean rect `(230,160,181,161)`. Both are in achromatic clusters. The
`otherHits` diagnostic confirms the clean rect scores 98.7% when scored in
isolation, meaning the matcher's internal path is selecting the blob over it.
Root cause not yet confirmed — likely the blob IS the single contour in the
isolated cluster, meaning the "n=1, penalty=1.0" cluster contains the merged
blob, not the clean rect.

### Gap 5 — SegmentDescriptor Accidental Matches in Noise
The primary discriminator (weight 0.38) is built from polygon approximation
traversal. In random-lines and random-mixed scenes, intersecting background
lines create complex merged contours whose traversal accidentally resembles
simple shapes. No evaluation of SegmentDescriptor reliability under noise
has been done.

### Gap 6 — Ref Signature vs Scene Signature Scale Mismatch
The ref image is 128×128. Scene is 640×480. Shapes are drawn at a fixed pixel
size in both (e.g. 181×161 in the scene, 103×79 in the ref). The `normalisedArea`
(contour area / image area) therefore differs by ~2.4×. The area-ratio gate in
`similarity()` uses a 10× tolerance, so this doesn't trigger, but it means
normalised area provides almost no discriminative signal cross-image.

### Gap 7 — Multi-Cluster Greedy Expansion
The expansion loop picks the single best scene contour for each remaining ref
cluster greedily. No backtracking. Sub-optimal for multi-colour shapes where a
better overall assignment exists.




