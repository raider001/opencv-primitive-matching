# Alternate Solution: Per-Colour Edge Grouping

## Overview

Instead of treating the scene as one flat greyscale edge map, decompose the scene
into **per-colour edge layers** — one layer per distinct colour present in the
reference image — and run edge detection independently on each layer.

This attacks the noise problem at the source: background edges that are not the
reference colour simply do not exist in the corresponding colour layer, so they
never reach contour extraction or signature comparison.

---

## Applicability

### Where this works well — cartoon-like scenes

This approach is most effective when the scene has a **low number of distinct
colours** with clear separation between them. Cartoon-style UIs, game graphics,
and icon-based interfaces are ideal:

- Few distinct hues (typically 3–8)
- High saturation — colours are well-separated in HSV space
- Clean boundaries between colour regions

### Where this degrades

- **Photographic scenes** — hundreds of distinct colours, K becomes very large,
  cost grows linearly with K
- **Gradient fills** — the HSV tolerance smears across many hue buckets,
  producing noisy isolated layers
- **Low-saturation scenes** — grey/white/black shapes are hard to isolate
  by hue alone; luminance-based isolation would be needed instead

---

## Cost Model

### The core constraint — cost grows linearly with K

Each additional reference colour adds:
- 1 × HSV mask pass (fast — single `Core.inRange` call, O(pixels))
- 1 × `findContours` call (dominant cost — O(pixels))
- 1 × contour set to score

| K | Overhead | Practical scenario |
|---|---|---|
| 1 | Baseline — identical to current | Single-colour icon |
| 2 | ~2× edge extraction | Outlined shape (fill + border) |
| 3 | ~3× edge extraction | Simple 3-colour graphic |
| 5+ | Grows linearly | Complex graphic — starts to hurt |
| 10+ | Expensive | Not suitable for this approach |

### The key insight — K is bounded by the reference, not the scene

The reference image is small (128×128) and controlled. The number of distinct
colours K is determined at **reference build time** and is fixed. For the
reference graphics we draw, K is typically 1–3. The scene complexity does not
affect K — only the reference does.

### Net cost can still be a reduction

On a noisy scene, the full flat edge map produces N contours — many of which are
noise that gets scored and rejected. With colour isolation, the same scene
produces far fewer contours per layer (noise that isn't the reference colour
simply doesn't exist in that layer). The saving in false candidate scoring can
outweigh the per-layer overhead when the background is complex.

---

## Colour Agnostic Matching

A critical property: **the reference colour is used to isolate the layer,
but the match itself is colour-agnostic**.

This means:
- A red triangle reference will correctly match a blue triangle in the scene
- The colour mask is applied to isolate *a* colour layer — whichever colour
  layer contains the triangle-shaped contour is the one that matches
- The `SegmentDescriptor` comparison operates purely on geometry, not colour

### How this works in practice

Instead of masking the scene to match the reference colour exactly, we
**iterate over the scene's own colour clusters** and test each one against the
reference geometry:

```
Reference → K colour groups → K geometric descriptors

Scene → colour quantise → M scene colour clusters (M independent of reference)
For each scene cluster i:
    Extract contours from cluster i
    Compare against all K reference descriptors
    Record best geometric match
```

This way a red-reference circle matches a blue scene circle because the blue
cluster's contour geometry matches the reference geometry regardless of hue.

### The colour is a gate, not a filter

The role of colour in this pipeline is:
1. **Separate** overlapping shapes in the scene that would otherwise merge contours
2. **Reduce** the number of candidates per comparison pass
3. **Not** to require that scene and reference colours match

---

## Controlling K — keeping it fast

### Quantise the reference colours at build time

Rather than treating every distinct pixel colour as a separate group, quantise
the reference image into at most **N_MAX colour clusters** (suggested N_MAX = 4)
using k-means or a simple HSV histogram bucketing. This hard-caps K regardless
of how many colours are drawn in the reference.

### Merge similar hues

Two reference colours within a configurable HSV distance threshold are merged
into one group. A yellow fill and a slightly-off-yellow highlight become one
group — one mask pass, one contour set.

### Skip achromatic groups

Black, white, and grey (low saturation) are separated by luminance rather than
hue. These can be handled as a single "achromatic" group with a luminance
threshold rather than as separate HSV masks — keeping the hue-based groups
focused on chromatic colours only.

### Lazy evaluation

Only run colour isolation if the fast single-channel match falls below a
confidence threshold. Single-channel match first (cheap), colour isolation
only if it's inconclusive (expensive but rare).

---

## Implementation Sketch

### Reference side (build time, once per ReferenceId)

```java
public final class ColourGroupDescriptor {
    public final HsvRange      hsvRange;      // the colour gate for scene isolation
    public final SegmentDescriptor geometry;  // shape geometry (colour-agnostic)
    public final double        normArea;      // normalised area of this group
}

// Built once:
List<ColourGroupDescriptor> refGroups = ColourGroupDescriptor.buildFromRef(refMat, N_MAX);
```

### Scene side (match time)

```java
// Quantise scene into M colour clusters (M ≤ some scene cap, e.g. 8)
List<SceneColourCluster> sceneClusters = SceneColourCluster.extract(sceneMat);

double bestScore = 0;
for (SceneColourCluster cluster : sceneClusters) {
    List<MatOfPoint> contours = extractContours(cluster.maskedMat);
    for (ColourGroupDescriptor refGroup : refGroups) {
        double geomScore = scoreGeometry(contours, refGroup.geometry);
        if (geomScore > bestScore) bestScore = geomScore;
    }
}
```

### API flag — disabled by default

```java
// New variant — colour-grouped matching
VectorVariant.VECTOR_NORMAL_COLOUR_GROUPED
```

---

## Relationship to Current Work

| Concern | Current approach | This approach |
|---|---|---|
| Noise merging with shape | `SegmentDescriptor` curvature gate | Colour isolation removes noise before extraction |
| Corner accuracy | `approxPolyDP` on clean threshold contour | Clean contour from colour-isolated layer |
| Colour requirement | Scene must match reference colour (CF) | Colour-agnostic — any colour can match |
| Cost | 1 × edge extraction | K × edge extraction (K = reference colour count, typically 1–3) |
| Best for | Any scene | Cartoon/icon scenes with few distinct colours |

Both approaches are **complementary**:
- `SegmentDescriptor` runs inside each colour layer (cleaner input)
- Colour isolation provides the clean contours `SegmentDescriptor` needs

---

## When to Implement

1. Current: fix contour quality issues with the clean-threshold extraction approach
2. Next: implement colour-isolated contour extraction as the primary extraction path
3. Later: add colour-agnostic scene cluster matching for cross-colour shape detection
