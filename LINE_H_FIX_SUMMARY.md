# LINE_H@BG_RANDOM_LINES Bbox Over-Expansion Fix

## Issue
LINE_H (horizontal line) on BG_RANDOM_LINES background had massive bbox over-expansion:
- **Before:** `iou=8.22  det@(149,237,342,74)` — detected height = 74px (8× too tall!)
- **Ground truth:** `gt@(149,237,342,9)` — expected height = 9px

## Root Cause
In `BboxExpander.trimAndUnionMatched()` (Step C), the matcher was unionizing all matched contours from the same achromatic cluster. For LINE_H on a random-lines background:
1. Anchor expansion matched the actual LINE_H (horizontal line)
2. But also matched nearby background horizontal lines from the same achromatic cluster
3. Step C unioned these together, causing vertical growth: 342×9 → 342×74

The actual LINE_H has aspect ratio ~56:1 (extremely elongated), so the anchor bbox alone captures the complete extent. Additional matched contours on busy backgrounds are almost always background noise.

## Solution
Added an aspect ratio check in `BboxExpander` to skip matched-contour union for extremely thin shapes:

### Code Changes

**File:** `src/main/java/org/example/matchers/vectormatcher/components/BboxExpander.java`

1. **Compute refAR before Step C** (lines 70-77):
   ```java
   // ── Compute reference aspect ratio (used in Steps C & D) ────────────
   double refMaxDim = Math.max(refFullBbox.width, refFullBbox.height);
   double refMinDim = Math.min(refFullBbox.width, refFullBbox.height);
   double refAR = (refMinDim > 0) ? refMaxDim / refMinDim : 1.0;

   // ── Step C: Union co-located matched contours ─────────────────────────
   Rect expandedBbox = trimAndUnionMatched(anchorBbox, matched, bestAnchor,
           refClusters, caps, referenceId, anchorDiag, refAR);
   ```

2. **Updated trimAndUnionMatched signature** to accept `refAR` parameter (line 188)

3. **Added AR guard in Step C** (lines 224-239):
   ```java
   // Skip union for extremely elongated shapes (AR > 3.0) — on busy backgrounds,
   // matched entries may include nearby LINE_SEGMENT-like background contours from
   // the same cluster (e.g. LINE_H on BG_RANDOM_LINES), and unionizing them causes
   // the bbox to grow perpendicular to the primary axis (e.g. 342×9 → 342×74).
   if (refAR <= 3.0) {
       for (SceneContourEntry m : matched) {
           // ...union logic...
       }
   } else if (VM_BBOX_DEBUG) {
       System.out.printf("[STEP-C-SKIP] %s: refAR=%.2f (>3.0) — skip matched-contour union%n",
               referenceId, refAR);
   }
   ```

4. **Step D already had AR guard** (lines 90-95) — skips sibling expansion for refAR > 3.0

### Documentation Updates

**File:** `src/main/java/org/example/matchers/vectormatcher/README.md`

- Updated Stage 5, Step C description (lines 188-196) to document the refAR > 3.0 skip
- Updated Stage 5, Step D description (lines 198-207) to clarify it also uses refAR > 3.0 guard

## Results

### Before Fix
```
// BIOU LINE_H on BG_RANDOM_LINES LINE_H  score= 82.3% iou=8.22  
     det@(149,237,342,74)  gt@(149,237,342,9)
```
**Status:** BadIoU (wrong location) — height 74px vs 9px

### After Fix
```
// OK   LINE_H on BG_RANDOM_LINES LINE_H  score= 82.3% iou=1.00  
     det@(149,237,342,9)  gt@(149,237,342,9)
```
**Status:** Correct — perfect bbox match!

### Test Suite Impact
- **Before:** 363 Total | 317 Correct | 43 BadIoU | 0 FP | 1 Missed
- **After:** 363 Total | **318 Correct** | **42 BadIoU** | 0 FP | 1 Missed
- **Gain:** +1 correct detection, -1 BadIoU
- **All 155 tests pass** with no regressions

## Affected Shapes
The AR > 3.0 threshold primarily affects:
- **LINE_H** (aspect ratio ~56:1)
- **LINE_V** (aspect ratio ~56:1)

Other elongated shapes (ELLIPSE_H, ELLIPSE_V, ARC_HALF) have AR ≈ 2.0, so they still use the normal union logic.

## Design Principles Maintained
✅ **Colour-agnostic scoring** — no colour terms in final match decision  
✅ **Geometry-driven** — bbox refinement based on structural properties (AR)  
✅ **Conservative expansion** — tighter caps for thin shapes prevent background pollution  
✅ **README in sync** — all changes documented per AGENTS.md maintenance rules

---
**Date:** March 22, 2026  
**Status:** ✅ Implemented, tested, documented  
**Test Run:** All 155 VectorMatchingTest cases pass

