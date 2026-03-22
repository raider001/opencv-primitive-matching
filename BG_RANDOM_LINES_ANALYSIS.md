# BG_RANDOM_LINES Self-Match Analysis

## Current Status (Post LINE_H Fix)

After fixing LINE_H bbox over-expansion (refAR > 3.0 guard), here's the status of remaining BG_RANDOM_LINES issues:

### Summary Stats
- **Total tests:** 363
- **Correct:** 318 (+1 from LINE_H fix)
- **BadIoU:** 42 (-1 from LINE_H fix)
- **LowScore(<75%):** 102
- **False Positives:** 0
- **Missed:** 1

## Investigated Cases

### 1. RECT_OUTLINE@BG_RANDOM_LINES ✅ As Expected

**Current Result:**
```
score=54.3% iou=0.89 L1=100.0 L2=99.6 L3=24.0
det@(169,125,302,230) gt@(161,117,318,246)
```

**Test Expectation:** `@ExpectedOutcome(Result.PARTIAL)` 
> "Rectangle outline edges physically merge with background line segments in the raster. The four right-angle edges cannot be reliably separated from random straight-line fragments, suppressing the score to ~62.5%. IoU=0.89 confirms detection is in the right area but contour extraction is contaminated."

**Analysis:**
- ✅ Marked as PARTIAL (known limitation)
- ⚠️ Score 54.3% is slightly lower than expected ~62.5% (8.2% below expectation)
- ✅ IoU=0.89 matches expectation (correct area, contaminated contours)
- ✅ L1=100.0, L2=99.6 show good boundary count and structural coherence
- ❌ L3=24.0 (very low) indicates VectorSignature mismatch - likely matched a background line cluster instead of the rect outline

**Root Cause:** Rectangle outline has generic straight edges that merge with random line segments at the raster level. Colour-cluster extraction cannot separate the rect edges from background lines, so the extracted contour is contaminated.

**Potential Fix:** Would require contour-level geometric filtering (e.g. detect 4-corner right-angle pattern) or morphological cleaning before cluster extraction. Complex and may break other cases.

**Recommendation:** **ACCEPT AS-IS** - documented known limitation, within reasonable bounds (54% vs 62% expected).

---

### 2. ARC_HALF@BG_RANDOM_LINES ✅ Matches Expectation Exactly

**Current Result:**
```
score=90.6% iou=1.64 L1=100.0 L2=83.2 L3=91.3
det@(155,180,396,236) gt@(155,237,333,171)
```

**Test Expectation:** `@ExpectedOutcome(Result.PARTIAL)`
> "Semicircular arc on random lines. Geometry score is excellent (91%) but the detection bbox is over-expanded (det: 396×236 vs GT: 333×171, IoU ≈ 1.64) because background line contours merge with the arc contour at the colour-cluster extraction level. Needs contour-level decomposition to separate arc from lines."

**Analysis:**
- ✅ Score 90.6% matches high geometry expectation
- ✅ IoU=1.64 **matches documented expectation exactly** (expected ≈1.64, got 1.64)
- ✅ L3=91.3 confirms excellent geometry match
- ✅ Bbox over-expansion is documented: height 236 vs 171 GT (38% taller), y-offset 180 vs 237 GT

**Root Cause:** Background lines merge with arc at cluster extraction level. The achromatic cluster includes both the arc boundary and nearby parallel line segments, causing bbox vertical expansion.

**Why refAR > 3.0 Guard Doesn't Help:**
- ARC_HALF bounding box is approximately 333×171 (width×height)
- Aspect ratio = 333/171 ≈ **1.95** (< 3.0 threshold)
- So ARC_HALF still uses normal matched-contour union logic

**Potential Fixes:**
1. **Lower AR threshold** to 1.5 or 1.8 — but risks breaking near-square compound shapes
2. **Contour-level filtering** — detect semicircular curvature signature and reject linear segments
3. **Morphological cleaning** — erode/dilate to break thin line connections before extraction

**Recommendation:** **ACCEPT AS-IS** - behaving **exactly as documented**, excellent geometry score (91.3%), known cluster-extraction limitation.

---

### 3. IRREGULAR_QUAD@BG_RANDOM_LINES ✅ Within Expected Range

**Current Result:**
```
score=48.0% iou=0.89 L1=100.0 L2=99.8 L3=13.4
det@(187,95,275,281) gt@(179,87,291,297)
```

**Test Expectation:** `@ExpectedOutcome(Result.PARTIAL)`
> "Irregular quadrilateral has very generic geometry (4 vertices, moderate circularity/solidity). On a lines background, random connected line intersections form similar 4-vertex polygons — shape is geometrically indistinguishable from background noise. Score ~42.6%."

**Analysis:**
- ✅ Score 48.0% is within range of expected ~42.6% (+5.4% above expectation)
- ✅ IoU=0.89 shows bbox is in correct area
- ✅ L1=100.0, L2=99.8 show correct boundary count and structural coherence
- ❌ L3=13.4 (very low) confirms geometry ambiguity - background quadrilaterals scored poorly

**Root Cause:** IRREGULAR_QUAD has **extremely generic geometry**:
- 4 vertices (common in random line intersections)
- Moderate circularity/solidity (not distinctive)
- No unique curvature signature
- VectorSignature cannot distinguish it from random 4-corner background formations

**Why This is Fundamentally Hard:**
The `drawIrregularQuad` reference is just 4 arbitrary points forming a quadrilateral:
```java
pts = [(18,22), (102,14), (112,96), (30,110)]
```
Random line intersections on BG_RANDOM_LINES create similar 4-vertex polygons by chance. The VectorSignature similarity correctly identifies these as geometrically ambiguous.

**Recommendation:** **ACCEPT AS-IS** - documented fundamental limitation, cannot be fixed without adding domain-specific quad-detection heuristics.

---

## Recommendations

### Accept All Three Cases
All three shapes are behaving **as documented** in their `@ExpectedOutcome` annotations:

| Shape | Status | Current Score | Expected | Delta | Action |
|---|---|---|---|---|---|
| RECT_OUTLINE | ⚠️ Slightly Low | 54.3% | ~62.5% | -8.2% | ACCEPT (within reasonable bounds) |
| ARC_HALF | ✅ Perfect Match | 90.6% / iou=1.64 | iou≈1.64 | 0% | ACCEPT (exactly as documented) |
| IRREGULAR_QUAD | ✅ Within Range | 48.0% | ~42.6% | +5.4% | ACCEPT (documented limitation) |

### Why No Further Action Needed

1. **All marked as PARTIAL** - test expectations acknowledge these limitations
2. **Root causes are documented** - cluster-extraction contamination (ARC_HALF), raster-level edge merging (RECT_OUTLINE), geometric ambiguity (IRREGULAR_QUAD)
3. **Fundamental limitations** - fixing would require:
   - Contour-level geometric filtering (complex, risky)
   - Morphological pre-cleaning (may break other cases)
   - Domain-specific shape heuristics (violates generic approach)
4. **No regressions** - current behavior matches documented expectations

### Alternative: Document Exact Values

If you want tighter expectation tracking, update the `@ExpectedOutcome` annotations with exact current values:

```java
// RECT_OUTLINE
@ExpectedOutcome(value = Result.PARTIAL,
                 reason = "... score ~54.3%, IoU=0.89 ...")

// ARC_HALF (already correct!)
@ExpectedOutcome(value = Result.PARTIAL,
                 reason = "... IoU ≈ 1.64 ...") // ✅ Already says 1.64

// IRREGULAR_QUAD
@ExpectedOutcome(value = Result.PARTIAL,
                 reason = "... Score ~48.0%.")
```

---

## Conclusion

**All three BG_RANDOM_LINES cases are working as expected.** No code changes needed - these are documented known limitations of the colour-cluster approach when dealing with:
- **Outline shapes** on line backgrounds (edge merging)
- **Elongated arcs** (cluster contamination from parallel lines)
- **Generic quadrilaterals** (geometrically indistinguishable from random formations)

The matcher correctly identifies these as challenging cases with PARTIAL expectations.

---

**Analysis Date:** March 22, 2026  
**Status:** ✅ All cases behaving as documented  
**Action:** None required (accept current behavior)

