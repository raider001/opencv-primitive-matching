# Why CandidateFilter Doesn't Match VectorMatcher

## Root Cause

**The `CandidateFilter` class was designed with DIFFERENT filtering logic than what VectorMatcher actually uses.** It's not just a refactored version - it's an ALTERNATIVE implementation with different behavior.

---

## Detailed Differences

### 1. `applyConnectedComponentFilter()`

| Feature | VectorMatcher | CandidateFilter | Impact |
|---------|---------------|-----------------|--------|
| Early return for ≤1 candidates | ✅ YES (line 1320) | ❌ NO | Minor |
| Achromatic threshold | ✅ 0.05 (vs 0.10) | ❌ Always 0.10 | **MAJOR - filters too aggressively** |
| Secondary bbox rule | ✅ YES (lines 1344-1355) | ❌ NO | **MAJOR - loses compound components** |

**Example:** For `COMPOUND_BULLSEYE`, achromatic outline rings need the 0.05 threshold to survive. CandidateFilter at 0.10 removes them → lower scores.

---

### 2. `applyGlobalSizeFilter()`

| Feature | VectorMatcher | CandidateFilter | Impact |
|---------|---------------|-----------------|--------|
| Early return for ≤1 candidates | ✅ YES (line 1392) | ❌ NO | Minor |
| Early return if maxArea ≤ 0 | ✅ YES (line 1398) | ❌ NO | Edge case |
| Fallback if filter empties list | ✅ `out.isEmpty() ? candidates : out` | ❌ Returns empty | **CRITICAL - can cause zero results** |

**Example:** If a self-match scene has only small contours, VectorMatcher keeps them all (fallback), but CandidateFilter returns empty → zero result.

---

### 3. `computeErosionDepth()`

| Feature | VectorMatcher | CandidateFilter | Impact |
|---------|---------------|-----------------|--------|
| Return value | ✅ Always `return 0` | ❌ Returns 1-3 based on solidity | **CRITICAL - applies unwanted morphological operations** |

**Comment in VectorMatcher (lines 1361-1369):**
> "Applying MORPH_OPEN — even at 1 px — rounds the corners of triangular colour sections and hexagon/circle outline strokes enough to shift their VectorSignature vertex angles, causing regressions on TRICOLOUR_TRIANGLE, HEXAGON_OUTLINE and BICOLOUR_CIRCLE_RING."

**CandidateFilter enables morphological opening** which VectorMatcher explicitly DISABLED to prevent regressions!

---

## Why Tests Fail

When using CandidateFilter:
1. **40 failures** because:
   - Achromatic contours filtered too aggressively (0.10 vs 0.05)
   - Compound component bbox rule missing
   - Empty-list fallback missing
   - Unwanted morphological opening enabled

2. **Self-match tests fail** (e.g., `lineHSelf` 66.7% vs >70%) because:
   - Filtering removes contours that contribute to the structural score
   - Missing bbox containment rule loses inner components

---

## Solution Options

### Option A: Sync CandidateFilter to Match VectorMatcher ✅
- Copy exact logic from VectorMatcher into CandidateFilter
- Make methods IDENTICAL line-for-line
- Risk: Redundant code duplication

### Option B: Don't Use CandidateFilter at All
- Keep filtering logic in VectorMatcher as private methods
- Only extract OTHER sections (anchor matching, bbox expansion)
- Safer - no risk of logic divergence

### Option C: Gradual Extraction ⭐ RECOMMENDED
- Extract only the methods that DO match (if any)
- Leave filtering in VectorMatcher for now
- Focus refactoring on runMatch() structure, not helper extractions

---

## Recommendation

**Do NOT use CandidateFilter in its current state.**

Instead:
1. ✅ Keep filtering methods as VectorMatcher private methods
2. ✅ Focus on extracting anchor/bbox logic (if those match)
3. ✅ Reduce runMatch() line count by simplifying its structure
4. ⚠️ Only extract helpers if they are PROVEN to match exactly

The goal is **line count reduction + readability**, not necessarily extraction to external classes if it causes behavior changes.

