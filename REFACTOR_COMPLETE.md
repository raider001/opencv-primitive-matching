# VectorMatcher Refactoring - COMPLETE

**Date:** 2026-03-22  
**Status:** ✅ All refactoring steps completed and verified

---

## Changes Made

### Step 1: CandidateFilter Integration ✅
**Status:** Already integrated (no changes needed)
- `CandidateFilter.applyConnectedComponentFilter()` — line 175
- `CandidateFilter.applyGlobalSizeFilter()` — line 181
- `CandidateFilter.computeErosionDepth()` / `reExtractTopCandidates()` — line 188

### Step 2: AnchorMatcher Integration ✅
**Replaced:** ~60 lines of inline anchor assignment and expansion logic  
**With:**
```java
RefCluster anchorRef = AnchorMatcher.assignAnchorToRef(anchor, anchorBboxArea, refClusters);
AnchorMatcher.MatchResult result = AnchorMatcher.expandFromAnchor(
        anchor, anchorRef, candidates, refClusters, sceneDiag);
```

### Step 3: BboxExpander Integration ✅
**Replaced:** ~170 lines of inline bbox expansion  
**With:**
```java
bestBbox = BboxExpander.expandBbox(bestBbox, bestAnchor, matched,
        candidates, refClusters, referenceId.name(), descriptor.sceneArea);
```
**Sub-tasks:** Rewrote BboxExpander's `unionConcentricAndOverlappingSiblings()` with full
4-pass logic; added `GeometryUtils.rectsIntersect()`.

### Step 4: Anchor Re-selection Extraction ✅
**Replaced:** ~70 lines of inline anchor re-selection logic  
**With:**
```java
var resel = reSelectAnchor(bestScore, bestAnchor, bestBbox,
        anchorScores, anchorBboxes, anchorEntries);
bestBbox   = resel.bbox;
bestAnchor = resel.anchor;
```

### Step 5: Dead Code Cleanup ✅
**Removed from VectorMatcher:**
- `reExtractTopCandidates()` + `reExtractCandidate()` (fixed `EROSION_TOP_K` compile error)
- `contourArea()`, `rectCentre()`, `rectsIntersect()` (unused/migrated)
- `unionConcentricAndOverlappingSiblings()` (migrated to BboxExpander)
- `VM_BBOX_DEBUG` field (migrated to BboxExpander)
- `primaryScene` variable and debug printf block (marked for removal)
- Dead `else` block in anchor loop (always-false condition)
- Dangling Javadoc on `scoreRegion`
- Extracted `isShapeTypeCompatible()` from 18-line inline check

---

## Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| VectorMatcher total lines | ~1,516 | 1,121 | **-395 lines (-26%)** |
| `runMatch()` method | ~254 lines | ~163 lines | **-91 lines (-36%)** |
| Dead code methods | 8+ | 0 | **All removed** |
| Helper classes active | 2/4 | 4/4 | **All integrated** |

---

## Test Results

- **Tests:** 155, **Failures:** 0, **Errors:** 0
- **Scores/IoU:** Identical to baseline (verified via test log comparison)

---

## Architecture After Refactoring

```
VectorMatcher.runMatch()          (~145 lines, orchestrator)
  ├── CandidateFilter              (filtering pipeline)
  │   ├── applyConnectedComponentFilter()
  │   ├── applyGlobalSizeFilter()
  │   └── reExtractTopCandidates()
  ├── AnchorMatcher                (anchor selection & expansion)
  │   ├── assignAnchorToRef()
  │   └── expandFromAnchor()
  ├── RegionScorer                 (3-layer scoring)
  │   ├── score()                    → RegionScore record
  │   ├── scoreBoundaryCount()       Layer 1
  │   ├── applyContaminationPenalty()  Fix B
  │   ├── scoreStructuralCoherence() Layer 2
  │   ├── scoreGeometry()            Layer 3
  │   ├── findMatchedEntryForRef()
  │   └── clampRect()
  ├── reSelectAnchor()             (post-loop bbox correction, internal)
  └── BboxExpander                 (post-score bbox refinement)
      ├── expandBbox()
      ├── estimateScale()
      ├── computeAreaCaps()
      ├── trimAndUnionMatched()
      └── unionConcentricAndOverlappingSiblings()
```

## Verification Checklist

- [x] Code compiles without errors
- [x] All 4 helper classes fully integrated
- [x] Dead code removed
- [x] Null checks preserved
- [x] Tests: 155 run, 0 failures, 0 errors
- [x] Score/IoU values unchanged from baseline
- [x] Colour-agnostic guardrails preserved
