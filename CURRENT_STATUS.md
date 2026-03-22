# VectorMatcher Refactoring - Current Status

## ✅ BASELINE ESTABLISHED

**Test Results:** 155 tests, **0 failures, 0 errors**  
**Baseline Log:** `test_run_baseline.log`  
**Date:** 2026-03-22

---

## What Was Already Done

1. ✅ **Created helper classes:**
   - `CandidateFilter.java` - candidate filtering logic
   - `AnchorMatcher.java` - anchor matching logic  
   - `BboxExpander.java` - bounding box expansion logic
   - `GeometryUtils.java` (assumed to exist) - geometry utilities

2. ✅ **Fixed accessor method calls:**
   - SceneContourEntry record accessors: `.bbox()`, `.area()`, `.sig()`, etc.
   - Fixed over-replacements for non-record fields (RefCluster, ColourCluster)

3. ✅ **Replaced geometry helper calls:**
   - `GeometryUtils.centreDist()`
   - `GeometryUtils.unionRect()`
   - `GeometryUtils.bboxIoU()`

4. ✅ **All tests passing** - perfect baseline!

---

## Current VectorMatcher State

- **Total lines:** ~1516
- **runMatch() method:** lines 176-429 (~254 lines)
- **Problem:** `runMatch()` still contains inline logic instead of using the helper classes

---

## Next Steps (Incremental Refactoring)

### RULE: After EACH step, run tests. If ANY test fails, ROLLBACK immediately.

### Step 1: Verify Helper Classes Are Ready

Before integrating, check that helper classes have the right methods:

```powershell
# Check CandidateFilter has filterCandidates() method
Get-Content "src\main\java\org\example\matchers\vectormatcher\CandidateFilter.java" | Select-String "public static.*filterCandidates"

# Check AnchorMatcher has findBestAnchor() method  
Get-Content "src\main\java\org\example\matchers\vectormatcher\AnchorMatcher.java" | Select-String "public static.*findBestAnchor"

# Check BboxExpander has expandBbox() method
Get-Content "src\main\java\org\example\matchers\vectormatcher\BboxExpander.java" | Select-String "public static.*expandBbox"
```

### Step 2: Read `runMatch()` to Understand Current Flow

```powershell
# See the current runMatch implementation
Get-Content "src\main\java\org\example\matchers\VectorMatcher.java" -TotalCount 450 | Select-Object -Last 280
```

### Step 3: ONE SMALL CHANGE - Replace Candidate Filtering

**Goal:** Replace inline candidate filtering in `runMatch()` with a call to `CandidateFilter.filterCandidates()`

**Process:**
1. Identify the exact lines in `runMatch()` that do filtering
2. Create a SINGLE replacement (one method call)
3. Test immediately
4. If passing → commit; if failing → rollback

**Test command:**
```powershell
mvn clean compile
mvn '-Dtest=org.example.vectormatcher.VectorMatchingTest' test
# Must see: 155 tests, 0 failures
```

### Step 4: ONE SMALL CHANGE - Replace Anchor Matching

Same process as Step 3, but for anchor matching logic.

### Step 5: ONE SMALL CHANGE - Replace Bbox Expansion

Same process as Step 3, but for bbox expansion logic.

---

## Success Criteria (Final State)

- ✅ All 155 tests still passing (0 failures)
- ✅ `runMatch()` reduced to < 100 lines
- ✅ Logic delegated to helper classes
- ✅ Code is more maintainable

---

## Rollback Command (If Something Breaks)

```powershell
# Rollback VectorMatcher.java to last working state
git checkout -- src/main/java/org/example/matchers/VectorMatcher.java

# Re-run tests to confirm rollback worked
mvn '-Dtest=org.example.vectormatcher.VectorMatchingTest' test
```

---

## CRITICAL RULE

**NEVER make multiple changes at once.**  
**ALWAYS test after EACH change.**  
**ROLLBACK immediately if tests fail.**

This ensures we can pinpoint exactly what change caused a problem.

