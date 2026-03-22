# Incremental Refactoring Plan - READY TO START

## ✅ BASELINE CONFIRMED

**Test Results:** 155 tests, 0 failures, 0 errors  
**Baseline Log:** `test_run_baseline.log` (created 2026-03-22)

---

## Summary

You have already successfully:
1. Created helper classes (`CandidateFilter`, `AnchorMatcher`, `BboxExpander`)
2. Fixed all accessor method calls
3. Achieved a **passing test baseline** (0 failures!)

## What's Left

The `runMatch()` method in `VectorMatcher.java` (lines 176-429, ~254 lines) still contains **inline logic** that should be delegated to the helper classes.

---

## INCREMENTAL REFACTORING STEPS

### **Rule**: Replace ONE section at a time, test immediately, rollback if anything breaks.

---

### Step 1: Replace Candidate Filtering Logic

**What to do:**  
Find the candidate filtering code in `runMatch()` and replace it with calls to:
- `CandidateFilter.applyConnectedComponentFilter(candidates)`
- `CandidateFilter.applyGlobalSizeFilter(candidates)`

**How to verify:**
```powershell
mvn clean compile
mvn '-Dtest=org.example.vectormatcher.VectorMatchingTest' test
# MUST see: 155 tests, 0 failures
```

**If tests fail:** 
```powershell
git checkout -- src/main/java/org/example/matchers/VectorMatcher.java
```

---

### Step 2: Replace Anchor Matching Logic

**What to do:**  
Find the anchor assignment and expansion code in `runMatch()` and replace it with calls to:
- `AnchorMatcher.assignAnchorToRef(anchor, anchorBboxArea, refClusters)`
- `AnchorMatcher.expandFromAnchor(anchor, anchorRef, candidates, refClusters, sceneDiag)`

**How to verify:**
```powershell
mvn clean compile
mvn '-Dtest=org.example.vectormatcher.VectorMatchingTest' test
# MUST see: 155 tests, 0 failures
```

**If tests fail:** 
```powershell
git checkout -- src/main/java/org/example/matchers/VectorMatcher.java
```

---

### Step 3: Replace Bbox Expansion Logic

**What to do:**  
Find the bbox expansion code in `runMatch()` and replace it with a call to:
- `BboxExpander.expandBbox(bestBbox, bestAnchor, matched, candidates, refClusters, referenceId, sceneArea)`

**How to verify:**
```powershell
mvn clean compile
mvn '-Dtest=org.example.vectormatcher.VectorMatchingTest' test
# MUST see: 155 tests, 0 failures
```

**If tests fail:** 
```powershell
git checkout -- src/main/java/org/example/matchers/VectorMatcher.java
```

---

## Expected Final Result

- ✅ `runMatch()` reduced from ~254 lines to < 100 lines
- ✅ All logic delegated to helper classes
- ✅ **Still 0 test failures**

---

## Files Modified

- `src/main/java/org/example/matchers/VectorMatcher.java` (only this file!)

## Files NOT Modified

- `CandidateFilter.java` (already exists, don't change)
- `AnchorMatcher.java` (already exists, don't change)
- `BboxExpander.java` (already exists, don't change)

---

## CRITICAL SUCCESS FACTOR

**DO NOT make all changes at once.**  
**Make ONE change → TEST → Next change.**

If at any point tests fail, you know EXACTLY which change caused the problem because you only made ONE change since the last test.

---

## Ready to Start?

The baseline is solid. You can now begin Step 1: replacing the candidate filtering logic.

Good luck! 🚀

