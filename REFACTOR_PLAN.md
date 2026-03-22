# Incremental Refactoring Plan for VectorMatcher

## Current State (Baseline)
- **Test Results:** 155 tests, 36 failures
- **File Size:** VectorMatcher.java ~1516 lines
- **Main method:** `runMatch()` at lines 176-429 (~254 lines)
- **Status:** Code compiles, tests run, accessor methods fixed

## Goal
Incrementally refactor VectorMatcher without breaking the current baseline (keep failures at 36 or less).

---

## Phase 1: Extract Helper Classes (NO LOGIC CHANGES)

### Step 1.1: Extract `CandidateFilter` from VectorMatcher
**Status**: ✅ ALREADY EXISTS
- File: `src/main/java/org/example/matchers/vectormatcher/CandidateFilter.java`
- Already contains filtering logic
- **Verification:** Run tests, confirm 36 failures remain

### Step 1.2: Extract `AnchorMatcher` from VectorMatcher  
**Status**: ✅ ALREADY EXISTS
- File: `src/main/java/org/example/matchers/vectormatcher/AnchorMatcher.java`
- Already contains anchor matching logic
- **Verification:** Run tests, confirm 36 failures remain

### Step 1.3: Extract `BboxExpander` from VectorMatcher
**Status**: ✅ ALREADY EXISTS
- File: `src/main/java/org/example/matchers/vectormatcher/BboxExpander.java`
- Already contains bbox expansion logic
- **Verification:** Run tests, confirm 36 failures remain

### Step 1.4: Extract `GeometryUtils` from VectorMatcher
**Status**: ✅ LIKELY EXISTS
- Check if `GeometryUtils` class exists
- Methods: `centreDist()`, `unionRect()`, `bboxIoU()`
- **Verification:** Compile check, no test run needed

---

## Phase 2: Verify Current Integrations

### Step 2.1: Verify VectorMatcher uses extracted classes
**Current replacements already done:**
- `GeometryUtils.centreDist()` ✅
- `GeometryUtils.unionRect()` ✅  
- `GeometryUtils.bboxIoU()` ✅
- SceneContourEntry accessor methods fixed (`.bbox()`, `.area()`, etc.) ✅

### Step 2.2: Run full test suite to establish REAL baseline
```powershell
mvn '-Dtest=org.example.vectormatcher.VectorMatchingTest' test
```
**Expected:** 155 tests, 36 failures (current state)
**Action:** Save this as baseline before ANY changes

---

## Phase 3: Incremental Integration (ONE METHOD AT A TIME)

### Step 3.1: Integrate `CandidateFilter` into `runMatch()`
**Objective:** Replace inline candidate filtering with `CandidateFilter.filterCandidates()`

**Changes:**
1. Identify candidate filtering code in `runMatch()` (lines ~200-250)
2. Replace with single call: `List<SceneContourEntry> filtered = CandidateFilter.filterCandidates(candidates, ...)`
3. **DO NOT** change any logic, just move to external call

**Verification:**
```powershell
mvn clean compile
mvn '-Dtest=org.example.vectormatcher.VectorMatchingTest' test
```
**Expected:** Still 36 failures
**Rollback if:** Failures increase to 37+

### Step 3.2: Integrate `AnchorMatcher` into `runMatch()`
**Objective:** Replace inline anchor matching with `AnchorMatcher.findBestAnchor()`

**Changes:**
1. Identify anchor scoring loop in `runMatch()` (lines ~260-350)
2. Replace with: `AnchorResult bestAnchor = AnchorMatcher.findBestAnchor(filtered, refClusters, ...)`
3. Update local variables to use returned `AnchorResult` fields

**Verification:**
```powershell
mvn clean compile
mvn '-Dtest=org.example.vectormatcher.VectorMatchingTest' test
```
**Expected:** Still 36 failures
**Rollback if:** Failures increase to 37+

### Step 3.3: Integrate `BboxExpander` into `runMatch()`
**Objective:** Replace inline bbox expansion with `BboxExpander.expandBbox()`

**Changes:**
1. Identify bbox expansion code in `runMatch()` (lines ~360-420)
2. Replace with: `Rect finalBbox = BboxExpander.expandBbox(bestBbox, bestAnchor, matched, candidates, refClusters, referenceId, sceneArea)`
3. Remove inline expansion logic

**Verification:**
```powershell
mvn clean compile
mvn '-Dtest=org.example.vectormatcher.VectorMatchingTest' test
```
**Expected:** Still 36 failures  
**Rollback if:** Failures increase to 37+

---

## Phase 4: Final Cleanup

### Step 4.1: Remove dead/unused code from VectorMatcher
- Remove commented-out sections
- Remove unused imports
- Remove unused private methods

**Verification:**
```powershell
mvn clean compile
mvn '-Dtest=org.example.vectormatcher.VectorMatchingTest' test
```
**Expected:** Still 36 failures

### Step 4.2: Update documentation
- Update method javadocs
- Update class-level documentation
- Verify AGENTS.md reflects new structure

---

## Rollback Strategy

At each step, if failures increase:
1. **Immediate rollback:** `git checkout -- <file>`
2. **Analyze:** Check which test started failing
3. **Debug:** Use `-Dtest=<specific-test>` to isolate issue
4. **Fix accessor calls:** Check if any field access needs `.field()` instead of `.field`
5. **Retry:** Make minimal fix and retest

---

## Success Criteria

- ✅ All 4 helper classes integrated into `runMatch()`
- ✅ Test baseline maintained: 155 tests, 36 failures (no regression)
- ✅ `runMatch()` method reduced from 254 lines to < 100 lines
- ✅ Code compiles without warnings
- ✅ All accessor methods use correct syntax

---

## Current Status: PHASE 2 - VERIFICATION

**Next Action:** Run test suite and confirm EXACT baseline (36 failures)
