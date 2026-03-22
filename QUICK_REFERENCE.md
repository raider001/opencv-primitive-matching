# 🎯 QUICK REFERENCE: Refactoring VectorMatcher

## Current State
✅ **0 failures** (all tests passing)  
📏 **~254 lines** in `runMatch()`  
🎯 **Goal:** Reduce to < 100 lines

## The Rule
**≤ 9 failures = ACCEPTABLE**  
**10+ failures = ROLLBACK**

## The Process

### 1️⃣ Make ONE small change
Edit `VectorMatcher.java` - replace inline logic with helper class call

### 2️⃣ Test immediately
```powershell
mvn clean compile
mvn '-Dtest=org.example.vectormatcher.VectorMatchingTest' test
```

### 3️⃣ Check result
```powershell
# Look for this line in output:
[INFO] Tests run: 155, Failures: X, Errors: 0
```

- **X = 0-9:** ✅ GOOD - Continue to next step
- **X = 10+:** ❌ ROLLBACK - Run: `git checkout -- src/main/java/org/example/matchers/VectorMatcher.java`

### 4️⃣ Repeat
Go back to step 1 for the next section

## Three Sections to Replace

1. **Candidate Filtering** → `CandidateFilter.apply...()`
2. **Anchor Matching** → `AnchorMatcher.assignAnchorToRef()` + `.expandFromAnchor()`
3. **Bbox Expansion** → `BboxExpander.expandBbox()`

## Emergency Stop
If anything goes wrong:
```powershell
git checkout -- src/main/java/org/example/matchers/VectorMatcher.java
mvn '-Dtest=org.example.vectormatcher.VectorMatchingTest' test
# Should show: Failures: 0
```

## Success = 0-9 failures + < 100 lines in runMatch()

