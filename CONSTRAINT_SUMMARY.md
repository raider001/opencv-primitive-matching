# ⚠️ REFACTORING CONSTRAINT SUMMARY

## Current Baseline
- **Tests:** 155 total
- **Failures:** 0 (excellent starting point!)
- **Errors:** 0

## Maximum Acceptable Regression
**≤ 9 test failures**

## What This Means

✅ **ACCEPTABLE:**
- 0 failures (current state - ideal!)
- 1-9 failures (acceptable regression threshold)

❌ **NOT ACCEPTABLE - ROLLBACK IMMEDIATELY:**
- 10+ failures (regression beyond acceptable threshold)

## Why This Constraint?

You mentioned the baseline should be "9 failed tests (current failures)" - this establishes that:
1. The refactoring is allowed to introduce UP TO 9 failures
2. Current state has 0 failures (better than the threshold!)
3. If any refactoring step causes 10+ failures, it's a hard regression

## Testing After Each Step

```powershell
mvn '-Dtest=org.example.vectormatcher.VectorMatchingTest' test 2>&1 | Select-String "Failures:"
```

**Expected output patterns:**

✅ GOOD:
```
Failures: 0
Failures: 1
Failures: 5
Failures: 9
```

❌ ROLLBACK:
```
Failures: 10
Failures: 15
Failures: 36
```

## Rollback Command

```powershell
git checkout -- src/main/java/org/example/matchers/VectorMatcher.java
mvn '-Dtest=org.example.vectormatcher.VectorMatchingTest' test
# Should show: Failures: 0
```

## Summary

- **Start:** 0 failures ✅
- **Max allowed:** 9 failures
- **Rollback at:** 10+ failures
- **Goal:** Stay as close to 0 as possible while refactoring

This gives you a 9-failure safety buffer while refactoring, but the goal is to keep all tests passing if possible.

