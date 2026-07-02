# Patch Notes: V2 Corrections

This patch fixes implementation gaps found after reviewing the first V2 archive.

## Fixed

- Ensured `diagnostics/builtin_cases.jsonl` is present and packaged.
- Added tests proving the diagnostic data loads and covers all six bias dimensions.
- Added unit tests for:
  - bias analyzer behavior,
  - consistency calculation,
  - reference-guided accuracy,
  - weighted reliability score,
  - metric-summary calculation,
  - style perturbation validation.
- Fixed style validation so both rewritten answers are validated in each style-bias case.
- Fixed single-pair consistency analysis so the baseline prompt counts as run 1.

## Verification

Local checks run successfully:

```bash
python -m compileall .
pytest -q
```

Result:

```text
10 passed
```

Pytest reports collection warnings because project classes are named `TestType` and `TestCase`; these warnings do not indicate failing tests.
