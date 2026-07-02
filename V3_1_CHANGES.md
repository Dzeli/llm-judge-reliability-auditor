# V3.1 Changes

This patch addresses three methodology/code-review comments from the V3 review.

## 1. Perturbation quality formula

V3 used an undocumented inner composite. V3.1 replaces it with:

```text
perturbation_quality = 0.60 × robust_accuracy + 0.40 × invariance
```

The rationale is that robust accuracy is the direct measure of whether the judge is correct under perturbation, while invariance captures whether irrelevant transformations changed the verdict. Baseline accuracy remains visible as context but is not double-counted in the inner perturbation-quality score.

## 2. Consistency profile labels

The report still includes all three numerical values:

```text
stability
consistency_accuracy
consistency_quality = stability × consistency_accuracy
```

V3.1 adds `consistency_profile` to make asymmetric failure modes explicit. For example, a stable-but-wrong judge and an unstable judge can have similar numeric scores, but they now receive different profile labels.

## 3. Explicit baseline routing

`TestCase` now has:

```python
role: Literal["baseline", "variant"]
```

The analyzer uses this field to route baselines into the appropriate test analyses. This is clearer than relying on naming conventions and prevents consistency-only runs from producing a spurious position result.

## Verification

```text
python -m compileall .
pytest -q
14 passed
```
