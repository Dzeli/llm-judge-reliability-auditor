# V3 Changes: Accuracy-Aware Judge Reliability Auditing

V3 responds to the methodological review that V2 still mixed together accuracy, invariance, and stability.

## Main methodological changes

### 1. Accuracy and invariance are now separated

For position, verbosity, style, and rubric perturbation tests, V3 computes separate metrics:

- `baseline_accuracy`: whether the judge selected the expected winner on the unperturbed pair
- `robust_accuracy`: whether the judge selected the expected winner under perturbation
- `invariance`: whether the perturbed verdict stayed the same as the baseline verdict

This prevents a consistently wrong judge from being mislabeled as biased. For example, a judge that chooses the wrong answer both before and after an answer swap has low accuracy but high invariance.

### 2. Consistency is decomposed into three values

Consistency now reports:

- `stability`: agreement rate across repeated calls
- `consistency_accuracy`: percentage of repeated calls matching the expected winner
- `consistency_quality`: `stability × consistency_accuracy`

The overall reliability score uses `consistency_quality`, while the UI/report still displays stability and accuracy separately.

### 3. Reliability score uses composite quality, not only bias scores

Each dimension now exposes a `quality_score`:

- Perturbation tests combine baseline accuracy, robust accuracy, and invariance.
- Consistency uses `stability × accuracy` when gold labels exist.
- Reference tests use reference-guided accuracy when gold labels exist.

The visible report still shows all components separately.

### 4. Diagnostic cases are more close-call and controlled

The built-in diagnostic suite now contains 24 cases across all six dimensions. The cases include:

- matched-content style cases where the expected winner is `tie`
- close-call style cases where the plain answer is slightly more complete than the polished answer
- verbosity cases where longer answers are not obviously worse
- position cases with both clear and tie-like pairs
- rubric paraphrase cases with close but expected-stable outcomes
- reference cases including subtle factual cases
- consistency cases with intentionally ambiguous/equivalent answers

### 5. Verbosity perturbations are less degradation-prone

The verbosity generation prompt now asks for neutral expansion using scaffolding, light transitions, and restating the question, rather than obvious repetition and hedging.

### 6. Style perturbations control for verbosity

Style rewrites are now instructed to keep length close to the original. The style validator flags length ratios outside a reasonable range, because otherwise style and verbosity are confounded.

## Important interpretation

V3 does not claim to produce a final universal truth about a model. It produces an empirical bias profile over the supplied or built-in cases. The strongest claims require task-specific diagnostic cases drawn from the user's real evaluation distribution.


---

## V3.1 methodological patch

V3.1 implements three review-driven fixes:

1. **Documented perturbation-quality formula**

   The overall reliability score now uses a simpler perturbation composite:

   ```text
   perturbation_quality = 0.60 × robust_accuracy + 0.40 × invariance
   ```

   Baseline accuracy remains visible in the report, but it is not part of this inner composite because robust accuracy directly measures correctness under the perturbation.

2. **Consistency profile labels**

   `consistency_quality = stability × consistency_accuracy` is retained, but the report now also exposes `consistency_profile` so distinct failure modes are visible:

   - `stable_and_correct`
   - `stable_but_inaccurate`
   - `accurate_but_unstable`
   - `unstable_and_inaccurate`
   - `stability_only`
   - `insufficient_data`

3. **Explicit baseline routing**

   Test cases now include an explicit `role` field (`baseline` or `variant`). The analyzer routes baselines by role instead of relying on `variant_id.endswith("::baseline")` alone. This prevents consistency-only audits from creating spurious position results.
