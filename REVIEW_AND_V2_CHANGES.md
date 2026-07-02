# LLM Judge Reliability Auditor — Review and V2 Change Log

This document summarizes the review of the original V1 implementation and the main upgrades implemented in V2.

---

## 1. Executive Summary

The V1 project was a strong capstone prototype: it had a clear problem, a useful UI, a clean Pydantic data model, deterministic scoring logic, and a practical pipeline for testing LLM-as-judge behavior.

The main concern was conceptual rather than structural: V1 audited judge behavior on a single answer pair and then reported a global-looking reliability score. That made the tool useful as a perturbation probe, but not yet strong enough to claim a full judge reliability certificate.

V2 keeps the original single-pair workflow but adds a more scientifically credible diagnostic-suite mode. The new framing distinguishes between:

```text
single-example fragility
vs.
controlled diagnostic bias profile
```

This makes the project more honest, more useful, and stronger as a GitHub/Kaggle portfolio project.

---

## 2. V1 Strengths

V1 already had several strong foundations:

- Clear product idea: empirically test whether an LLM judge is trustworthy before using it in a real evaluation pipeline.
- Six intuitive audit dimensions:
  - position bias
  - verbosity bias
  - style bias
  - consistency / response stability
  - rubric sensitivity
  - reference sensitivity
- Clean separation between generation, judging, deterministic analysis, and report writing.
- Pydantic models for structured inputs, test cases, judge decisions, and reports.
- Gradio UI with multi-model comparison.
- ADK/MCP integrations, making the project feel like more than a local script.

These were all worth preserving.

---

## 3. Main V1 Limitations

### 3.1 Single-pair audit looked too global

V1 used one user-provided question and two answers to generate a reliability score. That is useful for debugging one evaluation instance, but it is not enough to estimate the general reliability of a judge model.

The recommended reframing was:

```text
V1 output should mean:
"This particular evaluation example appears stable/fragile under perturbations."

Not:
"This judge model is globally reliable/unreliable."
```

### 3.2 Baseline judge decision was treated as the anchor

V1 compared perturbation verdicts against the baseline judge verdict. This is practical, but fragile, because the baseline judge can be wrong.

Example:

```text
Baseline without reference: judge chooses B
With reference: judge chooses A
Gold winner: A
```

In this case, the verdict changed, but that is not necessarily bad. The reference may have corrected the judge.

### 3.3 Reference sensitivity was not the same as reference helpfulness

V1 measured whether the verdict changed with versus without a reference answer. However, a verdict change can be good or bad depending on the expected winner.

V2 therefore separates:

```text
reference influence = did the reference change the verdict?
reference helpfulness = did the reference move the judge closer to the expected winner?
```

### 3.4 Perturbations could accidentally change meaning

V1 used an LLM to generate padded answers, polished/plain rewrites, and rubric paraphrases. This is useful, but it introduces a hidden risk: the generator may accidentally add facts, remove important details, or change the criterion.

V2 adds perturbation validation metadata so users can inspect whether a mutation looks fair.

### 3.5 Reliability score needed confidence/context

A single scalar score is easy to understand, but it should be interpreted differently depending on whether it comes from:

```text
1 user example
or
many controlled diagnostic cases
```

V2 adds confidence and interpretation fields to make this explicit.

---

## 4. V2 Design Goals

V2 was designed around five goals:

1. Preserve the working V1 pipeline and UI concept.
2. Make the scoring more honest and less overclaiming.
3. Add a controlled diagnostic-suite mode with expected winners.
4. Distinguish invariance, accuracy, and reference helpfulness.
5. Make generated perturbations inspectable and partially validated.

---

## 5. Main V2 Upgrades

### 5.1 Two audit modes

V2 introduces two explicit modes.

#### Mode 1: `single_pair`

This is the V1-style workflow.

Input:

```text
question
answer_a
answer_b
optional reference_answer
rubric
judge model
selected tests
```

Output interpretation:

```text
A perturbation-based reliability probe for this specific answer pair.
```

This mode is useful for debugging one evaluation example.

#### Mode 2: `diagnostic_suite`

This is the new V2 workflow.

Input:

```text
judge model
built-in diagnostic cases
selected diagnostic categories
```

Output interpretation:

```text
A controlled bias profile across multiple cases with known expected winners.
```

This mode is much closer to a real evaluator.

---

### 5.2 Diagnostic case library

V2 adds a built-in JSONL diagnostic suite:

```text
diagnostics/builtin_cases.jsonl
```

Each case includes:

```text
question
answer_a
answer_b
expected_winner
rubric
bias_target
difficulty
rationale
reference_answer
```

The key improvement is `expected_winner`. This allows the auditor to measure whether the judge is correct, not only whether it is stable.

---

### 5.3 New diagnostic models

V2 adds new data models for diagnostic auditing:

```text
models/diagnostics.py
```

Core concepts include:

```text
DiagnosticCase
DiagnosticSuiteResult
DiagnosticCaseResult
```

These support multi-case evaluation and make it possible to compute category-level and overall diagnostic metrics.

---

### 5.4 More honest report schema

V2 extends the report structure with fields such as:

```text
audit_mode
confidence_level
score_interpretation
metric_summary
generated_variants
validation_results
```

This makes the output clearer and more transparent.

Instead of presenting every score as a global reliability certificate, V2 can say things like:

```text
Confidence: Low
Reason: Based on a single pair probe.
```

or:

```text
Confidence: Medium/High
Reason: Based on multiple controlled diagnostic cases.
```

---

### 5.5 Reference helpfulness instead of raw reference sensitivity

V2 changes the conceptual meaning of the reference test.

V1 asked:

```text
Did the verdict change with a reference answer?
```

V2 asks:

```text
Did the reference answer move the judge toward the known expected winner?
```

This is a major conceptual improvement because reference answers are supposed to change decisions when the original decision was wrong.

---

### 5.6 Perturbation validation

V2 adds:

```text
agents/perturbation_validator.py
models/validation.py
```

The validator records heuristic checks such as:

```text
length ratio
similarity estimate
possible factual drift
rubric paraphrase length drift
style/verbosity mutation metadata
```

This does not fully solve semantic validation, but it gives the user more visibility into whether a perturbation was fair.

The goal is to avoid silently blaming the judge when the generated mutation itself may have changed the task.

---

### 5.7 Generated variant metadata

V2 stores generated variants so users can inspect them later.

This matters because for an auditor, transparency is essential. If the tool says the judge has style bias, users should be able to inspect:

```text
original answer
plain rewrite
polished rewrite
padded answer
rubric paraphrases
```

Without this, the report is hard to trust.

---

### 5.8 Revised scoring philosophy

V2 keeps a simple reliability score for usability, but adds more context.

The recommended interpretation is now:

```text
Reliability score = summary indicator
Confidence level = how much to trust the score
Metric summary = why the score looks the way it does
```

This is more scientifically defensible than presenting one score without context.

---

### 5.9 Updated Gradio UI

V2 includes a cleaner app interface with support for:

```text
single-pair probe
built-in diagnostic suite
multi-model comparison
score interpretation
confidence display
```

This makes the product easier to explain in a demo.

---

### 5.10 Updated MCP tools

V2 adds MCP support for both workflows:

```text
audit_single_pair
run_diagnostic_suite
list_supported_models
```

This makes the auditor usable from external MCP-compatible clients.

---

### 5.11 Basic tests

V2 adds basic tests around scoring and report behavior:

```text
tests/test_scoring.py
```

The implementation was compile-checked and the included tests passed locally.

---

## 6. Important Conceptual Distinctions Added in V2

### 6.1 Invariance tests

These are tests where the verdict should usually remain stable.

Examples:

```text
answer order swap
rubric paraphrase with same meaning
format/style changes that preserve content
```

If the verdict changes, that may indicate fragility.

### 6.2 Accuracy tests

These require an expected winner.

Example:

```text
Answer A is factually correct.
Answer B is polished but wrong.
Expected winner: A.
```

The judge should choose A.

### 6.3 Reference helpfulness tests

These compare judge correctness with and without a reference answer.

A verdict change is only considered helpful if it moves the judge toward the expected winner.

---

## 7. Recommended GitHub/Kaggle Framing

The project should be framed as:

```text
LLM Judge Reliability Auditor is a diagnostic evaluation toolkit for LLM-as-a-judge systems.
It probes whether judge decisions remain stable under irrelevant transformations such as answer order,
verbosity, style, and rubric paraphrasing, and whether reference answers improve or distort judgment.
The tool supports both single-example debugging and multi-case diagnostic audits.
```

Avoid overclaiming with wording like:

```text
This produces a definitive reliability certificate for any judge model.
```

Prefer:

```text
This produces a perturbation-based reliability profile.
```

or:

```text
This produces a diagnostic bias profile over the tested cases.
```

---

## 8. Suggested README Language

Recommended short description:

```text
A diagnostic toolkit for auditing LLM-as-a-judge reliability. It tests whether a judge is stable under answer order, verbosity, style, rubric paraphrasing, repeated calls, and reference-answer conditions. V2 supports both single-pair debugging and controlled multi-case diagnostic suites with expected winners.
```

Recommended limitation statement:

```text
This tool does not prove that a judge model is globally reliable. Results depend on the selected task, cases, prompts, model settings, and diagnostic suite. Single-pair mode should be interpreted as a local fragility probe, while diagnostic-suite mode provides a broader but still sample-dependent bias profile.
```

---

## 9. Suggested Future Improvements

V2 is much stronger than V1, but several future upgrades would make it even better.

### 9.1 Larger diagnostic suite

Add more cases per category:

```text
position
verbosity
style
rubric
reference
factual correctness
reasoning correctness
safety refusal quality
code review
summarization
RAG answer evaluation
```

### 9.2 Human-labeled or expert-labeled cases

For stronger credibility, include cases with known human/expert preference labels.

### 9.3 Statistical confidence intervals

Instead of only reporting point estimates, add uncertainty:

```text
accuracy: 78% ± 6%
position flip rate: 12% ± 4%
```

### 9.4 More robust perturbation validation

Use embedding similarity or NLI-style checks to validate that mutations preserve meaning.

### 9.5 Judge settings tracking

Store and report:

```text
temperature
top_p
seed if available
provider
model version/date
max tokens
```

This is important for reproducibility.

### 9.6 Dynamic OpenRouter model validation

Instead of relying only on a hardcoded model list, query OpenRouter's models endpoint or provide a free-text model ID input with validation.

### 9.7 Exportable reports

Add downloadable outputs:

```text
JSON report
Markdown report
HTML report
CSV model comparison
```

### 9.8 Leaderboard mode

Add a benchmark-style comparison table:

```text
Model | Accuracy | Position Bias | Style Bias | Verbosity Bias | Overall
```

This would make the project especially attractive for GitHub/Kaggle.

---

## 10. Final Assessment

V1 was already a strong engineering prototype. The most important improvement was not to rewrite everything, but to make the scientific framing more honest and add a controlled diagnostic mode.

V2 now has a clearer identity:

```text
V1: Single-example perturbation probe
V2: Single-example probe + controlled diagnostic audit suite
```

This is a much stronger capstone project because it demonstrates:

```text
agentic workflow design
LLM evaluation thinking
bias and robustness testing
structured outputs
multi-model comparison
scientific caution about claims
```

The next best step is to run the diagnostic suite against several OpenRouter models, save example reports, and include those results in the README or Kaggle writeup.

---

## Patch: post-review corrections

After reviewing the packaged V2 project, three implementation gaps were corrected:

1. **Diagnostic-suite data packaging**
   - Verified that `diagnostics/builtin_cases.jsonl` is included in the project and in the distributable ZIP.
   - Added a regression test that explicitly checks `DEFAULT_PATH.exists()` before loading cases.
   - Added a coverage test that verifies the built-in diagnostic library covers all six bias dimensions: position, verbosity, style, consistency, rubric, and reference.

2. **Expanded unit tests**
   - Added tests for deterministic bias-analyzer behavior.
   - Added tests for consistency scoring, including the single-pair baseline-as-run-1 case.
   - Added tests for reference-guided accuracy scoring.
   - Added tests for the weighted reliability score formula.
   - Added tests for metric-summary separation: accuracy, invariance, stability, and reference helpfulness.
   - Current local check: `10 passed` with only harmless Pytest collection warnings caused by Pydantic classes named `TestCase` / enum `TestType`.

3. **Style perturbation validation**
   - Fixed the style-bias perturbation validation so each style variant validates both mutated answers.
   - Example: `style_aplain_bpolished` now validates both A→plain and B→polished.
   - Example: `style_apolished_bplain` now validates both A→polished and B→plain.
   - Aggregated validation metrics now include `validated_rewrites=2`, plus per-answer expected-style metrics.

An additional small consistency fix was also applied: in single-pair mode, the baseline decision is now included as consistency run 1 when consistency variants are analyzed.
