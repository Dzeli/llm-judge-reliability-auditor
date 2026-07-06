Audit two or more judge models and produce a side-by-side comparison.

Models to compare: $ARGUMENTS (comma-separated, e.g. "google/gemini-2.5-flash, mistralai/mistral-large")

If $ARGUMENTS is empty, ask the user which models to compare before proceeding.

## Steps

1. Parse $ARGUMENTS as a comma-separated list of model IDs
2. Call `audit_judge_diagnostic_suite` for each model (one at a time)
   - `case_limit`: 18, `difficulty`: all, `consistency_runs`: 3
3. Produce a comparison table:

### Judge model comparison

| Metric | {model_1} | {model_2} | ... |
|---|---|---|---|
| Grade | | | |
| Reliability score | | | |
| Style verdict | | | |
| Verbosity verdict | | | |
| Position verdict | | | |
| Rubric verdict | | | |
| Reference quality | | | |
| Baseline accuracy | | | |
| Robust accuracy | | | |
| Invariance | | | |

### Recommendation
State which model you would recommend as a judge and why, based on:
- Overall reliability score
- Which biases matter most for pairwise evaluation use cases
- Confidence level of each result

### Common weaknesses
List any bias types where ALL tested models show MEDIUM or HIGH — these are systemic issues, not model-specific.
