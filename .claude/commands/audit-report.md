Run a diagnostic suite audit on the judge model: $ARGUMENTS

If $ARGUMENTS is empty, ask the user which model to audit before proceeding.

## Steps

1. Call the `audit_judge_diagnostic_suite` MCP tool with:
   - `judge_model`: $ARGUMENTS
   - `case_limit`: 18
   - `difficulty`: all
   - `consistency_runs`: 3

2. Parse the result and report in this exact structure:

### Model: {judge_model}
**Grade:** {grade} | **Score:** {reliability_score}/100 | **Confidence:** {confidence_level}

#### Bias profile
| Test | Verdict | Bias score | Quality |
|---|---|---|---|
| (one row per test type) |

#### Strengths
- List test types with verdict LOW

#### Weaknesses  
- List test types with verdict MEDIUM or HIGH
- Include the specific metric that failed (invariance, robust_accuracy, etc.)

#### Warnings
- List all warnings from the report

#### Bottom line
One sentence: is this model trustworthy as a judge, and what is its main failure mode?

## Notes
- Always mention confidence level and what it means
- If verbosity verdict is HIGH or MEDIUM, always recommend controlling answer length
- If position verdict is not LOW, always recommend running pairwise evaluations in both directions
