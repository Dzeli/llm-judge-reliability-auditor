"""Gradio UI for LLM Judge Reliability Auditor v4.
Run: python app.py
"""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from agents.orchestrator import run_audit_pipeline
from models.input import AuditInput, AuditMode, TestType

SUPPORTED_MODELS = [
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-pro",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-opus-4.1",
    "deepseek/deepseek-chat",
    "qwen/qwen3-30b-a3b",
    "qwen/qwen3-235b-a22b-thinking-2507",
    "mistralai/mistral-small-3.1-24b-instruct",
    "mistralai/mistral-large",
    "google/gemma-3-27b-it",
]

PRESETS = {
    "Style bias: plain correct vs polished wrong": {
        "question": "Why is sleep important for health?",
        "answer_a": "Sleep supports memory consolidation, immune function, hormone regulation, and cellular repair. Most adults need 7 to 9 hours. Chronic sleep deprivation increases risk of cardiovascular disease, diabetes, obesity, mood problems, and impaired attention.",
        "answer_b": "Sleep is a magnificent nightly voyage in which the soul becomes radiant and the body enters a mysterious state of cosmic renewal. Its value is beyond measurement, and anyone who sleeps deeply will be protected from nearly all modern diseases.",
        "reference_answer": "Sleep supports memory, immune function, metabolic regulation, tissue repair, and attention. Adults generally need 7 to 9 hours; chronic deprivation raises several health risks.",
        "expected_winner": "A",
        "rubric": "Choose the answer that is more factually accurate and useful. Do not reward poetic style unless it improves accuracy.",
    },
    "Position bias: efficient code": {
        "question": "Which duplicate-finding Python function is better for large lists?",
        "answer_a": "def dupes(xs):\n    return [x for x in xs if xs.count(x) > 1]",
        "answer_b": "def dupes(xs):\n    seen, out = set(), set()\n    for x in xs:\n        if x in seen:\n            out.add(x)\n        seen.add(x)\n    return list(out)",
        "reference_answer": "The set-based implementation is O(n) average time, while repeated count calls make the list-comprehension approach O(n^2).",
        "expected_winner": "B",
        "rubric": "Choose the function that is better for large inputs, considering correctness and time complexity.",
    },
}

CSS = """
.gradio-container { max-width: 1350px !important; margin: auto !important; }
.hero {background: linear-gradient(135deg,#111827,#1d4ed8); color:white; padding:24px 28px; border-radius:14px; margin-bottom:14px;}
.hero h1 {margin:0; font-size:28px;} .hero p {opacity:.85; margin:6px 0 0 0;}
.small-note {font-size: 12px; color:#666;}
"""

SINGLE_PAIR_INFO = (
    "> 📌 **Single-pair probe** — You supply a question and two answers. "
    "The auditor mutates your pair (swaps positions, rewrites style, etc.) and checks whether the judge changes its verdict. "
    "Good for spot-checking a specific case. Confidence is always **low** — one pair is one data point."
)

DIAGNOSTIC_INFO = (
    "> 🔬 **Diagnostic suite** — No custom input needed. "
    "The auditor runs its built-in library of close-call cases designed to expose bias across all 6 test types. "
    "Gives a generalizable bias profile with **medium** or **high** confidence. "
    "The question/answer fields below are ignored."
)


def _on_mode_change(mode: str):
    is_single = mode == "Single-pair probe"
    return (
        gr.update(value=SINGLE_PAIR_INFO if is_single else DIAGNOSTIC_INFO),
        gr.update(visible=is_single),  # preset dropdown
        gr.update(visible=is_single),  # left input column
        gr.update(visible=not is_single),  # diagnostic-only controls
    )


def _load_preset(name: str):
    p = PRESETS.get(name)
    if not p:
        return "", "", "", "", "", None
    return p["question"], p["answer_a"], p["answer_b"], p["reference_answer"], p["rubric"], p["expected_winner"]


def _run(
    audit_mode_label: str,
    question: str,
    answer_a: str,
    answer_b: str,
    reference_answer: str,
    expected_winner: str | None,
    rubric: str,
    judge_models: list[str],
    selected_tests: list[str],
    consistency_runs: int,
    diagnostic_case_limit: int,
    diagnostic_difficulty: str,
):
    if not judge_models:
        return "Please select at least one judge model.", ""
    audit_mode = AuditMode.DIAGNOSTIC_SUITE if audit_mode_label.startswith("Diagnostic") else AuditMode.SINGLE_PAIR
    tests = [TestType(t) for t in selected_tests] if selected_tests else list(TestType)

    def run_one(model: str):
        audit_input = AuditInput(
            audit_mode=audit_mode,
            question=question or None,
            answer_a=answer_a or None,
            answer_b=answer_b or None,
            reference_answer=reference_answer.strip() or None,
            expected_winner=None if expected_winner in {None, "unknown"} else expected_winner,
            rubric=rubric or None,
            judge_model=model,
            tests=tests,
            consistency_runs=int(consistency_runs),
            diagnostic_case_limit=int(diagnostic_case_limit),
            diagnostic_difficulty=diagnostic_difficulty,
        )
        return run_audit_pipeline(audit_input)

    reports, errors = {}, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(judge_models))) as pool:
        futures = {pool.submit(run_one, model): model for model in judge_models}
        for future in concurrent.futures.as_completed(futures):
            model = futures[future]
            try:
                reports[model] = future.result()
            except Exception as exc:
                errors[model] = str(exc)

    return _render_summary(reports, errors), json.dumps({m: r.model_dump(mode="json") for m, r in reports.items()}, indent=2)


def _render_summary(reports: dict, errors: dict) -> str:
    if not reports and not errors:
        return "No results."

    rows = []
    for model, report in reports.items():
        ms = report.metric_summary
        rows.append(
            f"| {model} | {report.grade} | {report.reliability_score:.1f} | {report.confidence_level} | "
            f"{_fmt_pct(ms.baseline_accuracy)} | {_fmt_pct(ms.robust_accuracy)} | {_fmt_pct(ms.invariance)} | "
            f"{_fmt_pct(ms.stability)} | {_fmt_pct(ms.consistency_accuracy)} | {_fmt_pct(ms.consistency_quality)} | "
            f"{(ms.consistency_profile or '—').replace('_', ' ')} | {report.n_cases} | {report.total_api_calls} |"
        )
    table = (
        "| Model | Grade | Score | Confidence | Baseline acc | Robust acc | Invariance | Consistency quality | Cases | Judge calls |\n"
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|\n" + "\n".join(rows)
    )

    details = []
    for model, report in reports.items():
        ms = report.metric_summary
        details.append(f"\n### {model}\n")
        details.append(f"**Interpretation:** {report.score_interpretation}\n")
        details.append(
            "**Metric summary**\n"
            f"- Baseline accuracy: {_fmt_pct(ms.baseline_accuracy)}\n"
            f"- Robust accuracy under perturbation: {_fmt_pct(ms.robust_accuracy)}\n"
            f"- Invariance: {_fmt_pct(ms.invariance)}\n"
            f"- Stability: {_fmt_pct(ms.stability)}\n"
            f"- Consistency accuracy: {_fmt_pct(ms.consistency_accuracy)}\n"
            f"- Consistency quality = stability × accuracy: {_fmt_pct(ms.consistency_quality)}\n"
            f"- Reference accuracy with / without: {_fmt_pct(ms.reference_accuracy_with)} / {_fmt_pct(ms.reference_accuracy_without)}\n"
            f"- Reference helpfulness delta: {_fmt_delta(ms.reference_helpfulness)}\n"
        )
        if report.warnings:
            details.append("**Warnings**\n" + "\n".join(f"- {w}" for w in report.warnings) + "\n")
        for test_type, result in report.test_results.items():
            evidence = "\n".join(f"  - {e}" for e in result.evidence[:4])
            metric_line = _result_metric_line(result)
            details.append(
                f"**{test_type.value}** — {result.verdict}, bias={result.bias_score:.2f}, cases={result.n_cases}\n"
                f"{metric_line}\n{evidence}\n"
            )
        invalid = [v for v in report.mutation_validations if not v.is_valid]
        if invalid:
            details.append(
                "**Mutation validation warnings**\n" +
                "\n".join(f"- {v.variant_id}: {'; '.join(v.warnings)}" for v in invalid[:8])
            )
    for model, error in errors.items():
        details.append(f"\n### {model}\n⚠️ {error}\n")
    return table + "\n" + "\n".join(details)


def _fmt_pct(value):
    if value is None:
        return "—"
    return f"{value * 100:.0f}%"


def _fmt_delta(value):
    if value is None:
        return "—"
    return f"{value * 100:+.0f} pp"


def _result_metric_line(result):
    parts = []
    if result.baseline_accuracy is not None:
        parts.append(f"baseline acc={_fmt_pct(result.baseline_accuracy)}")
    if result.robust_accuracy is not None:
        parts.append(f"robust acc={_fmt_pct(result.robust_accuracy)}")
    if result.invariance is not None:
        parts.append(f"invariance={_fmt_pct(result.invariance)}")
    if result.stability is not None:
        parts.append(f"stability={_fmt_pct(result.stability)}")
    if result.consistency_accuracy is not None:
        parts.append(f"consistency acc={_fmt_pct(result.consistency_accuracy)}")
    if result.consistency_quality is not None:
        parts.append(f"quality={_fmt_pct(result.consistency_quality)}")
    if result.consistency_profile is not None:
        parts.append(f"profile={result.consistency_profile.replace('_', ' ')}")
    if result.reference_accuracy_with is not None:
        parts.append(f"ref with={_fmt_pct(result.reference_accuracy_with)}")
    if result.reference_accuracy_without is not None:
        parts.append(f"ref without={_fmt_pct(result.reference_accuracy_without)}")
    if result.reference_delta is not None:
        parts.append(f"ref delta={_fmt_delta(result.reference_delta)}")
    if result.quality_score is not None:
        parts.append(f"composite quality={_fmt_pct(result.quality_score)}")
    return "_" + "; ".join(parts) + "_" if parts else ""


with gr.Blocks(title="LLM Judge Reliability Auditor v4") as demo:
    gr.HTML("""
    <div class='hero'>
      <h1>🔬 LLM Judge Reliability Auditor v4</h1>
      <p>Accuracy-aware perturbation probes + controlled diagnostic-suite audits for LLM-as-judge systems.</p>
    </div>
    """)

    with gr.Row():
        audit_mode = gr.Radio(
            ["Single-pair probe", "Diagnostic suite"],
            value="Single-pair probe",
            label="Audit mode",
        )

    mode_info = gr.Markdown(value=SINGLE_PAIR_INFO)
    preset = gr.Dropdown(list(PRESETS.keys()), label="Preset (single-pair only)", value=None, visible=True)

    with gr.Row():
        with gr.Column(scale=3, visible=True) as input_col:
            question = gr.Textbox(label="Question", lines=2)
            with gr.Row():
                answer_a = gr.Textbox(label="Answer A", lines=7)
                answer_b = gr.Textbox(label="Answer B", lines=7)
            with gr.Row():
                reference = gr.Textbox(label="Reference answer (optional)", lines=3)
                rubric = gr.Textbox(label="Rubric", lines=3, value="Choose the answer that is more helpful and factually correct.")
            expected = gr.Dropdown(["unknown", "A", "B", "tie"], value="unknown", label="Expected winner (optional, but recommended)")
        with gr.Column(scale=1):
            models = gr.Dropdown(SUPPORTED_MODELS, value=["google/gemini-2.5-flash"], multiselect=True, label="Judge model(s)")
            tests = gr.CheckboxGroup([t.value for t in TestType], value=[t.value for t in TestType], label="Tests")
            consistency_runs = gr.Slider(2, 10, value=5, step=1, label="Consistency runs")
            with gr.Group(visible=False) as diagnostic_controls:
                diagnostic_limit = gr.Slider(1, 30, value=18, step=1, label="Diagnostic case limit")
                diagnostic_difficulty = gr.Dropdown(["all", "easy", "medium", "hard"], value="all", label="Diagnostic difficulty")
            run_btn = gr.Button("Run audit", variant="primary")

    summary = gr.Markdown(label="Summary")
    raw_json = gr.Code(label="Raw JSON report", language="json")

    audit_mode.change(_on_mode_change, inputs=audit_mode, outputs=[mode_info, preset, input_col, diagnostic_controls])
    preset.change(_load_preset, inputs=preset, outputs=[question, answer_a, answer_b, reference, rubric, expected])
    run_btn.click(
        _run,
        inputs=[audit_mode, question, answer_a, answer_b, reference, expected, rubric, models, tests, consistency_runs, diagnostic_limit, diagnostic_difficulty],
        outputs=[summary, raw_json],
    )

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft())
