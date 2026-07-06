"""Gradio UI for LLM Judge Reliability Auditor v4.
Provides a clean, premium, dark-mode user interface for auditing LLM-as-a-judge models.
"""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

# Load environmental variables
load_dotenv()

# Agents & Pipeline imports
from agents.bias_analyzer import analyze as analyze_bias
from agents.perturbation_generator import generate_test_cases
from agents.judge_runner import run_test_cases
from agents.report_writer import build_report
from models.input import AuditInput, AuditMode, TestType
from models.report import AuditReport

# HTML presentation helpers
from utils.html_renderer import (
    EMPTY_STATE_HTML,
    build_status_bar,
    build_results_html,
)

# ── Paths for assets ───────────────────────────────────────────────────────────
CURRENT_DIR = Path(__file__).parent
CSS_PATH = CURRENT_DIR / "static" / "style.css"
JS_PATH = CURRENT_DIR / "static" / "patch_checkboxes.js"

# Read frontend assets
with open(CSS_PATH, "r", encoding="utf-8") as f:
    CSS_CONTENT = f.read()

with open(JS_PATH, "r", encoding="utf-8") as f:
    JS_CONTENT = f.read()

# ── Configuration Constants ───────────────────────────────────────────────────
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
    "Style bias — plain correct vs polished wrong": {
        "question": "Why is sleep important for health?",
        "answer_a": "Sleep supports memory consolidation, immune function, hormone regulation, and cellular repair. Most adults need 7 to 9 hours. Chronic sleep deprivation increases risk of cardiovascular disease, diabetes, obesity, mood problems, and impaired attention.",
        "answer_b": "Sleep is a magnificent nightly voyage in which the soul becomes radiant and the body enters a mysterious state of cosmic renewal. Its value is beyond measurement, and anyone who sleeps deeply will be protected from nearly all modern diseases.",
        "reference_answer": "Sleep supports memory, immune function, metabolic regulation, tissue repair, and attention. Adults generally need 7 to 9 hours; chronic deprivation raises several health risks.",
        "expected_winner": "A",
        "rubric": "Choose the answer that is more factually accurate and useful. Do not reward poetic style unless it improves accuracy.",
    },
    "Position bias — efficient code": {
        "question": "Which duplicate-finding Python function is better for large lists?",
        "answer_a": "def dupes(xs):\n    return [x for x in xs if xs.count(x) > 1]",
        "answer_b": "def dupes(xs):\n    seen, out = set(), set()\n    for x in xs:\n        if x in seen:\n            out.add(x)\n        seen.add(x)\n    return list(out)",
        "reference_answer": "The set-based implementation is O(n) average time, while repeated count calls make the list-comprehension approach O(n^2).",
        "expected_winner": "B",
        "rubric": "Choose the function that is better for large inputs, considering correctness and time complexity.",
    },
    "Verbosity bias — concise correct vs padded wrong": {
        "question": "What is the difference between a list and a tuple in Python?",
        "answer_a": "Lists are mutable (you can add, remove, or change elements); tuples are immutable. Tuples are generally faster and use less memory. Use a tuple when the data should not change.",
        "answer_b": "In the Python programming language, which is widely used, there exist two commonly used data structures: the list and the tuple. Lists, created with square brackets, are mutable — elements can be added, removed, or changed. Tuples, created with parentheses, are immutable — once created, they cannot be changed. From a performance angle, tuples tend to be faster and use less memory than lists.",
        "expected_winner": "A",
        "rubric": "Choose the answer that is clearer and more useful. Do not reward length or verbosity.",
    },
    "Rubric sensitivity — security guidance": {
        "question": "Should websites always use HTTPS?",
        "answer_a": "Yes. HTTPS encrypts traffic with TLS, preventing eavesdropping and tampering. Browsers mark HTTP sites as 'Not Secure'. Free certificates via Let's Encrypt remove any cost barrier.",
        "answer_b": "HTTPS is a good idea for most sites, but may be unnecessary for simple static pages that don't transmit sensitive data or collect user input.",
        "expected_winner": "A",
        "rubric": "Choose the answer that provides more complete and accurate security guidance.",
    },
}

# ── Event Helper Functions ───────────────────────────────────────────────────
def load_preset(name: str) -> tuple[str, str, str, str, str, str]:
    """Retrieves field inputs associated with a saved test preset case."""
    preset_data = PRESETS.get(name)
    if not preset_data:
        return ("", "", "", "", "Choose the answer that is more helpful and factually correct.", "Not specified")
    return (
        preset_data.get("question", ""),
        preset_data.get("answer_a", ""),
        preset_data.get("answer_b", ""),
        preset_data.get("reference_answer", ""),
        preset_data.get("rubric", ""),
        preset_data.get("expected_winner", "Not specified"),
    )

# ── Pipeline Execution Orchestrator ──────────────────────────────────────────
def run_audit_pipeline(
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
    active_mode: str,
    progress=gr.Progress(),
):
    """Executes the reliability auditing pipeline for selected judge models.
    Yields (status_html, result_html, json_str) incrementally.
    """
    if not judge_models:
        error_msg = '<div class="empty-state"><div class="empty-title" style="color:#f87171">Please select at least one judge model.</div></div>'
        yield build_status_bar(""), error_msg, ""
        return

    # Determine execution mode and active configurations
    audit_mode = AuditMode.DIAGNOSTIC_SUITE if active_mode == "diagnostic" else AuditMode.SINGLE_PAIR

    if audit_mode == AuditMode.SINGLE_PAIR:
        missing = [label for label, val in [("Question", question), ("Answer A", answer_a), ("Answer B", answer_b)] if not val or not str(val).strip()]
        if missing:
            error_msg = f'<div class="empty-state"><div class="empty-title" style="color:#f87171">Please fill in the required fields: {", ".join(missing)}.</div></div>'
            yield build_status_bar(""), error_msg, ""
            return
    tests_to_run = [TestType(t) for t in selected_tests] if selected_tests else list(TestType)
    ew_winner = None if expected_winner in (None, "Not specified") else expected_winner

    num_models = len(judge_models)
    progress(0, desc="Starting...")
    yield build_status_bar(f"Starting audit for <strong>{num_models}</strong> model(s)..."), EMPTY_STATE_HTML, ""

    reports: dict[str, AuditReport] = {}
    errors: dict[str, str] = {}

    def construct_input(model_name: str) -> AuditInput:
        return AuditInput(
            audit_mode=audit_mode,
            question=question or None,
            answer_a=answer_a or None,
            answer_b=answer_b or None,
            reference_answer=reference_answer.strip() or None,
            expected_winner=ew_winner,
            rubric=rubric or None,
            judge_model=model_name,
            tests=tests_to_run,
            consistency_runs=int(consistency_runs),
            diagnostic_case_limit=int(diagnostic_case_limit),
            diagnostic_difficulty=diagnostic_difficulty,
        )

    if num_models == 1:
        # ── Single Model Audit: Detailed step-by-step progress feedback ──
        model = judge_models[0]
        try:
            progress(0.05, desc="Generating perturbation test cases...")
            yield build_status_bar(f"⚙️ Generating test cases for <strong>{model}</strong>..."), EMPTY_STATE_HTML, ""
            audit_input = construct_input(model)
            cases = generate_test_cases(audit_input)

            progress(0.25, desc=f"Querying judge ({len(cases)} test cases)...")
            yield build_status_bar(f"🤖 Querying judge model — <strong>{len(cases)}</strong> queries in progress..."), EMPTY_STATE_HTML, ""
            decisions = run_test_cases(cases, model)

            progress(0.70, desc="Analyzing statistical biases...")
            yield build_status_bar("📊 Analyzing bias dimensions..."), EMPTY_STATE_HTML, ""
            results = analyze_bias(decisions, cases)

            progress(0.90, desc="Synthesizing final report...")
            yield build_status_bar("📝 Synthesizing final reliability report..."), EMPTY_STATE_HTML, ""
            report = build_report(audit_input, results, cases, total_api_calls=len(decisions))
            reports[model] = report

        except Exception as e:
            errors[model] = str(e)
    else:
        # ── Multi-Model Audit: Parallel execution via ThreadPoolExecutor ──
        yield build_status_bar(f"🚀 Running <strong>{num_models}</strong> models in parallel..."), EMPTY_STATE_HTML, ""
        progress(0.05, desc=f"Spawning {num_models} parallel workers...")

        completed_count = 0

        def audit_single_worker(model_name: str) -> AuditReport:
            inp = construct_input(model_name)
            cases = generate_test_cases(inp)
            dec = run_test_cases(cases, model_name)
            res = analyze_bias(dec, cases)
            return build_report(inp, res, cases, total_api_calls=len(dec))

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, num_models)) as executor:
            futures = {executor.submit(audit_single_worker, m): m for m in judge_models}
            for future in concurrent.futures.as_completed(futures):
                m = futures[future]
                completed_count += 1
                try:
                    reports[m] = future.result()
                except Exception as e:
                    errors[m] = str(e)
                
                # Report incremental results as they complete
                progress(0.1 + (completed_count / num_models) * 0.85, desc=f"{completed_count}/{num_models} complete")
                yield (
                    build_status_bar(f"✅ <strong>{completed_count}/{num_models}</strong> models finished — <em>{m}</em> complete"),
                    build_results_html(reports, errors),
                    "",
                )

    progress(1.0, desc="Audit Finished")
    serialized_reports = json.dumps({m: r.model_dump(mode="json") for m, r in reports.items()}, indent=2)
    yield "", build_results_html(reports, errors), serialized_reports

# ── Gradio Blocks Layout ───────────────────────────────────────────────────────
with gr.Blocks(js=JS_CONTENT) as demo:

    # ── Header Banner ──
    gr.HTML("""
    <div class="hero-banner">
      <div class="hero-badge">v4 · Kaggle AI Agents Capstone</div>
      <h1 class="hero-title">
        🔬 LLM Judge Reliability Auditor
        <i class="hero-info" data-tip="A multi-agent diagnostic toolkit that tests whether an LLM-as-a-judge is trustworthy — measuring position bias, verbosity bias, style bias, consistency, rubric sensitivity, and reference distortion before you use it in an evaluation pipeline.">i</i>
      </h1>
    </div>
    """)

    active_mode_state = gr.State(value="single")

    # ── Columns Layout ──
    with gr.Row(equal_height=False):

        # Left side: Input Forms
        with gr.Column(scale=7):
            with gr.Tabs() as mode_tabs:

                # Single-pair test case generator tab
                with gr.TabItem("🎯  Single-pair Probe"):
                    gr.HTML(
                        '<div class="mode-callout">Provide a reference prompt and two potential answers. '
                        'The auditor mutates the pair (order swap, verbosity padding, style adjustments) '
                        'and verifies if the judge verdict shifts. <strong>Note:</strong> single pairs offer qualitative insights. '
                        'Use presets below to start quickly.</div>'
                    )

                    preset_select = gr.Dropdown(
                        choices=list(PRESETS.keys()),
                        label="Load a preset example",
                        value=None,
                        interactive=True,
                    )

                    prompt_input = gr.Textbox(
                        label="Question or task prompt",
                        lines=2,
                        placeholder="e.g. Write a Python function to sort a list of dictionary values.",
                    )

                    with gr.Row():
                        answer_a_input = gr.Textbox(
                            label="Answer A",
                            lines=7,
                            placeholder="Candidate answer A...",
                        )
                        answer_b_input = gr.Textbox(
                            label="Answer B",
                            lines=7,
                            placeholder="Candidate answer B...",
                        )

                    with gr.Row():
                        reference_input = gr.Textbox(
                            label="Reference answer (optional — enables reference-distortion validation)",
                            lines=3,
                            placeholder="Ground truth reference answer...",
                        )
                        rubric_input = gr.Textbox(
                            label="Evaluation rubric",
                            lines=3,
                            value="Choose the answer that is more helpful and factually correct.",
                        )

                    expected_select = gr.Dropdown(
                        choices=["Not specified", "A", "B", "Tie"],
                        value="Not specified",
                        label="Expected winner (optional — enables model accuracy calculations)",
                        interactive=True,
                    )

                # Diagnostic Suite Tab
                with gr.TabItem("🔬  Diagnostic Suite"):
                    gr.HTML(
                        '<div class="mode-callout">Run the built-in library of <strong>close-call scenarios</strong> '
                        'specifically structured to isolate and trigger position, verbosity, and style biases. '
                        'Yields a high-confidence bias profile. Filter scenarios below.</div>'
                    )

                    with gr.Row():
                        limit_slider = gr.Slider(
                            minimum=1,
                            maximum=30,
                            value=18,
                            step=1,
                            label="Scenario case limit (18 loads the full core library)",
                        )
                        difficulty_select = gr.Dropdown(
                            choices=["all", "easy", "medium", "hard"],
                            value="all",
                            label="Case difficulty filter",
                        )

                    gr.HTML("""
                    <div style="margin-top:20px;padding:16px;background:#110e28;border:1px solid #231f47;border-radius:10px;">
                      <div style="font-size:11px;font-weight:700;color:#5c527a;text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px;">Diagnostic Library Modules</div>
                      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
                        <div>Order Swap</strong><br>Detects first-position selection favoritism.</div>
                        <div>Padding</strong><br>Tests resilience to irrelevant length changes.</div>
                        <div>Refinement</strong><br>Measures vulnerability to aesthetic or polished phrasing.</div>
                        <div>Stability</strong><br>Repeated runs to profile consistency under high temperature.</div>
                        <div>Semantic Shift</strong><br>Verifies impact of minor rubric rewording.</div>
                        <div>Ref Anchor</strong><br>Quantifies bias introduced by matching references.</div>
                      </div>
                    </div>
                    """)

        # Right side: Settings & Action Panel
        with gr.Column(scale=3, min_width=260):
            gr.HTML('<div class="cfg-section">Target Judge Models</div>', elem_classes=["sh-block"])
            selected_models = gr.Dropdown(
                choices=SUPPORTED_MODELS,
                value=["google/gemini-2.5-flash"],
                multiselect=True,
                label="Select judge models to audit",
                interactive=True,
            )

            gr.HTML('<div class="cfg-section" style="margin-top:18px;">Active Bias Probes</div>', elem_classes=["sh-block"])
            active_probes = gr.CheckboxGroup(
                choices=[t.value for t in TestType],
                value=[t.value for t in TestType],
                label="",
                show_label=False,
                interactive=True,
            )

            runs_slider = gr.Slider(
                minimum=2,
                maximum=10,
                value=5,
                step=1,
                label="Consistency iterations per case",
            )

            gr.HTML('<div style="margin-top:20px;"></div>')

            with gr.Row():
                submit_button = gr.Button("▶  Run Audit", variant="primary", scale=3)
                stop_button = gr.Button("✕", variant="secondary", scale=1)

            gr.HTML("""
            <div style="margin-top:12px;font-size:11px;color:#5c527a;line-height:1.6;text-align:center;">
              Multiple judges run concurrently.<br>
              Incremental results stream in live.
            </div>
            """)

    # ── Live Status & Report Output Cards ──
    pipeline_status = gr.HTML(value="")
    pipeline_results = gr.HTML(value=EMPTY_STATE_HTML)

    with gr.Accordion("📋  Raw JSON Audit Report", open=False):
        raw_report_json = gr.Code(language="json", label="")

    # ── Event Wiring ──

    # Tab select updates state mode
    mode_tabs.select(
        fn=lambda evt: "diagnostic" if getattr(evt, "index", 0) == 1 else "single",
        outputs=[active_mode_state],
    )

    # Preset selection populates prompt and answer values
    preset_select.change(
        fn=load_preset,
        inputs=[preset_select],
        outputs=[prompt_input, answer_a_input, answer_b_input, reference_input, rubric_input, expected_select],
    )

    # Trigger audit execution pipeline
    execution_event = submit_button.click(
        fn=run_audit_pipeline,
        inputs=[
            prompt_input,
            answer_a_input,
            answer_b_input,
            reference_input,
            expected_select,
            rubric_input,
            selected_models,
            active_probes,
            runs_slider,
            limit_slider,
            difficulty_select,
            active_mode_state,
        ],
        outputs=[pipeline_status, pipeline_results, raw_report_json],
    )

    # Stop button event cancellation
    stop_button.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[execution_event],
    )

# ── Application Launch ──
if __name__ == "__main__":
    demo.launch(css=CSS_CONTENT, theme=gr.themes.Base())
