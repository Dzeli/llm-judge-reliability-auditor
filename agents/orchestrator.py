"""V3 orchestrator.

The direct Python pipeline is the primary UI path. ADK tools expose the same stages for the
capstone requirement and for interactive agent debugging.
"""
from __future__ import annotations

import json
from typing import Any

try:
    from google.adk.agents import Agent
    from google.adk.tools import FunctionTool
except Exception:  # ADK is optional for local tests/imports.
    Agent = None  # type: ignore
    FunctionTool = None  # type: ignore

from agents.bias_analyzer import analyze
from agents.judge_runner import run_test_cases
from agents.perturbation_generator import generate_test_cases
from agents.report_writer import build_report
from models.input import AuditInput
from models.report import AuditReport
from models.test_case import JudgeDecision, TestCase


def run_audit_pipeline(audit_input: AuditInput) -> AuditReport:
    test_cases = generate_test_cases(audit_input)
    decisions = run_test_cases(test_cases, audit_input.judge_model)
    test_results = analyze(decisions, test_cases)
    return build_report(audit_input, test_results, test_cases, total_api_calls=len(decisions))


def tool_generate_perturbations(audit_input_json: str) -> str:
    audit_input = AuditInput.model_validate_json(audit_input_json)
    cases = generate_test_cases(audit_input)
    return json.dumps([c.model_dump(mode="json") for c in cases], indent=2)


def tool_run_judge(test_cases_json: str, judge_model: str) -> str:
    cases = [TestCase.model_validate(c) for c in json.loads(test_cases_json)]
    decisions = run_test_cases(cases, judge_model)
    return json.dumps([d.model_dump(mode="json") for d in decisions], indent=2)


def tool_analyze_bias(decisions_json: str, test_cases_json: str) -> str:
    decisions = [JudgeDecision.model_validate(d) for d in json.loads(decisions_json)]
    cases = [TestCase.model_validate(c) for c in json.loads(test_cases_json)]
    results = analyze(decisions, cases)
    return json.dumps({k.value: v.model_dump(mode="json") for k, v in results.items()}, indent=2)


def tool_build_report(audit_input_json: str, test_results_json: str, test_cases_json: str, total_api_calls: int) -> str:
    from models.input import TestType
    from models.report import TestResult
    audit_input = AuditInput.model_validate_json(audit_input_json)
    raw_results: dict[str, Any] = json.loads(test_results_json)
    results = {TestType(k): TestResult.model_validate(v) for k, v in raw_results.items()}
    cases = [TestCase.model_validate(c) for c in json.loads(test_cases_json)]
    return build_report(audit_input, results, cases, total_api_calls).model_dump_json(indent=2)


if Agent is not None:
    root_agent = Agent(
        name="llm_judge_auditor_v3",
        model="gemini-2.5-flash",
        description="Audits LLM-as-judge reliability using single-pair probes and controlled diagnostic cases.",
        instruction="""
You are the LLM Judge Reliability Auditor v3 orchestrator.
Follow the pipeline: generate perturbations, run judge, analyze bias, build report.
Explain that single-pair mode is a local probe and diagnostic-suite mode gives stronger but still empirical estimates.
""",
        tools=[
            FunctionTool(tool_generate_perturbations),
            FunctionTool(tool_run_judge),
            FunctionTool(tool_analyze_bias),
            FunctionTool(tool_build_report),
        ],
    )
else:
    root_agent = None
