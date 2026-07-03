from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


class AuditMode(str, Enum):
    SINGLE_PAIR = "single_pair"
    DIAGNOSTIC_SUITE = "diagnostic_suite"


class TestType(str, Enum):
    POSITION = "position"
    VERBOSITY = "verbosity"
    STYLE = "style"
    CONSISTENCY = "consistency"
    RUBRIC = "rubric"
    REFERENCE = "reference"


class AuditInput(BaseModel):
    """User-facing configuration for an audit run.

    V4 supports two modes:
    - single_pair: perturb one user-provided A/B example. This is a local probe, not a global certificate.
    - diagnostic_suite: run controlled built-in cases with expected winners for a stronger bias profile.
    """

    audit_mode: AuditMode = Field(default=AuditMode.SINGLE_PAIR)

    question: str | None = Field(None, description="The question/task being evaluated in single-pair mode")
    answer_a: str | None = Field(None, description="First candidate answer in single-pair mode")
    answer_b: str | None = Field(None, description="Second candidate answer in single-pair mode")
    reference_answer: Optional[str] = Field(
        None,
        description="Ground-truth answer, optional in single-pair mode; included in relevant diagnostic cases",
    )
    expected_winner: Optional[Literal["A", "B", "tie"]] = Field(
        None,
        description="Optional gold winner for the supplied pair. Enables accuracy-aware scoring.",
    )
    judge_model: str = Field(..., description="OpenRouter model ID to use as judge")
    rubric: str | None = Field(None, description="Evaluation criteria for single-pair mode")
    tests: list[TestType] = Field(default_factory=lambda: list(TestType))
    consistency_runs: int = Field(default=5, ge=2, le=10)

    diagnostic_case_limit: int = Field(
        default=18,
        ge=1,
        le=100,
        description="Maximum number of built-in diagnostic cases to run in diagnostic-suite mode",
    )
    diagnostic_difficulty: Literal["all", "easy", "medium", "hard"] = "all"

    @model_validator(mode="after")
    def validate_mode_inputs(self):
        if self.audit_mode == AuditMode.SINGLE_PAIR:
            missing = [
                name for name, value in {
                    "question": self.question,
                    "answer_a": self.answer_a,
                    "answer_b": self.answer_b,
                    "rubric": self.rubric,
                }.items()
                if not value or not str(value).strip()
            ]
            if missing:
                raise ValueError(f"single_pair mode requires: {', '.join(missing)}")
        return self

    def active_tests(self) -> list[TestType]:
        runnable = list(dict.fromkeys(self.tests))
        if self.audit_mode == AuditMode.SINGLE_PAIR and not self.reference_answer:
            runnable = [t for t in runnable if t != TestType.REFERENCE]
        return runnable
