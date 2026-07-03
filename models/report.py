from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, computed_field
from models.input import AuditInput, AuditMode, TestType
from models.test_case import JudgeDecision
from models.validation import MutationValidation


class TestResult(BaseModel):
    """Result for one audit dimension.

    V3 deliberately separates correctness from invariance/stability:
    - baseline_accuracy: correctness on the original unperturbed pair
    - robust_accuracy: correctness on perturbed variants
    - invariance: whether the perturbed verdict stayed the same as baseline
    - stability: repeated-call agreement
    - consistency_accuracy: repeated-call correctness
    - consistency_quality: stability × consistency_accuracy

    bias_score remains a 0..1 failure-style score used for LOW/MEDIUM/HIGH labels,
    but it is no longer the only signal in the report.
    """

    test_type: TestType
    decisions: list[JudgeDecision]
    bias_score: float = Field(..., ge=0.0, le=1.0)
    verdict: Literal["LOW", "MEDIUM", "HIGH"]
    evidence: list[str]

    n_cases: int = 0
    n_failures: int = 0
    metric_family: Literal["accuracy", "invariance", "stability", "influence", "mixed"] = "mixed"

    # Accuracy / invariance decomposition for perturbation tests
    baseline_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    robust_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    invariance: float | None = Field(default=None, ge=0.0, le=1.0)

    # Consistency-specific decomposition
    stability: float | None = Field(default=None, ge=0.0, le=1.0)
    consistency_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    consistency_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    consistency_profile: Literal[
        "stable_and_correct",
        "stable_but_inaccurate",
        "accurate_but_unstable",
        "unstable_and_inaccurate",
        "stability_only",
        "insufficient_data",
    ] | None = None

    # Reference-specific decomposition
    reference_accuracy_with: float | None = Field(default=None, ge=0.0, le=1.0)
    reference_accuracy_without: float | None = Field(default=None, ge=0.0, le=1.0)
    reference_delta: float | None = Field(default=None, ge=-1.0, le=1.0)

    # Composite dimension score used by the overall reliability index.
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)


class MetricSummary(BaseModel):
    baseline_accuracy: float | None = None
    robust_accuracy: float | None = None
    invariance: float | None = None
    stability: float | None = None
    consistency_accuracy: float | None = None
    consistency_quality: float | None = None
    consistency_profile: str | None = None
    reference_accuracy_with: float | None = None
    reference_accuracy_without: float | None = None
    reference_helpfulness: float | None = None
    bias_susceptibility: float | None = None

    reference_helpfulness_label: str | None = None


class AuditReport(BaseModel):
    judge_model: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    audit_mode: AuditMode
    reliability_score: float = Field(..., ge=0.0, le=100.0)
    confidence_level: Literal["low", "medium", "high"]
    score_interpretation: str
    metric_summary: MetricSummary
    test_results: dict[TestType, TestResult]
    warnings: list[str]
    recommendations: list[str]
    total_api_calls: int
    n_cases: int
    n_valid_mutations: int = 0
    n_invalid_mutations: int = 0
    mutation_validations: list[MutationValidation] = Field(default_factory=list)
    generated_variants: list[dict[str, str | bool | int | float | None]] = Field(default_factory=list)
    audit_input: AuditInput

    @computed_field
    @property
    def grade(self) -> Literal["A", "B", "C", "D", "F"]:
        if self.reliability_score >= 90:
            return "A"
        if self.reliability_score >= 75:
            return "B"
        if self.reliability_score >= 60:
            return "C"
        if self.reliability_score >= 45:
            return "D"
        return "F"
