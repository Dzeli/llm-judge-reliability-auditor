from typing import Literal, Optional, Tuple
from pydantic import BaseModel, Field
from models.input import TestType
from models.validation import MutationValidation


class TestCase(BaseModel):
    test_type: TestType
    variant_id: str = Field(..., description="Unique ID, e.g. diag_001::position_swapped")
    role: Literal["baseline", "variant"] = Field(
        default="variant",
        description="Explicit case role. Baselines are routed into analyses without relying on variant_id naming conventions.",
    )
    case_id: str = Field(default="user_case")
    prompt: str = Field(..., description="Exact prompt sent to the judge model")
    answer_order: Tuple[str, str] = Field(
        default=("A", "B"),
        description="Original answer labels shown as prompt A and prompt B",
    )
    mutation_description: str
    expected_winner: Optional[Literal["A", "B", "tie"]] = None
    should_preserve_winner: bool = Field(
        default=True,
        description="True when the mutation should not change the correct verdict",
    )
    metric_family: Literal["accuracy", "invariance", "stability", "influence"] = "invariance"
    validation: Optional[MutationValidation] = None
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class JudgeDecision(BaseModel):
    variant_id: str
    winner: Literal["A", "B", "tie"]
    reasoning: str
    raw_response: str
    latency_ms: float
    tokens_used: Optional[int] = None
    case_id: str = "user_case"
