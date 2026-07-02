from typing import Literal
from pydantic import BaseModel, Field
from models.input import TestType


class DiagnosticCase(BaseModel):
    id: str
    target_bias: TestType
    question: str
    answer_a: str
    answer_b: str
    expected_winner: Literal["A", "B", "tie"]
    rubric: str
    reference_answer: str | None = None
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    rationale: str = Field(..., description="Why the expected winner is known")
