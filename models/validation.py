from typing import Literal
from pydantic import BaseModel, Field


class MutationValidation(BaseModel):
    variant_id: str
    is_valid: bool
    validation_type: Literal["heuristic", "llm", "not_applicable"] = "heuristic"
    metrics: dict[str, float | int | str | bool] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
