from typing import Dict, List
from pydantic import BaseModel, Field

class FindingFeatures(BaseModel):
    """Exact feature contract matching the PRism model."""
    rule_id: str
    rule_family: str
    severity: int
    is_error: int
    message_length: int
    has_fix: int
    fix_text_length: int
    fix_range_length: int
    has_suggestions: int
    suggestion_count: int
    changed_line_count: int
    file_size_lines: int
    start_line: int
    finding_start_line_ratio: float
    finding_span_lines: int
    finding_span_columns: int
    pr_change_code_lines: int
    pr_total_findings_in_file: int
    same_rule_findings_in_file: int
    same_rule_findings_in_repo: int
    finding_overlaps_change: int
    finding_change_distance: int


class PredictionResult(BaseModel):
    model_version: str
    positive_class: str
    risk_score: float
    decision: str
    threshold: float
    component_scores: Dict[str, float]


class BatchPredictionRequest(BaseModel):
    items: List[FindingFeatures] = Field(..., max_length=500)


class BatchPredictionResponse(BaseModel):
    model_version: str
    threshold: float
    predictions: List[PredictionResult]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_version: str
    positive_class: str
    threshold: float
    feature_count: int
    model_family: str
