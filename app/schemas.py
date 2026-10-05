"""Request and response schemas for the surface-vs-suppressed model API."""

from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field


class FindingFeatures(BaseModel):
    """The 43-feature request contract; absent artifact-derived features are imputed."""

    model_config = ConfigDict(extra="forbid")

    message: str
    rule_id: str
    rule_family: str
    severity_text: str
    language: str
    path_extension: str

    severity: int
    is_error: int
    is_warning: int
    message_length: int
    message_word_count: int
    start_line: int
    start_column: int
    end_line: int
    end_column: int
    finding_span_lines: int
    finding_span_columns: int
    has_fix: int
    fix_text_length: int
    fix_range_length: int
    has_suggestions: int
    suggestion_count: int
    changed_line_start: int
    changed_line_end: int
    changed_line_count: int
    finding_overlaps_change: int
    finding_change_distance: int
    file_size_lines: int
    pr_change_code_lines: int
    base_file_finding_count: int
    base_rule_finding_count: int
    base_file_has_findings: int
    base_rule_exists_in_file: int
    prior_findings_in_file: int
    pr_total_findings_in_file: int
    same_rule_findings_in_file: int
    same_rule_findings_in_repo: int
    path_depth: int
    path_has_test: int
    path_has_src: int
    path_has_config: int
    path_has_generated: int

    finding_start_line_ratio: float


class PredictionResult(BaseModel):
    probabilities: Dict[str, float]
    ensemble_surface_probability: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(..., ge=0.0, le=1.0)
    decision: Literal["surface", "suppress"]


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
    negative_class: str
    threshold: float
    feature_count: int
    input_feature_count: int
    model_family: str
    component_models: List[str]
    includes_message_tfidf: bool
