/**
 * TypeScript client for the PRism SURFACE/SUPPRESS model service.
 * Pass the 43 request features; the service median-imputes 12 artifact features
 * that are not part of this request contract.
 */

const ML_SERVICE_URL = (process.env.ML_SERVICE_URL || 'http://localhost:8000').replace(/\/+$/, '');

export interface FindingFeatures {
    message: string;
    rule_id: string;
    rule_family: string;
    severity_text: string;
    language: string;
    path_extension: string;
    severity: number;
    is_error: number;
    is_warning: number;
    message_length: number;
    message_word_count: number;
    start_line: number;
    start_column: number;
    end_line: number;
    end_column: number;
    finding_span_lines: number;
    finding_span_columns: number;
    has_fix: number;
    fix_text_length: number;
    fix_range_length: number;
    has_suggestions: number;
    suggestion_count: number;
    changed_line_start: number;
    changed_line_end: number;
    changed_line_count: number;
    finding_overlaps_change: number;
    finding_change_distance: number;
    file_size_lines: number;
    finding_start_line_ratio: number;
    pr_change_code_lines: number;
    base_file_finding_count: number;
    base_rule_finding_count: number;
    base_file_has_findings: number;
    base_rule_exists_in_file: number;
    prior_findings_in_file: number;
    pr_total_findings_in_file: number;
    same_rule_findings_in_file: number;
    same_rule_findings_in_repo: number;
    path_depth: number;
    path_has_test: number;
    path_has_src: number;
    path_has_config: number;
    path_has_generated: number;
}

export interface PredictionResult {
    probabilities: Record<'logistic_regression' | 'random_forest' | 'xgboost', number>;
    ensemble_surface_probability: number;
    threshold: number;
    decision: 'surface' | 'suppress';
}

export interface BatchPredictionResponse {
    model_version: string;
    threshold: number;
    predictions: PredictionResult[];
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
    const response = await fetch(`${ML_SERVICE_URL}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(body)
    });
    if (!response.ok) {
        throw new Error(`ML service returned ${response.status}: ${await response.text()}`);
    }
    return response.json() as Promise<T>;
}

export function predictSurface(finding: FindingFeatures): Promise<PredictionResult> {
    return postJson<PredictionResult>('/predict', finding);
}

export function predictSurfaceBatch(
    findings: FindingFeatures[]
): Promise<BatchPredictionResponse> {
    return postJson<BatchPredictionResponse>('/predict/batch', { items: findings });
}
