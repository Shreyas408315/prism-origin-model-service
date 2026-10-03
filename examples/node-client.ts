/**
 * PRism Origin Classification Model Client (Node.js / TypeScript)
 * 
 * Production-ready client implementation for calling the standalone Python ML backend.
 * Adheres strictly to the 22-feature contract and includes timeout, retry/fallback handling,
 * and resilient non-crashing failure modes.
 */

const ML_SERVICE_URL = (process.env.ML_SERVICE_URL || 'http://localhost:8000').replace(/\/+$/, '');
const DEFAULT_TIMEOUT_MS = parseInt(process.env.ML_SERVICE_TIMEOUT_MS || '5000', 10);

export interface FindingFeatures {
    rule_id: string;
    rule_family: string;
    severity: number;
    is_error: number;
    message_length: number;
    has_fix: number;
    fix_text_length: number;
    fix_range_length: number;
    has_suggestions: number;
    suggestion_count: number;
    changed_line_count: number;
    file_size_lines: number;
    start_line: number;
    finding_start_line_ratio: number;
    finding_span_lines: number;
    finding_span_columns: number;
    pr_change_code_lines: number;
    pr_total_findings_in_file: number;
    same_rule_findings_in_file: number;
    same_rule_findings_in_repo: number;
    finding_overlaps_change: number;
    finding_change_distance: number;
}

export interface PredictionResult {
    model_version: string;
    positive_class: string;
    risk_score: number;
    decision: 'INTRODUCED' | 'PRE_EXISTING';
    threshold: number;
    component_scores: {
        random_forest: number;
        logistic_regression: number;
        xgboost: number;
        [key: string]: number;
    };
}

export interface BatchPredictionResponse {
    model_version: string;
    threshold: number;
    predictions: PredictionResult[];
}

export interface SafePredictionOutcome<T = FindingFeatures> {
    success: boolean;
    rawFinding: T;
    prediction?: PredictionResult;
    mlStatus: 'AVAILABLE' | 'UNAVAILABLE' | 'ERROR';
    fallbackDecision?: 'INTRODUCED' | 'PRE_EXISTING';
    error?: string;
}

/**
 * Validate that the server response conforms strictly to the expected schema.
 */
function validatePredictionResult(data: any): data is PredictionResult {
    if (!data || typeof data !== 'object') return false;
    if (typeof data.risk_score !== 'number' || isNaN(data.risk_score)) return false;
    if (data.decision !== 'INTRODUCED' && data.decision !== 'PRE_EXISTING') return false;
    if (typeof data.threshold !== 'number') return false;
    if (typeof data.component_scores !== 'object' || data.component_scores === null) return false;
    return true;
}

/**
 * Send single prediction request to ML service.
 * Throws on network, timeout, 4xx, or 5xx error.
 */
export async function predictOrigin(
    finding: FindingFeatures,
    timeoutMs = DEFAULT_TIMEOUT_MS
): Promise<PredictionResult> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
        const response = await fetch(`${ML_SERVICE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(finding),
            signal: controller.signal
        });

        if (!response.ok) {
            const errorText = await response.text().catch(() => '');
            if (response.status >= 400 && response.status < 500) {
                throw new Error(`[ML Client 4xx Error] Status ${response.status}: Invalid feature payload - ${errorText}`);
            } else if (response.status >= 500) {
                throw new Error(`[ML Service 5xx Error] Status ${response.status}: Inference service failure - ${errorText}`);
            }
            throw new Error(`[ML Service Error] Status ${response.status}: ${errorText}`);
        }

        const data = await response.json();
        if (!validatePredictionResult(data)) {
            throw new Error('[ML Client Error] Received malformed prediction schema from ML service.');
        }

        return data;
    } catch (err: any) {
        if (err.name === 'AbortError') {
            throw new Error(`[ML Client Timeout] ML service timed out after ${timeoutMs}ms`);
        }
        throw err;
    } finally {
        clearTimeout(timeoutId);
    }
}

/**
 * Send batch prediction request to ML service.
 */
export async function predictOriginBatch(
    findings: FindingFeatures[],
    timeoutMs = DEFAULT_TIMEOUT_MS * 2
): Promise<BatchPredictionResponse> {
    if (findings.length === 0) {
        return {
            model_version: 'prism-origin-ensemble-clean-v1',
            threshold: 0.574674670640332,
            predictions: []
        };
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
        const response = await fetch(`${ML_SERVICE_URL}/predict/batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({ items: findings }),
            signal: controller.signal
        });

        if (!response.ok) {
            const errorText = await response.text().catch(() => '');
            throw new Error(`[ML Service Batch Error] Status ${response.status}: ${errorText}`);
        }

        const data = await response.json() as BatchPredictionResponse;
        if (!data || !Array.isArray(data.predictions)) {
            throw new Error('[ML Client Error] Malformed batch response from ML service.');
        }
        return data;
    } catch (err: any) {
        if (err.name === 'AbortError') {
            throw new Error(`[ML Client Timeout] Batch prediction timed out after ${timeoutMs}ms`);
        }
        throw err;
    } finally {
        clearTimeout(timeoutId);
    }
}

/**
 * Resilient wrapper: NEVER crashes the PR review pipeline if ML service is down or errors.
 * Preserves raw findings and falls back gracefully.
 */
export async function predictOriginWithFallback<T extends FindingFeatures>(
    finding: T,
    timeoutMs = DEFAULT_TIMEOUT_MS
): Promise<SafePredictionOutcome<T>> {
    try {
        const prediction = await predictOrigin(finding, timeoutMs);
        return {
            success: true,
            rawFinding: finding,
            prediction,
            mlStatus: 'AVAILABLE'
        };
    } catch (error: any) {
        // Log error internally without throwing
        console.warn(`[PRism ML Origin Warning] Inference failed, executing fallback: ${error.message}`);
        
        // PRism Fallback Logic:
        // If overlaps with change lines, default to INTRODUCED; otherwise PRE_EXISTING
        const fallbackDecision = finding.finding_overlaps_change === 1 ? 'INTRODUCED' : 'PRE_EXISTING';

        return {
            success: false,
            rawFinding: finding,
            mlStatus: error.message.includes('Timeout') || error.message.includes('fetch failed') ? 'UNAVAILABLE' : 'ERROR',
            fallbackDecision,
            error: error.message
        };
    }
}

// Smoke test verification when executed directly with ts-node or node
async function runSmokeTest() {
    const sampleFinding: FindingFeatures = {
        rule_id: "no-unused-vars",
        rule_family: "possible-problems",
        severity: 2,
        is_error: 1,
        message_length: 46,
        has_fix: 0,
        fix_text_length: 0,
        fix_range_length: 0,
        has_suggestions: 1,
        suggestion_count: 1,
        changed_line_count: 4,
        file_size_lines: 6,
        start_line: 4,
        finding_start_line_ratio: 0.6666666666666666,
        finding_span_lines: 1,
        finding_span_columns: 8,
        pr_change_code_lines: 1,
        pr_total_findings_in_file: 2,
        same_rule_findings_in_file: 1,
        same_rule_findings_in_repo: 20,
        finding_overlaps_change: 1,
        finding_change_distance: 0
    };

    console.log(`Connecting to ML service at: ${ML_SERVICE_URL}`);
    const outcome = await predictOriginWithFallback(sampleFinding);
    console.log('Outcome result:', JSON.stringify(outcome, null, 2));
}

if (typeof require !== 'undefined' && require.main === module) {
    runSmokeTest().catch(console.error);
}
