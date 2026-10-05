from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException

from app.schemas import (
    FindingFeatures,
    PredictionResult,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from app.inference import model_instance


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_instance.load_model()
    yield

app = FastAPI(
    title="PRism ESLint Surface Classification Service",
    description=(
        "Classifies engineered ESLint findings as SURFACE or SUPPRESS using "
        "the weighted Logistic Regression, Random Forest, and XGBoost hybrid ensemble."
    ),
    lifespan=lifespan,
    docs_url="/docs"
)

@app.get("/health", response_model=HealthResponse)
async def health():
    if not model_instance.is_loaded:
        return HealthResponse(
            status="error",
            model_version=model_instance.model_version,
            model_loaded=False
        )
        
    return HealthResponse(
        status="ok",
        model_version=model_instance.model_version,
        model_loaded=True
    )

@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    if not model_instance.is_loaded:
        raise HTTPException(
            status_code=503,
            detail={"error": {"code": "MODEL_UNAVAILABLE", "message": "Model not loaded"}},
        )
        
    return ModelInfoResponse(
        model_version=model_instance.model_version,
        positive_class=model_instance.positive_class,
        negative_class="suppressed",
        threshold=model_instance.default_threshold,
        feature_count=len(model_instance.features),
        input_feature_count=len(model_instance.api_features),
        model_family="hybrid_ensemble",
        component_models=list(model_instance.models),
        includes_message_tfidf=True,
    )

@app.post("/predict", response_model=PredictionResult)
async def predict(finding: FindingFeatures):
    if not model_instance.is_loaded:
        raise HTTPException(status_code=503, detail={"error": {"code": "MODEL_UNAVAILABLE", "message": "Model not loaded"}})
        
    try:
        return model_instance.predict(finding)
    except Exception as error:
        logger.exception("Surface model inference failed")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "MODEL_INFERENCE_FAILED",
                    "message": "Model inference failed.",
                }
            },
        ) from error

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    if not model_instance.is_loaded:
        raise HTTPException(status_code=503, detail={"error": {"code": "MODEL_UNAVAILABLE", "message": "Model not loaded"}})
        
    try:
        return model_instance.predict_batch(request.items)
    except Exception as error:
        logger.exception("Surface batch inference failed")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "MODEL_INFERENCE_FAILED",
                    "message": "Model inference failed.",
                }
            },
        ) from error
