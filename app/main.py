from fastapi import FastAPI, HTTPException, status
from contextlib import asynccontextmanager

from app.schemas import (
    FindingFeatures,
    PredictionResult,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from app.inference import model_instance

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model exactly once on startup
    try:
        model_instance.load_model()
    except Exception as e:
        print(f"Failed to load model on startup: {e}")
        # Not exiting so the health endpoint can explicitly report failure
    yield

app = FastAPI(
    title="PRism ML Origin Classification Service",
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
        raise HTTPException(status_code=503, detail={"error": {"code": "MODEL_UNAVAILABLE", "message": "Model not loaded"}})
        
    return ModelInfoResponse(
        model_version=model_instance.model_version,
        positive_class=model_instance.positive_class,
        threshold=model_instance.default_threshold,
        feature_count=len(model_instance.features),
        model_family="ensemble"
    )

@app.post("/predict", response_model=PredictionResult)
async def predict(finding: FindingFeatures):
    if not model_instance.is_loaded:
        raise HTTPException(status_code=503, detail={"error": {"code": "MODEL_UNAVAILABLE", "message": "Model not loaded"}})
        
    try:
        return model_instance.predict(finding)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": {"code": "MODEL_INFERENCE_FAILED", "message": "Model inference failed."}})

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    if not model_instance.is_loaded:
        raise HTTPException(status_code=503, detail={"error": {"code": "MODEL_UNAVAILABLE", "message": "Model not loaded"}})
        
    try:
        return model_instance.predict_batch(request.items)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": {"code": "MODEL_INFERENCE_FAILED", "message": "Model inference failed."}})
