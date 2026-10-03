# PRism ML Origin Classification Service

Standalone Python ML backend service for the PRism code review platform.
Loads the exact `prism_ensemble_clean.joblib` artifact (Random Forest + Logistic Regression + XGBoost) and provides high-throughput FastAPI inference endpoints to classify ESLint findings as `INTRODUCED` or `PRE_EXISTING`.

## Local Windows Development

1. Create a fresh Python 3.11 environment:
   ```powershell
   py -3.11 -m venv .venv
   ```
2. Activate and install dependencies:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
3. Run the regression/inspection scripts:
   ```powershell
   .\.venv\Scripts\python.exe scripts\inspect_model.py
   ```
4. Start the server locally:
   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```

Verify API is healthy:
- Health: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- Model Info: [http://127.0.0.1:8000/model-info](http://127.0.0.1:8000/model-info)
- Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Run tests:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/
```

## Docker Deployment (Cloud / Render)

Build the Docker image:
```bash
docker build -t prism-model-service .
```

Run the container:
```bash
docker run -p 8000:8000 prism-model-service
```

This Dockerfile is optimized for deployment to Render, AWS App Runner, GCP Cloud Run, or any container platform that uses the `$PORT` environment variable.

## Node.js Integration

See `examples/node-client.ts` for a TypeScript example on how to construct requests to this backend. Do not perform any ML preprocessing inside the Node backend. Just forward the ESLint finding JSON features matching `model_schema.json`.
