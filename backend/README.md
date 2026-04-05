# Backend API Template

This backend is a FastAPI template for the multimodal lung disease support app.

## Setup

1. Create a Python virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.template` to `.env` and update model paths.

## Run the API

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

## API endpoints

- `GET /health` - service health check
- `POST /api/predict` - run inference with image/audio/tabular input

## Notes

The current backend returns placeholder branch predictions. Replace the stubbed evaluation functions in `backend/model_stub.py` with real model loading and preprocessing logic from the notebook.
