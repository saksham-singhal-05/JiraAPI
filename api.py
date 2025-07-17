from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import yaml
import numpy as np

app = FastAPI(
    title="Jira Automated Assignee API",
    description="Optimized API for predicting ticket assignee",
    version="1.0.0"
)

# Allow CORS for any origin (adjust for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Ticket(BaseModel):
    summary: str = ""
    description: str = ""

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

# Load all models and config ONCE at startup (outside endpoints)
try:
    cfg = load_config()
    xgb_model = joblib.load("xgb_model.joblib")
    svc_model = joblib.load("svc_model.joblib")
    rf_model = joblib.load("rf_model.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
    embedding_method = None
    embedding_model = None
    if cfg.get("embedding_provider") == "openai":
        import openai
        openai.api_key = cfg["openai"]["api_key"]
        embedding_method = "openai"
        # No heavy model loaded for OpenAI
    else:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(cfg["transformers"]["model_name"])
        embedding_method = "local"
except Exception as e:
    # All errors on startup will prevent API from serving
    raise RuntimeError(f"Model or config loading failed: {e}")

def get_embedding(text):
    try:
        if embedding_method == "openai":
            import openai
            emb = openai.embeddings.create(
                model=cfg["openai"]["embedding_model"], input=[text]
            ).data[0].embedding
            return np.array(emb, dtype=np.float32)
        elif embedding_method == "local":
            return embedding_model.encode([text])[0]
        else:
            raise RuntimeError("No valid embedding method configured.")
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")

@app.post("/predict")
async def predict(ticket: Ticket):
    summary = ticket.summary or ""
    description = ticket.description or ""
    if not summary and not description:
        raise HTTPException(status_code=400, detail="Provide at least summary or description.")

    combined_text = summary + " " + description
    try:
        emb = get_embedding(combined_text).reshape(1, -1)
        pred_xgb = label_encoder.inverse_transform(xgb_model.predict(emb))[0]
        pred_svc = label_encoder.inverse_transform(svc_model.predict(emb))[0]
        pred_rf = label_encoder.inverse_transform(rf_model.predict(emb))[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return {
        "predictions": {
            "XGBoost": pred_xgb,
            "SVC": pred_svc,
            "RandomForest": pred_rf
        },
        "success": True
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return {
        "success": False,
        "error": str(exc)
    }
