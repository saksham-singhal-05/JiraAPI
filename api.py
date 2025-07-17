from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import yaml
import numpy as np

app = FastAPI(
    title="Jira Automated Assignee API",
    description="API for predicting ticket assignee with ML models",
    version="1.0.0"
)

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
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Config loading failed: {e}")

def get_embedding(text, cfg):
    try:
        if cfg.get("embedding_provider") == "openai":
            import openai
            openai.api_key = cfg["openai"]["api_key"]
            emb = openai.embeddings.create(
                model=cfg["openai"]["embedding_model"],
                input=[text]
            ).data[0].embedding
            return np.array(emb, dtype=np.float32)
        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(cfg["transformers"]["model_name"])
            return model.encode([text])[0]
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")

# Load models and config on startup
try:
    cfg = load_config()
    xgb_model = joblib.load("xgb_model.joblib")
    svc_model = joblib.load("svc_model.joblib")
    rf_model = joblib.load("rf_model.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
except Exception as e:
    # You may want to log this more formally
    raise RuntimeError(f"Startup Model Load Error: {e}")

@app.post("/predict")
async def predict(ticket: Ticket):
    # Input validation
    summary = ticket.summary or ""
    description = ticket.description or ""
    if not summary and not description:
        raise HTTPException(status_code=400, detail="Provide at least summary or description.")

    combined_text = summary + " " + description

    # Compute embedding and predict
    try:
        emb = get_embedding(combined_text, cfg)
        emb = emb.reshape(1, -1)
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
