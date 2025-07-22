from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import yaml
import numpy as np

app = Flask(__name__)
CORS(app)  # allowing all domains

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

try:
    cfg = load_config()
    svc_model = joblib.load("svc_model.joblib")
    rf_model = joblib.load("rf_model.joblib")
    xgb_model = joblib.load("xgb_model.joblib")
    label_encoder = joblib.load("label_encoder.joblib")

    model_to_use = cfg.get("model_to_use", 0)  # default to 0 (svc)

    embedding_method = None
    embedding_model = None
    if cfg.get("embedding_provider") == "openai":
        import openai
        openai.api_key = cfg["openai"]["api_key"]
        embedding_method = "openai"
    else:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(cfg["transformers"]["model_name"])
        embedding_method = "local"

except Exception as e:
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

def top_n_predictions(model, X, label_encoder, n=3):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]  # probs for single sample
        top_n_idx = np.argsort(proba)[::-1][:n]
        labels = label_encoder.inverse_transform(top_n_idx)
        scores = proba[top_n_idx]
        return list(zip(labels, scores))
    else:
        # fallback: use predict and assign score 1.0 for predicted class
        pred = model.predict(X)
        label = label_encoder.inverse_transform(pred)[0]
        return [(label, 1.0)]

model_map = {
    0: ("SVC", svc_model),
    1: ("RandomForest", rf_model),
    2: ("XGBoost", xgb_model)
}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}
    summary = data.get("summary", "")
    description = data.get("description", "")

    if not summary and not description:
        return jsonify({"success": False, "error": "Provide at least summary or description."}), 400

    combined_text = summary + " " + description
    try:
        emb = get_embedding(combined_text).reshape(1, -1)
    except Exception as e:
        return jsonify({"success": False, "error": f"Embedding error: {str(e)}"}), 500

    try:
        model_name, model_obj = model_map.get(model_to_use, (None, None))
        if model_obj is None:
            return jsonify({"success": False, "error": "Invalid model_to_use config value."}), 500

        preds = top_n_predictions(model_obj, emb, label_encoder)

    except Exception as e:
        return jsonify({"success": False, "error": f"Inference error: {str(e)}"}), 500

    return jsonify({
        "predictions": {
            model_name: [{"label": label, "score": float(score)} for label, score in preds]
        },
        "success": True
    })


@app.errorhandler(Exception)
def handle_global_exception(e):
    return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
