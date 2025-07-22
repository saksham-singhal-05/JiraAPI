from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import yaml
import numpy as np


app = Flask(__name__)
CORS(app)  #allowing all domains


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


try:
    cfg = load_config()
    svc_model = joblib.load("svc_model.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
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


#local setup
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
        pred_svc = label_encoder.inverse_transform(svc_model.predict(emb))[0]
    except Exception as e:
        return jsonify({"success": False, "error": f"Inference error: {str(e)}"}), 500


    return jsonify({
        "predictions": {"SVC": pred_svc},
        "success": True
    })


@app.errorhandler(Exception)
def handle_global_exception(e):
    return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
    import requests
