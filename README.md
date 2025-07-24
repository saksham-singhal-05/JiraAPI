# JiraAPI

## Issue Category Prediction API

Predicts the category of an issue report from its text fields using a trained ML model and text embeddings.

### Run Locally

1. Install dependencies:
pip install -r requirements.txt

2. Make sure `config.yaml`, `svc_model.joblib`,`rf_model.joblib`,`xgb_model.joblib`, and `label_encoder.joblib` are in the same folder.

3. Start the API:
python FlaskApp.py

