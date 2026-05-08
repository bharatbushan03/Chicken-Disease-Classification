# Chicken Disease Classification

A Flask app that classifies chicken images as Coccidiosis or Healthy using a CNN.

## Features
- Browser UI plus JSON API
- Base64 image upload
- Optional training endpoint (disabled by default for deployment)

## Local setup
1. python -m venv .venv
2. .venv\Scripts\activate
3. pip install -r requirements.txt
4. python app.py
5. Open http://localhost:5000

## API
- POST /predict
	- Body: {"image": "data:image/jpeg;base64,..."}
	- Response: {"label": "Healthy", "confidence": 0.95}
- GET /health
- POST /train (requires ENABLE_TRAINING=true)

## Configuration
- MODEL_PATH: path to model file (default resolves to model.keras)
- ENABLE_TRAINING: set to true to enable /train

## Render deployment
1. Ensure model.keras is committed to the repo.
2. Push to GitHub.
3. Create a Render web service using render.yaml.
4. Set MODEL_PATH and ENABLE_TRAINING in Render if you want to override defaults.