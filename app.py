import base64
import io
import logging
import os

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image

from CNNClassifier.pipeline import (
    PredictionPipeline,
    Stage01_DataIngestion,
    Stage02_PrepareBaseModel,
    Stage03_Training,
    Stage04_Evaluation,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Instantiate prediction pipeline once at startup (lazy-loads model on first call)
predictor = PredictionPipeline(model_path="model.keras")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body: { "image": "<base64-encoded image data>" }
    The base64 string may include a data-URL prefix (data:image/...;base64,...)
    or be raw base64.
    Returns JSON: { "label": "...", "confidence": 0.95 }
    """
    try:
        data = request.get_json(force=True)
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        image_data = data["image"]
        # Strip data-URL prefix if present
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        img_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(img_bytes))

        result = predictor.predict(image)
        return jsonify(result)

    except Exception as e:
        logger.exception("Error during prediction")
        return jsonify({"error": str(e)}), 500


@app.route("/train", methods=["GET"])
def train():
    """Trigger the full training pipeline."""
    try:
        logger.info(">> Stage 01: Data Ingestion")
        Stage01_DataIngestion().main()

        logger.info(">> Stage 02: Prepare Base Model")
        Stage02_PrepareBaseModel().main()

        logger.info(">> Stage 03: Training")
        Stage03_Training().main()

        logger.info(">> Stage 04: Evaluation")
        Stage04_Evaluation().main()

        return jsonify({"status": "Training complete"})
    except Exception as e:
        logger.exception("Error during training")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
