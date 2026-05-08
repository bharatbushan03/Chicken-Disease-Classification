import base64
import binascii
import io
import logging
import os
from pathlib import Path

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
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024  # 6 MB


def resolve_model_path() -> str:
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return env_path
    if Path("model.keras").exists():
        return "model.keras"
    return "artifacts/training/model.keras"


def training_enabled() -> bool:
    return os.getenv("ENABLE_TRAINING", "false").strip().lower() in {"1", "true", "yes"}

# Instantiate prediction pipeline once at startup (lazy-loads model on first call)
predictor = PredictionPipeline(model_path=resolve_model_path())


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    model_path = resolve_model_path()
    return jsonify(
        {
            "status": "ok",
            "model_path": model_path,
            "model_exists": Path(model_path).exists(),
            "training_enabled": training_enabled(),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body: { "image": "<base64-encoded image data>" }
    The base64 string may include a data-URL prefix (data:image/...;base64,...)
    or be raw base64.
    Returns JSON: { "label": "...", "confidence": 0.95 }
    """
    try:
        data = request.get_json(silent=True)
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        image_data = data["image"]
        if not isinstance(image_data, str) or not image_data.strip():
            return jsonify({"error": "Invalid image payload"}), 400
        # Strip data-URL prefix if present
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        try:
            img_bytes = base64.b64decode(image_data, validate=True)
        except (binascii.Error, ValueError):
            return jsonify({"error": "Invalid base64 image data"}), 400

        try:
            image = Image.open(io.BytesIO(img_bytes))
            image.load()
        except Exception:
            return jsonify({"error": "Invalid image file"}), 400

        result = predictor.predict(image)
        return jsonify(result)

    except FileNotFoundError as e:
        logger.error("Model file missing: %s", e)
        return jsonify({"error": "Model file not found"}), 503
    except Exception as e:
        logger.exception("Error during prediction")
        return jsonify({"error": str(e)}), 500


@app.route("/train", methods=["POST"])
def train():
    """Trigger the full training pipeline."""
    try:
        if not training_enabled():
            return (
                jsonify({"error": "Training is disabled. Set ENABLE_TRAINING=true to enable."}),
                403,
            )
        logger.info(">> Stage 01: Data Ingestion")
        Stage01_DataIngestion().main()

        logger.info(">> Stage 02: Prepare Base Model")
        Stage02_PrepareBaseModel().main()

        logger.info(">> Stage 03: Training")
        Stage03_Training().main()

        logger.info(">> Stage 04: Evaluation")
        Stage04_Evaluation().main()

        predictor.reset_model()

        return jsonify({"status": "Training complete"})
    except Exception as e:
        logger.exception("Error during training")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
