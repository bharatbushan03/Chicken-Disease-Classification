import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from CNNClassifier.config.configuration import ConfigurationManager
from CNNClassifier.components import DataIngestion, Evaluation, PrepareBaseModel, Training

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Pipeline Stage Classes
# ─────────────────────────────────────────────

STAGE_NAME_01 = "Data Ingestion"
STAGE_NAME_02 = "Prepare Base Model"
STAGE_NAME_03 = "Training"
STAGE_NAME_04 = "Model Evaluation"


class Stage01_DataIngestion:
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.initiate_data_ingestion()


class Stage02_PrepareBaseModel:
    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()


class Stage03_Training:
    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.initiate_training()


class Stage04_Evaluation:
    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(config=eval_config)
        evaluation.initiate_evaluation()


# ─────────────────────────────────────────────
# Prediction Pipeline
# ─────────────────────────────────────────────

# Class names inferred from Kaggle dataset folder order (alphabetical)
CLASS_NAMES = ["Coccidiosis", "Healthy"]
IMAGE_SIZE = (224, 224)


class PredictionPipeline:
    """Loads a trained model and predicts the disease class for a given image."""

    def __init__(self, model_path: str = "model.keras"):
        self.model_path = Path(model_path)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found at '{self.model_path}'. "
                    "Run training first."
                )
            self._model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Model loaded from '{self.model_path}'.")
        return self._model

    def predict(self, image: Image.Image) -> dict:
        """
        Args:
            image: PIL Image object.
        Returns:
            dict with keys 'label' (str) and 'confidence' (float 0-1).
        """
        img = image.convert("RGB").resize(IMAGE_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

        prob = float(self.model.predict(arr, verbose=0)[0][0])

        # Sigmoid output: >0.5 → class 1, else class 0
        # Alphabetical: 0=Coccidiosis, 1=Healthy
        if prob > 0.5:
            label = CLASS_NAMES[1]  # Healthy
            confidence = prob
        else:
            label = CLASS_NAMES[0]  # Coccidiosis
            confidence = 1.0 - prob

        return {"label": label, "confidence": round(confidence, 4)}
