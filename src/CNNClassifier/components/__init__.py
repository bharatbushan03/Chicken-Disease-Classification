import logging
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    MaxPooling2D,
    RandomFlip,
    RandomRotation,
)
from tensorflow.keras.optimizers import Adam

from CNNClassifier.entity import (
    DataIngestionConfig,
    EvaluationConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
)
from CNNClassifier.utils import create_directories, save_json

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. Data Ingestion
# ─────────────────────────────────────────────
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        src = Path(self.config.local_data_file)
        dst = Path(self.config.unzip_dir)
        if not src.exists():
            logger.warning(
                f"Source data directory '{src}' not found. "
                "Skipping ingestion – place your dataset at that path."
            )
            return
        if dst.exists() and dst != src:
            shutil.rmtree(dst)
        if dst != src:
            shutil.copytree(src, dst)
            logger.info(f"Dataset copied from '{src}' to '{dst}'.")
        else:
            logger.info(f"Dataset already at '{dst}'. No copy needed.")


# ─────────────────────────────────────────────
# 2. Prepare Base Model
# ─────────────────────────────────────────────
class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    @staticmethod
    def _build_model(input_shape: list, num_classes: int) -> Sequential:
        """Builds the same CNN architecture used in the notebook."""
        model = Sequential([
            Conv2D(32, 3, activation="relu", input_shape=tuple(input_shape)),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(64, 3, activation="relu"),
            Conv2D(64, 3, activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, 3, activation="relu"),
            Conv2D(128, 3, activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),

            GlobalAveragePooling2D(),

            Dense(128, activation="relu"),
            Dense(64, activation="relu"),

            Dense(1, activation="sigmoid"),
        ])
        return model

    def get_base_model(self):
        self.model = self._build_model(
            input_shape=self.config.params_input_shape,
            num_classes=self.config.params_num_classes,
        )
        self.save_model(self.config.base_model_path, self.model)
        logger.info(f"Base model saved at: {self.config.base_model_path}")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)


# ─────────────────────────────────────────────
# 3. Training
# ─────────────────────────────────────────────
class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def _load_dataset(self):
        image_size = tuple(self.config.params_image_size)
        data_dir = self.config.training_data

        full_ds = tf.keras.utils.image_dataset_from_directory(
            directory=str(data_dir),
            labels="inferred",
            label_mode="binary",
            image_size=image_size,
            batch_size=self.config.params_batch_size,
            shuffle=True,
            seed=self.config.params_seed,
        )

        n_batches = tf.data.experimental.cardinality(full_ds).numpy()
        train_size = int(n_batches * self.config.params_train_split)

        augmentation_layers = []
        if self.config.params_horizontal_flip:
            augmentation_layers.append(RandomFlip("horizontal"))
        if self.config.params_rotation_range:
            factor = self.config.params_rotation_range / 360.0
            augmentation_layers.append(RandomRotation(factor))

        train_ds = full_ds.take(train_size)
        val_ds = full_ds.skip(train_size)

        # Apply augmentation only to training data
        if augmentation_layers:
            aug_model = tf.keras.Sequential(augmentation_layers)
            train_ds = train_ds.map(
                lambda x, y: (aug_model(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        # Normalise pixels to [0, 1]
        normalise = lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)
        train_ds = train_ds.map(normalise).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.map(normalise).prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds

    def initiate_training(self):
        data_dir = Path(self.config.training_data)
        if not data_dir.exists():
            logger.error(
                f"Training data directory '{data_dir}' not found. "
                "Run data ingestion first."
            )
            return

        # Use existing trained model if present, else build from scratch
        model_path = Path("model.keras")
        if model_path.exists():
            logger.info(f"Loading existing model from '{model_path}'.")
            model = tf.keras.models.load_model(model_path)
        else:
            from CNNClassifier.components import PrepareBaseModel
            from CNNClassifier.config.configuration import ConfigurationManager
            cm = ConfigurationManager()
            pbm = PrepareBaseModel(cm.get_prepare_base_model_config())
            pbm.get_base_model()
            model = tf.keras.models.load_model(
                cm.get_prepare_base_model_config().base_model_path
            )

        model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=self.config.params_learning_rate),
            metrics=["accuracy"],
        )

        train_ds, val_ds = self._load_dataset()
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.params_epochs,
        )

        create_directories([self.config.root_dir])
        model.save(self.config.trained_model_path)
        logger.info(f"Trained model saved at: {self.config.trained_model_path}")


# ─────────────────────────────────────────────
# 4. Evaluation
# ─────────────────────────────────────────────
class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _load_val_dataset(self):
        image_size = tuple(self.config.params_image_size)
        full_ds = tf.keras.utils.image_dataset_from_directory(
            directory=str(self.config.training_data),
            labels="inferred",
            label_mode="binary",
            image_size=image_size,
            batch_size=self.config.params_batch_size,
            shuffle=False,
        )
        n_batches = tf.data.experimental.cardinality(full_ds).numpy()
        val_size = max(1, int(n_batches * 0.2))
        val_ds = full_ds.take(val_size)
        normalise = lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)
        return val_ds.map(normalise).prefetch(tf.data.AUTOTUNE)

    def evaluation(self):
        model = tf.keras.models.load_model(self.config.path_of_model)
        val_ds = self._load_val_dataset()
        loss, accuracy = model.evaluate(val_ds)
        self.score = {"loss": loss, "accuracy": accuracy}
        logger.info(f"Evaluation results — loss: {loss:.4f}, accuracy: {accuracy:.4f}")

    def save_score(self):
        save_json(path=self.config.report_file, data=self.score)
        logger.info(f"Evaluation report saved at: {self.config.report_file}")

    def initiate_evaluation(self):
        self.evaluation()
        self.save_score()
