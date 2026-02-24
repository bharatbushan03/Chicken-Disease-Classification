from pathlib import Path

from CNNClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from CNNClassifier.entity import (
    DataIngestionConfig,
    EvaluationConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
)
from CNNClassifier.utils import create_directories, read_yaml


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
        )

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([config.root_dir])
        return PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            params_image_size=self.params.model.image_size,
            params_input_shape=self.params.model.input_shape,
            params_num_classes=self.params.model.num_classes,
        )

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        data_ingestion = self.config.data_ingestion
        create_directories([training.root_dir])
        return TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            training_data=Path(data_ingestion.unzip_dir),
            params_epochs=self.params.train.epochs,
            params_batch_size=self.params.train.batch_size,
            params_learning_rate=self.params.train.learning_rate,
            params_image_size=self.params.model.image_size,
            params_train_split=self.params.data.train_split,
            params_seed=self.params.seed,
            params_horizontal_flip=self.params.augmentation.horizontal_flip,
            params_rotation_range=self.params.augmentation.rotation_range,
        )

    def get_evaluation_config(self) -> EvaluationConfig:
        model_eval = self.config.model_evaluation
        data_ingestion = self.config.data_ingestion
        create_directories([model_eval.root_dir])
        return EvaluationConfig(
            path_of_model=Path(self.config.training.trained_model_path),
            training_data=Path(data_ingestion.unzip_dir),
            all_params=dict(self.params),
            params_image_size=self.params.model.image_size,
            params_batch_size=self.params.train.batch_size,
            report_file=Path(model_eval.report_file),
        )
