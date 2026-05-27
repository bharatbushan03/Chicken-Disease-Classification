"""
CNNClassifier package for Chicken Disease Classification.
Provides modules for data ingestion, model preparation, training, and evaluation.
"""
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
