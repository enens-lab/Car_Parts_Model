"""Inference: load fine-tuned checkpoints, predict, serialise, draw."""
from .predictor import CarPartsModel, CarPartsPipeline, Prediction

__all__ = ["CarPartsModel", "CarPartsPipeline", "Prediction"]
