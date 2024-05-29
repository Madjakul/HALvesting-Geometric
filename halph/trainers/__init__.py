# halph/trainers/__init__.py

from halph.trainers.train_link_prediction import LinkPredictionTrainer
from halph.trainers.train_node_classification import NodeClassificationTrainer

__all__ = ["NodeClassificationTrainer", "LinkPredictionTrainer"]
