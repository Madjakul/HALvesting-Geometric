# halph/trainers/train_node_classification.py

import logging

import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
from torcheval.metrics.functional import multiclass_f1_score
from tqdm import tqdm

from halph.models import NodeClassification


class NodeClassificationTrainer:
    """Pass.

    Parameters
    ----------

    Attributes
    ----------

    References
    ----------
    """

    def __init__(
        self, model: NodeClassification, lr: float, device: str, weight_decay: float = 0
    ):
        self.model = model
        self.device = device
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, train_dataloader: NeighborLoader):
        self.model.train()
        total_loss = 0
        total_examples = 0
        total_correct = 0
        for batch in tqdm(train_dataloader):
            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            batch_size = batch["paper"].batch_size
            y = batch["paper"].y[:batch_size]
            # print(batch.x_dict)
            out = self.model(batch)
            out = out[:batch_size]
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()
            total_examples += batch_size
            total_loss += float(loss) * batch_size
            total_correct += int((out.argmax(dim=-1) == y).sum())
        return (total_loss / total_examples), (total_correct / total_examples)

    @torch.no_grad()
    def validate(self, val_dataloader: NeighborLoader):
        self.model.eval()
        total_examples = 0
        total_correct = 0
        total_f1 = 0
        y_true = []
        y_pred = []
        for batch in tqdm(val_dataloader):
            batch = batch.to(self.device)
            batch_size = batch["paper"].batch_size
            y = batch["paper"].y[:batch_size]
            pred = self.model(batch)[:batch_size].argmax(dim=-1)
            total_examples += batch_size
            total_correct += int((pred == y).sum())
            y_true.extend(y.cpu())
            y_pred.extend(pred.cpu())
        f1 = multiclass_f1_score(
            torch.LongTensor(y_pred),
            torch.LongTensor(y_true),
            num_classes=12,
            average="macro",
        )
        return f1, (total_correct / total_examples)

    def train(
        self,
        train_dataloader: NeighborLoader,
        val_dataloader: NeighborLoader,
        epochs: int,
    ):
        for epoch in tqdm(range(epochs)):
            logging.info(f"Epoch {epoch + 1:02d}")
            loss, train_accuracy = self.fit(train_dataloader)
            logging.info(f"Loss {loss:.4f}, Train Acc: {train_accuracy:.4f}")
            val_f1, val_accuracy = self.validate(val_dataloader)
            logging.info(f"Val F1: {val_f1:.4f}, Val Acc: {val_accuracy:.4f}")
