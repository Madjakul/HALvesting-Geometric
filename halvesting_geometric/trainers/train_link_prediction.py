# halvesting_geometric/trainers/train_link_prediction.py

import logging

import torch
import torch.nn as nn
import wandb
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from halvesting_geometric.models import LinkPrediction


class LinkPredictionTrainer:
    """Trainer for link prediction models.

    Parameters
    ----------
    model: LinkPrediction
        Link prediction model to train.
    lr: float
        Learning rate.
    device: str
        Device to use for training.
    weight_decay: float
        Weight decay for the optimizer.

    Attributes
    ----------
    model: LinkPrediction
        Link prediction model to train.
    device: str
        Device to use for training.
    weight_decay: float
        Weight decay for the optimizer.
    optimizer: torch.optim.Adam
        Optimizer to use for training.
    criterion: nn.BCEWithLogitsLoss
        Loss function to use for training.

    Examples
    --------
    >>> from halvesting_geometric.models import LinkPrediction
    >>> from halvesting_geometric.trainers import LinkPredictionTrainer
    >>> model = LinkPrediction(128, 128)
    >>> trainer = LinkPredictionTrainer(model, 0.001, "cuda")
    >>> trainer.train(train_dataloader, val_dataloader, 10)
    """

    def __init__(
        self,
        model: LinkPrediction,
        lr: float,
        device: str,
        weight_decay: float = 0,
    ):
        self.model = model
        self.device = device
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def fit(self, train_dataloader: NeighborLoader):
        """Fit the model to the training data.

        Parameters
        ----------
        train_dataloader : NeighborLoader
            Dataloader for the training data.

        Returns
        -------
        float
            Loss of the model on the training data.
        """
        self.model.train()
        total_loss = 0
        total_examples = 0
        for batch in tqdm(train_dataloader):
            self.optimizer.zero_grad()
            batch = batch.to(self.device)
            y = batch["author", "writes", "paper"].edge_label
            out = self.model(batch)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss) * out.numel()
            total_examples += out.numel()
        return total_loss / total_examples

    @torch.no_grad()
    def validate(self, val_dataloader: NeighborLoader):
        """Validate the model on the validation data.

        Parameters
        ----------
        val_dataloader : NeighborLoader
            Dataloader for the validation data.

        Returns
        -------
        float
            Area under the ROC curve on the validation data.
        """
        self.model.eval()
        y_true = []
        y_pred = []
        for batch in tqdm(val_dataloader):
            batch = batch.to(self.device)
            y_true.append(batch["author", "writes", "paper"].edge_label)
            y_pred.append(self.model(batch))
        y_true = torch.cat(y_true, dim=0).cpu().numpy()
        y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
        auc = roc_auc_score(y_true, y_pred)
        return auc

    def train(
        self,
        train_dataloader: NeighborLoader,
        val_dataloader: NeighborLoader,
        epochs: int,
    ):
        """Train the model.

        Parameters
        ----------
        train_dataloader : NeighborLoader
            Dataloader for the training data.
        val_dataloader : NeighborLoader
            Dataloader for the validation data.
        epochs : int
            Number of epochs to train the model for.
        """
        for epoch in tqdm(range(epochs)):
            logging.info(f"Epoch {epoch + 1:02d}")
            loss = self.fit(train_dataloader)
            wandb.log({"loss": loss})
            logging.info(f"Loss {loss:.4f}")
            auc = self.validate(val_dataloader)
            wandb.log({"val-auc": auc}, commit=False)
            logging.info(f"Val AUC: {auc:.4f}")
