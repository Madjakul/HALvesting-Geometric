# halvesting_geometric/modules/modeling_link_prediction.py

from typing import Literal

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero
from torch_geometric.typing import Metadata
from torcheval.metrics import BinaryAUROC

from halvesting_geometric.modules.gat import GAT
from halvesting_geometric.modules.link_classifier import LinkClassifier
from halvesting_geometric.modules.rggc import RGGC
from halvesting_geometric.modules.sage import GraphSage


class LinkPrediction(L.LightningModule):
    """Link prediction model for heterogeneous graphs. The model consists of a GNN that
    embeds the nodes of the graph and a link classifier that predicts the existence of
    edges between nodes. The model is trained using binary cross-entropy loss.

    Parameters
    ----------
    gnn : Literal["sage", "gat", "rggc"]
        Type of GNN to use.
    metadata : Metadata
        Metadata of the graph.
    batch_size : int
        Number of samples in a batch.
    author_num_nodes : int
        Number of author nodes.
    institution_num_nodes : int
        Number of institution nodes.
    domain_num_nodes : int
        Number of domain nodes.
    hidden_channels : int
        Number of hidden channels in the GNN.
    dropout : float
        Dropout probability.
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay.

    Attributes
    ----------
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay.
    batch_size : int
        Number of samples in a batch.
    metric : BinaryAUROC
        Metric for validation.
    paper_embedding : nn.Embedding
        Embedding layer for paper nodes.
    author_embedding : nn.Embedding
        Embedding layer for author nodes.
    domain_embedding : nn.Embedding
        Embedding layer for domain nodes.
    institution_embedding : nn.Embedding
        Embedding layer for institution nodes.
    gnn : Union[GraphSage, GAT, RGGC]
        GNN model.
    classifier : LinkClassifier
        Link classifier.
    gnn_map : dict
        Mapping from GNN types to classes.

    Examples
    --------
    >>> from torch_geometric.typing import Metadata
    >>> metadata = Metadata()
    >>> model = LinkPrediction(
    ...     gnn="sage",
    ...     metadata=metadata,
    ...     batch_size=64,
    ...     author_num_nodes=1000,
    ...     institution_num_nodes=100,
    ...     domain_num_nodes=10,
    ...     hidden_channels=64,
    ...     dropout=0.5,
    ...     lr=0.01,
    ...     weight_decay=0.001,
    ... )
    """

    gnn_map = {"sage": GraphSage, "gat": GAT, "rggc": RGGC}

    def __init__(
        self,
        gnn: Literal["sage", "gat", "rggc"],
        metadata: Metadata,
        batch_size: int,
        author_num_nodes: int,
        institution_num_nodes: int,
        domain_num_nodes: int,
        hidden_channels: int,
        dropout: float,
        lr: float,
        weight_decay: float,
    ):
        super().__init__()
        # Training parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        # Validation parameters
        self.metric = BinaryAUROC()
        # Model parameters
        # Embedding layers for all node types
        self.paper_embedding = nn.Embedding(100000, hidden_channels)
        self.author_embedding = nn.Embedding(author_num_nodes, hidden_channels)
        self.domain_embedding = nn.Embedding(domain_num_nodes, hidden_channels)
        self.institution_embedding = nn.Embedding(
            institution_num_nodes, hidden_channels
        )
        # Instantiate homogeneous GNN:
        self.gnn = self.gnn_map[gnn](
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            dropout=dropout,
        )
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=metadata)
        self.classifier = LinkClassifier()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def forward(self, batch: HeteroData):
        x_dict = {
            "author": self.author_embedding(batch["author"].n_id),
            "affiliation": self.institution_embedding(batch["affiliation"].n_id),
            "domain": self.domain_embedding(batch["domain"].n_id),
            "paper": self.paper_embedding(batch["paper"].n_id % 100000),
        }
        x_dict = self.gnn(x_dict, batch.edge_index_dict)
        pred = self.classifier(
            x_author=x_dict["author"],
            x_paper=x_dict["paper"],
            edge_label_index=batch["author", "writes", "paper"].edge_label_index,
        )
        return pred

    def test_step(self, batch: HeteroData, batch_idx: int):
        y = batch["author", "writes", "paper"].edge_label
        out = self(batch)
        self.metric.update(out, y)
        loss = F.binary_cross_entropy_with_logits(out, y)
        roc_auc = self.metric.compute()
        self.log_dict(
            {"test_loss": float(loss), "test_roc_auc": float(roc_auc)},
            # on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

    def training_step(self, batch: HeteroData, batch_idx: int):
        y = batch["author", "writes", "paper"].edge_label
        out = self(batch)
        loss = F.binary_cross_entropy_with_logits(out, y)
        self.log(
            "train_loss",
            float(loss),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        return loss

    def validation_step(self, batch: HeteroData, batch_idx: int):
        y = batch["author", "writes", "paper"].edge_label
        out = self(batch)
        self.metric.update(out, y)
        loss = F.binary_cross_entropy_with_logits(out, y)
        roc_auc = self.metric.compute()
        self.log_dict(
            {"val_loss": float(loss), "val_roc_auc": float(roc_auc)},
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
