# halvesting_geometric/models/modeling_link_prediction.py

from typing import Literal

import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero
from torch_geometric.typing import Metadata

from halvesting_geometric.models.gat import GAT
from halvesting_geometric.models.link_classifier import LinkClassifier
from halvesting_geometric.models.rggc import RGGC
from halvesting_geometric.models.sage import GraphSage

_GNN_MAP = {"sage": GraphSage, "gat": GAT, "rggc": RGGC}


class LinkPrediction(nn.Module):
    """Link prediction model. This model predicts whether an author has written a paper.

    Parameters
    ----------
    gnn: Literal["sage", "gat", "rggc"]
        Graph neural network model to use.
    metadata: Metadata
        Metadata object.
    paper_num_nodes: int
        Number of paper nodes.
    author_num_nodes: int
        Number of author nodes.
    institution_num_nodes: int
        Number of institution nodes.
    domain_num_nodes: int
        Number of domain nodes.
    hidden_channels: int
        Number of hidden features.
    dropout: float
        Dropout probability.

    Attributes
    ----------
    paper_embedding: nn.Embedding
        Paper embedding layer.
    author_embedding: nn.Embedding
        Author embedding layer.
    domain_embedding: nn.Embedding
        Domain embedding layer.
    institution_embedding: nn.Embedding
        Institution embedding layer.
    gnn: nn.Module
        Graph neural network model.
    classifier: LinkClassifier
        Link classifier model.

    Examples
    --------
    >>> import torch
    >>> from torch_geometric.data import HeteroData
    >>> from halvesting_geometric.models import LinkPrediction
    >>> metadata = ...
    >>> model = LinkPrediction("sage", metadata, 100, 100, 100, 100, 128, 0.5)
    >>> batch = HeteroData(...)
    >>> output = model(batch)
    """

    def __init__(
        self,
        gnn: Literal["sage", "gat", "rggc"],
        metadata: Metadata,
        paper_num_nodes: int,
        author_num_nodes: int,
        institution_num_nodes: int,
        domain_num_nodes: int,
        hidden_channels: int,
        dropout: float,
    ):
        super().__init__()
        self.paper_embedding = nn.Embedding(paper_num_nodes, hidden_channels)
        self.author_embedding = nn.Embedding(author_num_nodes, hidden_channels)
        self.domain_embedding = nn.Embedding(domain_num_nodes, hidden_channels)
        self.institution_embedding = nn.Embedding(
            institution_num_nodes, hidden_channels
        )
        # Instantiate homogeneous GNN:
        self.gnn = _GNN_MAP[gnn](
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            dropout=dropout,
        )
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=metadata)
        self.classifier = LinkClassifier()

    def forward(self, batch: HeteroData):
        """Forward pass.

        Parameters
        ----------
        batch : HeteroData
            A batch of data.

        Returns
        -------
        pred : Tensor
            Predictions.
        """
        x_dict = {
            "author": self.author_embedding(batch["author"].n_id),
            "institution": self.institution_embedding(batch["institution"].n_id),
            "domain": self.domain_embedding(batch["domain"].n_id),
            "paper": self.paper_embedding(batch["paper"].n_id),
        }
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, batch.edge_index_dict)
        pred = self.classifier(
            x_author=x_dict["author"],
            x_paper=x_dict["paper"],
            edge_label_index=batch["author", "writes", "paper"].edge_label_index,
        )
        return pred
