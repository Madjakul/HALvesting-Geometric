# halph/models/modeling_bbp_link_prediction.py

from typing import Literal

import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero
from torch_geometric.typing import Metadata

from halph.models.bbp_link_classifier import BigBirdPegasusLinkClassifier
from halph.models.gat import GAT
from halph.models.rggc import RGGC
from halph.models.sage import GraphSage
from halph.utils.data import BigBirdPegasusDataset

_GNN_MAP = {"sage": GraphSage, "gat": GAT, "rggc": RGGC}


class BigBirdPegasusLinkPrediction(nn.Module):
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
        bbp_dataset: BigBirdPegasusDataset,
    ):
        super().__init__()
        self.bbp_dataset = bbp_dataset
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
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
        self.classifier = BigBirdPegasusLinkClassifier(hidden_channels, dropout)

    def forward(self, batch: HeteroData) -> Tensor:
        inputs = self.bbp_dataset.get(
            batch["paper"].n_id[batch["author", "writes", "paper"].edge_label_index[1]]
        )
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
            inputs=inputs,
        )
        return pred
