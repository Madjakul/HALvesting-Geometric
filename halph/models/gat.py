# halph/models/modeling_gat.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float,
        heads: int = 8,
        **kwargs
    ):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            add_self_loops=False,
        )
        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=heads,
            concat=False,
            dropout=dropout,
            add_self_loops=False,
        )

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x
