# halph/models/rggc.py

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ResGatedGraphConv


class RGGC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float,
        **kwargs
    ):
        super().__init__()
        self.dropout = dropout
        self.conv1 = ResGatedGraphConv(in_channels, hidden_channels)
        self.conv2 = ResGatedGraphConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)
        x = self.conv2(x, edge_index, edge_weight)
        return x
