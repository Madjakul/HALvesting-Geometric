# halvesting_geometric/modules/modeling_gat.py

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """Graph Attention Network model.

    Parameters
    ----------
    in_channels: int
        Number of input features.
    hidden_channels: int
        Number of hidden features.
    out_channels: int
        Number of output features.
    dropout: float
        Dropout probability.
    heads: int
        Number of attention heads.

    Attributes
    ----------
    dropout: float
        Dropout probability.
    conv1: GATConv
        First GAT convolution layer.
    conv2: GATConv
        Second GAT convolution layer.

    Examples
    --------
    >>> import torch
    >>> from halvesting_geometric.models import GAT
    >>> model = GAT(128, 128, 128, 0.5)
    >>> input = torch.randn(10, 128)
    >>> edge_index = torch.randint(0, 10, (2, 100))
    >>> output = model(input, edge_index)
    """

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

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features.
        edge_index : torch.Tensor
            Graph edge indices.
        edge_weight : Optional[torch.Tensor], optional
            Edge weights, by default None.

        Returns
        -------
        x : torch.Tensor
            Node embeddings.
        """
        x = self.conv1(x, edge_index, edge_weight)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        x = self.conv2(x, edge_index, edge_weight)
        return x
