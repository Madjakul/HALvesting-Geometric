# halvesting_geometric/models/rggc.py

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ResGatedGraphConv


class RGGC(nn.Module):
    """Residual Gated Graph Convolution model.

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

    Attributes
    ----------
    dropout: float
        Dropout probability.
    conv1: ResGatedGraphConv
        First Residual Gated Graph Convolution layer.
    conv2: ResGatedGraphConv
        Second Residual Gated Graph Convolution layer.

    Examples
    --------
    >>> import torch
    >>> from halvesting_geometric.models import RGGC
    >>> model = RGGC(128, 128, 128, 0.5)
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
        **kwargs
    ):
        super().__init__()
        self.dropout = dropout
        self.conv1 = ResGatedGraphConv(in_channels, hidden_channels)
        self.conv2 = ResGatedGraphConv(hidden_channels, out_channels)

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
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)
        x = self.conv2(x, edge_index, edge_weight)
        return x
