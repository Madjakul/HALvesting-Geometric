# halph/models/classifier.py

import torch.nn as nn
from torch import Tensor


class LinkClassifier(nn.Module):
    def forward(
        self, x_author: Tensor, x_paper: Tensor, edge_label_index: Tensor
    ) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_author = x_author[edge_label_index[0]]
        edge_feat_paper = x_paper[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_author * edge_feat_paper).sum(dim=-1)