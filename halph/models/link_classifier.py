# halph/models/classifier.py

import torch.nn as nn
from torch import Tensor


class LinkClassifier(nn.Module):
    def forward(
        self, x_user: Tensor, x_movie: Tensor, edge_label_index: Tensor
    ) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)
