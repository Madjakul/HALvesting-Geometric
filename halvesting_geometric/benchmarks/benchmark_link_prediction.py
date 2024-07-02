# halvesting_geometric/benchmarks/benchmark_link_prediction.py

from typing import Literal

import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from halvesting_geometric.models import LinkPrediction


class BenchmarkLinkPrediction:
    """Link prediction benchmark.

    This benchmark evaluates the area under the ROC curve (AUC) of a
    link prediction model.
    """

    @classmethod
    @torch.no_grad()
    def run(
        cls,
        model: LinkPrediction,
        test_dataloader: NeighborLoader,
        device: Literal["cpu", "cuda"],
    ):
        """Run the link prediction benchmark.

        Parameters
        ----------
        model : LinkPrediction
            Link prediction model.
        test_dataloader : NeighborLoader
            Test dataloader.
        device : Literal['cpu', 'cuda']
            Device to use.

        Returns
        -------
        auc : float
            Area under the ROC curve (AUC).
        """
        model.eval()
        y_true = []
        y_pred = []
        for batch in tqdm(test_dataloader):
            batch = batch.to(device)
            y_true.append(batch["author", "writes", "paper"].edge_label)
            y_pred.append(model(batch))
        y_true = torch.cat(y_true, dim=0).cpu().numpy()
        y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
        auc = roc_auc_score(y_true, y_pred)
        return auc
