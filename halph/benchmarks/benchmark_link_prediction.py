# halph/benchmarks/benchmark_link_prediction.py

import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from halph.models import LinkPrediction


class BenchMarkLinkPrediction:

    @classmethod
    @torch.no_grad()
    def run(cls, model: LinkPrediction, test_dataloader: NeighborLoader, device: str):
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
