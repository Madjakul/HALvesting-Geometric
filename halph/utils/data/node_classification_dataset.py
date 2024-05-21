# halph/utils/data/node_classification_dataset.py

import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import HeteroData, InMemoryDataset


class NodeClassificationDataset(InMemoryDataset):
    """Class.

    Parameters
    ----------

    Attributes
    ----------

    Examples
    --------
    """

    def __init__(
        self,
        root: str,
        preprocess: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        preprocess = None if preprocess is None else preprocess.lower()
        self.preprocess = preprocess
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    def __repr__(self) -> str:
        return "halph()"

    @property
    def num_classes(self) -> int:
        assert isinstance(self._data, HeteroData)
        return int(self._data["paper"].y.max()) + 1

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "processed")

    @property
    def raw_file_names(self) -> List[str]:
        file_names = ["nodes", "edges"]

        if self.preprocess is not None:
            file_names += [f"halph_{self.preprocess}_emb.pt"]

        return file_names

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def process(self) -> None:
        import pandas as pd

        data = HeteroData()

        # Get paper labels
        path = osp.join(self.raw_dir, "nodes", "id_paper.csv.gz")
        paper = pd.read_csv(
            path,
            sep="\t",
            compression="gzip",
            names=["paper_id", "halid", "year", "name"],
            index_col=0,
        )

        path = osp.join(self.raw_dir, "nodes", "labels", "paper__domain.csv.gz")
        df = pd.read_csv(path, sep="\t", names=["idx", "paper_id", "y"], index_col=0)
        df = df.join(paper, on="paper_id")

        data["paper"].year = torch.from_numpy(df["year"].values)
        data["paper"].halid = torch.from_numpy(df["halid"].values)
        data["paper"].y = torch.from_numpy(df["y"].values)
        data["paper"].y_index = torch.from_numpy(df.index.values)

        # Get edges
        for edge_type in [
            ("author", "affiliated_with", "institution"),
            ("author", "writes", "paper"),
            ("paper", "cites", "paper"),
        ]:
            f = "__".join(edge_type)
            path = osp.join(self.raw_dir, "edges", f"{f}.csv.gz")
            edge_index = pd.read_csv(
                path, sep="\t", compression="gzip", header=None, dtype=np.int64
            )
            edge_index = edge_index.drop_duplicates(keep=False).values
            edge_index = torch.from_numpy(edge_index).t().contiguous()
            data[edge_type].edge_index = edge_index

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
