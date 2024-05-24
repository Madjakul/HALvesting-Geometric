# halph/utils/data/link_prediction_dataset.py

import os.path as osp
from typing import Callable, List, Optional

import torch
from torch_geometric.data import HeteroData, InMemoryDataset
from torch_geometric.utils import coalesce


class LinkPredictionDataset(InMemoryDataset):
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

        # Get paper nodes, fatures and labels
        path = osp.join(self.raw_dir, "nodes", "id_paper.csv.gz")
        df = pd.read_csv(
            path,
            sep="\t",
            compression="gzip",
            names=["paper_id", "halid", "year", "name"],
            index_col=0,
        )
        data["paper"].x = torch.from_numpy(df[["year", "halid"]].values)
        data["paper"].num_features = 2
        data["paper"].index = torch.from_numpy(df.index.values)
        data["paper"].num_nodes = df.shape[0]

        # Get author nodes
        path = osp.join(self.raw_dir, "nodes", "halauthorid_author.csv.gz")
        df = pd.read_csv(
            path,
            sep="\t",
            names=["idx", "halauthorid", "name"],
            index_col=0,
            compression="gzip",
        )
        data["author"].index = torch.from_numpy(df.index.values)
        data["author"].num_nodes = df.shape[0]
        data["author"].num_features = 0

        # Get institution nodes
        path = osp.join(self.raw_dir, "nodes", "id_institution.csv.gz")
        df = pd.read_csv(
            path,
            sep="\t",
            names=["idx", "institution"],
            index_col=0,
            compression="gzip",
        )
        data["institution"].index = torch.from_numpy(df.index.values)
        data["institution"].num_nodes = df.shape[0]
        data["institution"].num_features = 0

        # Get domain nodes
        path = osp.join(self.raw_dir, "nodes", "domains.csv.gz")
        df = pd.read_csv(
            path,
            sep="\t",
            names=["idx", "domain"],
            index_col=0,
            compression="gzip",
        )
        data["domain"].index = torch.from_numpy(df.index.values)
        data["domain"].num_nodes = df.shape[0]
        data["domain"].num_features = 0

        edge_dir_path = osp.join(self.raw_dir, "edges")

        # Get author <-> paper edges
        path = osp.join(edge_dir_path, "author__writes__paper.csv.gz")
        df = pd.read_csv(path, sep="\t", names=["author", "paper"], compression="gzip")
        df = df.drop_duplicates(keep=False)
        df = torch.from_numpy(df.values)
        df = df.t().contiguous()
        M, N = int(df[0].max() + 1), int(df[1].max() + 1)
        df = coalesce(df, num_nodes=max(M, N))
        data["paper", "writes", "paper"].edge_index = df

        # Get author <-> institution edges
        path = osp.join(edge_dir_path, "author__affiliated_with__institution.csv.gz")
        df = pd.read_csv(
            path, sep="\t", names=["author", "institution"], compression="gzip"
        )
        df = df.drop_duplicates(keep=False)
        df = torch.from_numpy(df.values)
        df = df.t().contiguous()
        M, N = int(df[0].max() + 1), int(df[1].max() + 1)
        df = coalesce(df, num_nodes=max(M, N))
        data["author", "affiliated_with", "institution"].edge_index = df

        # Get paper <-> paper edges
        path = osp.join(edge_dir_path, "paper__cites__paper.csv.gz")
        df = pd.read_csv(path, sep="\t", names=["paper", "c_paper"], compression="gzip")
        df = df.drop_duplicates(keep=False)
        df = torch.from_numpy(df.values)
        df = df.t().contiguous()
        M, N = int(df[0].max() + 1), int(df[1].max() + 1)
        df = coalesce(df, num_nodes=max(M, N))
        data["paper", "cites", "paper"].edge_index = df

        # Get paper <-> domain edges
        path = osp.join(edge_dir_path, "paper__has_topic__domain.csv.gz")
        df = pd.read_csv(path, sep="\t", names=["paper", "domain"], compression="gzip")
        df = df.drop_duplicates(keep=False)
        df = torch.from_numpy(df.values)
        df = df.t().contiguous()
        M, N = int(df[0].max() + 1), int(df[1].max() + 1)
        df = coalesce(df, num_nodes=max(M, N))
        data["paper", "has_topic", "domain"].edge_index = df

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
