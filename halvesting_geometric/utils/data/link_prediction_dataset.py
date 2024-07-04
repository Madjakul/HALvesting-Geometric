# halvesting_geometric/utils/data/link_prediction_dataset.py

import os.path as osp
from typing import Callable, List, Optional

import torch
from torch_geometric.data import HeteroData, InMemoryDataset
from torch_geometric.utils import coalesce


class LinkPredictionDataset(InMemoryDataset):
    """Base class for link prediction datasets.

    Parameters
    ----------
    root : str
        Directory where the dataset should be stored.
    preprocess : str, optional
        Preprocessing method to apply to the dataset.
    transform : Callable, optional
        A function/transform that takes in an HeteroData object and returns a transformed version.
    pre_transform : Callable, optional
        A function/transform that takes in an HeteroData object and returns a transformed version.
    force_reload : bool, optional
        If set to :obj:`True`, the dataset will be re-downloaded and preprocessed, even if it already exists on disk.
    lang : str, optional
        Language of the dataset. Can be :obj:`"en"`, :obj:`"fr"` or :obj:`"all"`.

    Attributes
    ----------
    url : str
        URL to download the dataset.
    preprocess : str
        Preprocessing method to apply to the dataset.

    Examples
    --------
    >>> TODO
    """

    urls = {"en": "", "fr": "", "default": ""}

    def __init__(
        self,
        root: str,
        preprocess: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
        lang: Optional[str] = None,
    ):
        if lang is None:
            self.url = self.urls["default"]
        else:
            assert lang in ["en", "fr", "all"]
            self.url = self.urls[lang]
        preprocess = None if preprocess is None else preprocess.lower()
        self.preprocess = preprocess
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    def __repr__(self) -> str:
        return "halvest_geometric()"

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

    # TODO: Modify this method to download the data from HuggingFace Datasets
    def download(self) -> None:
        raise NotImplementedError(
            "Download from HuggingFace Datasets is not implemented yet."
        )

    # TODO: Modify this method to download the data from HuggingFace Datasets
    def process(self) -> None:
        import pandas as pd

        data = HeteroData()

        # Get paper nodes
        path = osp.join(self.raw_dir, "nodes", "papers.csv.gz")
        df = pd.read_csv(
            path,
            sep="\t",
            compression="gzip",
            dtype={
                "halid": str,
                "year": int,
                "title": str,
                "lang": str,
                "paper_idx": int,
            },
        )
        data["paper"].num_features = 0
        data["paper"].index = torch.from_numpy(df.index.values)
        data["paper"].num_nodes = df.shape[0]

        # Get author nodes
        path = osp.join(self.raw_dir, "nodes", "authors.csv.gz")
        df = pd.read_csv(
            path,
            sep="\t",
            compression="gzip",
            dtype={"name": str, "halauthorid": int, "author_idx": int},
        )
        data["author"].index = torch.from_numpy(df.index.values)
        data["author"].num_nodes = df.shape[0]
        data["author"].num_features = 0

        # Get affiliation nodes
        path = osp.join(self.raw_dir, "nodes", "affiliations.csv.gz")
        df = pd.read_csv(
            path,
            sep="\t",
            compression="gzip",
            dtype={"affiliations": str, "affiliation_idx": int},
        )
        data["affiliation"].index = torch.from_numpy(df.index.values)
        data["affiliation"].num_nodes = df.shape[0]
        data["affiliation"].num_features = 0

        # Get domain nodes
        path = osp.join(self.raw_dir, "nodes", "domains.csv.gz")
        df = pd.read_csv(
            path,
            sep="\t",
            compression="gzip",
            dtype={"domain": int, "domain_idx": int},
        )
        data["domain"].index = torch.from_numpy(df.index.values)
        data["domain"].num_nodes = df.shape[0]
        data["domain"].num_features = 0

        edge_dir_path = osp.join(self.raw_dir, "edges")

        # Get author <-> paper edges
        path = osp.join(edge_dir_path, "author__writes__paper.csv.gz")
        df = pd.read_csv(
            path,
            sep="\t",
            compression="gzip",
            dtype={"author_idx": int, "paper_idx": int},
        )
        df = torch.from_numpy(df.values)
        df = df.t().contiguous()
        M, N = int(df[0].max() + 1), int(df[1].max() + 1)
        df = coalesce(df, num_nodes=max(M, N))
        data["author", "writes", "paper"].edge_index = df

        # Get author <-> affiliation edges
        path = osp.join(edge_dir_path, "author__affiliated_with__affiliation.csv.gz")
        df = pd.read_csv(
            path,
            sep="\t",
            compression="gzip",
            dtype={"author_idx": int, "affiliation_idx": int},
        )
        df = torch.from_numpy(df.values)
        df = df.t().contiguous()
        M, N = int(df[0].max() + 1), int(df[1].max() + 1)
        df = coalesce(df, num_nodes=max(M, N))
        data["author", "affiliated_with", "affiliation"].edge_index = df

        # Get paper <-> paper edges
        path = osp.join(edge_dir_path, "paper__cites__paper.csv.gz")
        df = pd.read_csv(
            path,
            sep="\t",
            compression="gzip",
            dtype={"paper_idx": int, "c_paper_idx": int},
        )
        df = torch.from_numpy(df.values)
        df = df.t().contiguous()
        M, N = int(df[0].max() + 1), int(df[1].max() + 1)
        df = coalesce(df, num_nodes=max(M, N))
        data["paper", "cites", "paper"].edge_index = df

        # Get paper <-> domain edges
        path = osp.join(edge_dir_path, "paper__has_topic__domain.csv.gz")
        df = pd.read_csv(
            path,
            sep="\t",
            compression="gzip",
            dtype={"paper_idx": int, "domain_idx": int},
        )
        df = torch.from_numpy(df.values)
        df = df.t().contiguous()
        M, N = int(df[0].max() + 1), int(df[1].max() + 1)
        df = coalesce(df, num_nodes=max(M, N))
        data["paper", "has_topic", "domain"].edge_index = df

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
