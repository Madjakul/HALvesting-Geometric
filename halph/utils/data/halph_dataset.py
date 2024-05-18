# halph/utils/data/halph_dataset.py

import os
import shutil
from typing import Callable, List, Literal, Optional

import numpy as np
import torch
from torch_geometric.data import (HeteroData, InMemoryDataset, download_url,
                                  extract_zip)
from torch_geometric.io import fs


class HALphDataset(InMemoryDataset):
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
        template: Literal["node_classification", "link_prediction"],
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
