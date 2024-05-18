# halph/utils/data/link_prediction_metadata.py

import multiprocessing
from typing import Any, Dict

from datasets import DatasetDict

from halph.utils.data.metadata import Metadata


class LinkPredictionMetadata(Metadata):
    """Class.

    Parameters
    ----------

    Attributes
    ----------
    """

    def __init__(self, dataset: DatasetDict, json_dir_path: str, xml_dir_path: str):
        super().__init__(
            template="link_prediction",
            json_dir_path=json_dir_path,
            xml_dir_path=xml_dir_path,
        )

    def _worker(self, q: multiprocessing.Queue, document: Dict[str, Any], path: str):
        pass

    def _listener(self, q: multiprocessing.Queue, output_dir_path: str):
        pass

    def build(self, output_dir_path: str):
        pass

    def __call__(self, output_dir_path: str):
        pass
