# halph/utils/data/node_classification_metadata.py

import logging
import multiprocessing
from typing import Any, Dict

from datasets import DatasetDict
from lxml import etree

from halph.utils.data.metadata import Metadata


class NodeClassificationMetadata(Metadata):
    """This class builds the raw documents used to creates graphs out of HAL
    that can be loaded by **torch_geometric**.

    Parameters
    ----------

    Attributes
    ----------

    Examples
    --------
    """

    def __init__(
        self,
        dataset: DatasetDict,
        json_dir_path: str,
        xml_dir_path: str,
        num_proc: int,
    ):
        super().__init__(
            template="node_classification",
            json_dir_path=json_dir_path,
            xml_dir_path=xml_dir_path,
        )
        self.num_proc = num_proc
        self.dataset = dataset

    def __call__(self, output_dir_path: str):
        pass

    def _worker(
        self, q: multiprocessing.Manager.Queue, header: Dict[str, Dict[str, Any]]
    ):
        # Process XML file with the header.
        pass

    def _listener(self, q: multiprocessing.Manager.Queue, ouput_dir_path: str):
        while True:
            if q is None:
                break

    def build(self, output_dir_path: str):
        manager = multiprocessing.Manager()
        q = manager.Queue()
        file_pool = multiprocessing.Pool(1)
        file_pool.apply_async(self._listener, (q, output_dir_path))

        pool = multiprocessing.Pool(self.num_proc)
        jobs = []
        for header in self.headers:
            job = pool.apply_async(self._worker, (q, header))
            jobs.append(job)

        for job in jobs:
            job.get()

        q.put(None)
        pool.close()
        pool.join()
