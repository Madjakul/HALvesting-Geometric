# halph/utils/data/metadata_builder.py

import json
import logging
import multiprocessing
import os
from typing import Any, Dict, List, Optional, Union

from datasets import DatasetDict
from lxml import etree
from tqdm import tqdm

from halph.utils import helpers


class MetadataBuilder:
    """This class builds the raw documents used to creates graphs out of HAL
    that can be loaded by **torch_geometric**.

    Parameters
    ----------

    Attributes
    ----------

    Examples
    --------
    """

    headers: Dict[str, Dict[str, Any]]

    def __init__(
        self,
        template: str,
        dataset: DatasetDict,
        json_dir_path: str,
        xml_dir_path: str,
        filtering_batch_size: int,
        num_proc: int,
    ):
        self.template = template
        self.num_proc = num_proc
        self.xml_dir_path = xml_dir_path
        self.xml_files = os.path.join(xml_dir_path)
        self.xml_file_paths = [
            os.path.join(xml_dir_path, xml_file) for xml_file in self.xml_files
        ]
        json_files = os.listdir(json_dir_path)
        json_file_paths = [
            os.path.join(json_dir_path, json_file_path) for json_file_path in json_files
        ]
        headers = helpers.jsons_to_dict(json_file_paths, on="halid")
        logging.info("Filtering the headers...")
        self.headers = self._filter_headers(headers)
        logging.info("Filtering the dataset...")
        self.dataset = dataset.filter(
            lambda batch: self._filter_dataset(batch),
            batched=True,
            batch_size=filtering_batch_size,
            num_proc=num_proc,
        )

    def __call__(self, output_dir_path: str):
        pass

    def _filter_headers(self, headers: Dict[str, Dict[str, Any]]):
        keys = list(self.headers.keys())
        for key in tqdm(keys):
            if f"{key}.grobid.tei.xml" in self.xml_files:
                continue
            del headers[key]
        return headers

    def _filter_dataset(self, batch: Dict[str, List[Any]]):
        mask = []
        for halid in batch["halid"]:
            if halid in self.headers:
                mask.append(True)
            else:
                mask.append(False)
        return mask

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
