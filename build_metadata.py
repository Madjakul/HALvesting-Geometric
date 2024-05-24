# build_metadata.py

import datasets

from halph.utils import logging_config
from halph.utils.data import LinkPredictionMetadata, NodeClassificationMetadata

logging_config()

if __name__ == "__main__":
    dataset = datasets.load_dataset("Madjakul/HALvest", "fr", split="train")
    metadata = NodeClassificationMetadata(
        dataset=dataset,
        json_dir_path="./data/mock/responses",
        xml_dir_path="./data/mock/grobid_out",
    )
    metadata("./data/mock/raw")
