# build_metadata.py

import datasets

from halph.utils.data import NodeClassificationMetadata

if __name__ == "__main__":
    dataset = datasets.load_dataset("Madjakul/HALvest", "fr", split="train")
    metadata = NodeClassificationMetadata(
        dataset=dataset,
        json_dir_path="./data/mock/responses",
        xml_dir_path="./data/mock/grobid_out",
        num_proc=3,
    )
    metadata("./data/raw")
