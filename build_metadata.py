# build_metadata.py

import gc
import logging
import os
import os.path as osp

import pandas as pd

from halph.utils import helpers, logging_config
from halph.utils.data import LinkPredictionMetadata, NodeClassificationMetadata

logging_config()

if __name__ == "__main__":
    # jsonl_dir = "./data/jsons"
    json_dir = "./data/responses"
    json_file_names = os.listdir(json_dir)
    json_file_paths = [
        osp.join(json_dir, json_file_name)
        for json_file_name in json_file_names
        if json_file_name.endswith(".json")
    ]
    # jsonl_file_paths = [
    #     osp.join(jsonl_dir, json_file_name)
    #     for json_file_name in json_file_names
    #     if json_file_name.endswith(".json")
    # ]
    # df = helpers.pd_read_jsons(json_file_paths)
    # print(df.tail())
    # helpers.jsons_to_jsonls(input_paths=json_file_paths, output_paths=jsonl_file_paths)
    # dataset = datasets.load_dataset(
    #     "Madjakul/HALvest", "en", split="train"
    # ).remove_columns(["text", "url", "year", "domain", "lang"])

    # df = helpers.pd_read_jsons(json_file_paths)
    # df = df.explode("authors").reset_index(drop=True)
    # logging.info("Normalizing authors...")
    # df_ = pd.json_normalize(df["authors"]).reset_index(drop=True)
    # df = pd.concat(
    #     [
    #         df[["halid", "title", "lang", "year", "domain"]],
    #         df_[["name", "halauthorid", "affiliations"]],
    #     ],
    #     axis=1,
    # )
    # logging.info(df)
    # del df_
    # gc.collect()

    # df = df.explode("authors").reset_index(drop=True)
    # logging.info("Normalizing authors...")
    # df_ = pd.json_normalize(df["authors"]).reset_index(drop=True)
    # df = pd.concat([df, df_], axis=1)

    df = None

    metadata = LinkPredictionMetadata(
        root_dir="./data",
        json_dir="./data/responses",
        xml_dir="./data/output_tei_xml",
        num_proc=4,
    )
    # metadata.compute_nodes(df, lang="en")
    metadata.compute_edges(df)
    # metadata("./data/mock/raw")
