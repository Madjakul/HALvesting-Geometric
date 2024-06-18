# build_metadata.py

import gc
import logging
import os
import os.path as osp

import datasets
import pandas as pd
import psutil
from dask.dataframe import from_pandas
from tqdm import tqdm

from halph.utils import helpers, logging_config
from halph.utils.argparsers import BuildMetadataArgparse
from halph.utils.data import LinkPredictionMetadata

CONFIGS = [
    # "en",
    # "fr",
    # "es",
    "it",
    # "pt",
    # "de",
    # "ru",
    # "eu",
    # "pl",
    # "el",
    # "ro",
    # "ca",
    # "da",
    # "br",
    # "ko",
    # "tr",
    # "hu",
    # "eo",
    # "fa",
    # "hy",
    # "cs",
    # "bg",
    # "id",
    # "he",
    # "hr",
    # "et",
    # "sv",
    # "no",
    # "fi",
    # "sw",
    # "gl",
    # "th",
    # "sl",
    # "sk",
]
NUM_PROC = psutil.cpu_count(logical=False)  # Number of physical CPUs

logging_config()

if __name__ == "__main__":
    args = BuildMetadataArgparse.parse_known_args()

    logging.info(f"num_proc: {NUM_PROC}")

    halids = []
    for config in tqdm(CONFIGS):
        dataset = datasets.load_dataset(
            "Madjakul/HALvest", config, split="train", cache_dir=args.cache_dir
        )
        halids.extend(dataset["halid"])

    json_file_names = os.listdir(args.json_dir)
    json_file_paths = [
        osp.join(args.json_dir, json_file_name)
        for json_file_name in json_file_names
        if json_file_name.endswith(".json")
    ]

    df = helpers.pd_read_jsons(json_file_paths, lines=False)
    df = df.explode("authors").reset_index(drop=True)
    logging.info("Normalizing authors...")
    df_ = pd.json_normalize(df["authors"]).reset_index(drop=True)
    df = pd.concat(
        [
            df[["halid", "title", "lang", "year", "domain"]],
            df_[["name", "halauthorid", "affiliations"]],
        ],
        axis=1,
    )

    del df_
    gc.collect()

    ddf = from_pandas(df, npartitions=NUM_PROC)
    logging.info(ddf)

    print(len(halids))
    print(halids[:30])

    metadata = LinkPredictionMetadata(
        root_dir=args.root_dir,
        json_dir=args.json_dir,
        xml_dir=args.xml_dir,
        halids=halids,
    )
    if args.compute_nodes:
        metadata.compute_nodes(ddf, lang=CONFIGS)
    if args.compute_edges:
        metadata.compute_edges(df)
