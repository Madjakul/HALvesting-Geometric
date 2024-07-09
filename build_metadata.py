# build_metadata.py

import gc
import logging
import os
import os.path as osp

import dask
import datasets
import pandas as pd
import psutil
from dask.dataframe import from_pandas  # type: ignore
from tqdm import tqdm

from halvesting_geometric.utils import helpers, logging_config
from halvesting_geometric.utils.argparsers import BuildMetadataArgparse
from halvesting_geometric.utils.data import LinkPredictionMetadata
from halvesting_geometric.utils.helpers import WIDTH

# Number of physical CPUs
NUM_PROC = psutil.cpu_count(logical=False)


logging_config()


if __name__ == "__main__":
    args = BuildMetadataArgparse.parse_known_args()

    logging.info(f"{('=' * WIDTH)}")
    logging.info(f"Computing HAL's graph".center(WIDTH))
    logging.info(f"{('=' * WIDTH)}")

    dask.config.config["dataframe"]["convert-string"] = False  # type: ignore

    halids = []
    # Get `halid` from clean dataset
    with open(args.dataset_config_path, "r") as f:
        configs = f.read().splitlines()
    for config in tqdm(configs):
        dataset = datasets.load_dataset(
            args.dataset_checkpoint,
            config,
            split="train",
            cache_dir=args.cache_dir,
        )
        halids.extend(dataset["halid"])  # type: ignore

    # Get stored responses from HAL API
    json_file_names = os.listdir(args.json_dir)
    json_file_paths = [
        osp.join(args.json_dir, json_file_name)
        for json_file_name in json_file_names
        if json_file_name.endswith(".json")
    ]

    # Transform responses into dataframe
    df = helpers.pd_read_jsons(json_file_paths, lines=False)
    df = df.explode("authors").reset_index(drop=True)
    logging.info("Normalizing authors...")
    df_ = pd.json_normalize(df["authors"].to_list()).reset_index(drop=True)
    df = pd.concat(
        [
            df[["halid", "title", "lang", "year", "domain"]],
            df_[["name", "halauthorid", "affiliations"]],
        ],
        axis=1,
    )

    del df_
    gc.collect()

    # Transform dataframe into dask dataframe
    ddf = from_pandas(df, npartitions=NUM_PROC)
    logging.info(ddf)

    metadata = LinkPredictionMetadata(
        root_dir=args.root_dir,
        json_dir=args.json_dir,
        xml_dir=args.xml_dir,
        halids=halids,
        raw_dir=args.raw_dir,
    )
    if args.compute_nodes:
        metadata.compute_nodes(ddf, langs=configs)
    if args.compute_edges:
        metadata.compute_edges(ddf, num_proc=NUM_PROC)

    if args.zip_compress:
        helpers.zip_compress(args.raw_dir)
