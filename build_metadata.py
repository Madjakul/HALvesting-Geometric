# build_metadata.py

import gc
import logging
import os
import os.path as osp

import pandas as pd

from halph.utils import helpers, logging_config
from halph.utils.argparsers import BuildMetadataArgparse
from halph.utils.data import LinkPredictionMetadata

LANGS = [
    "en",
    "fr",
    "es",
    "it",
    "pt",
    "de",
    "ru",
    "eu",
    "pl",
    "el",
    "ro",
    "ca",
    "da",
    "br",
    "hu",
    "cs",
]

logging_config()

if __name__ == "__main__":
    args = BuildMetadataArgparse.parse_known_args()

    json_file_names = os.listdir(args.json_dir)
    json_file_paths = [
        osp.join(args.json_dir, json_file_name)
        for json_file_name in json_file_names
        if json_file_name.endswith(".json")
    ]

    df = helpers.pd_read_jsons(json_file_paths)
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
    logging.info(df)
    del df_
    gc.collect()

    metadata = LinkPredictionMetadata(
        root_dir=args.root_dir,
        json_dir=args.json_dir,
        xml_dir=args.xml_dir,
    )
    if args.compute_nodes:
        metadata.compute_nodes(df, lang=LANGS)
    if args.compute_edges:
        metadata.compute_edges(df)
