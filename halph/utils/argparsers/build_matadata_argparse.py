# halph/utils/argparsers/build_matadata_argparse.py

import argparse

from halph.utils.helpers import boolean


class BuildMetadataArgparse:
    """Argument parser used to build the graphs' metadata."""

    @classmethod
    def parse_known_args(cls):
        """Parses arguments.

        Returns
        -------
        args: Any
            Parsed arguments.
        """
        parser = argparse.ArgumentParser(
            description="Argument parser used to build the graphs' metadata."
        )
        parser.add_argument(
            "--dataset_checkpoint",
            type=str,
            help="Name of the HuggingFace dataset containing the filtered documents.",
        )
        parser.add_argument(
            "--cache_dir_path",
            type=str,
            nargs="?",
            const=None,
            help="Path to the HuggingFace cache directory.",
        )
        parser.add_argument(
            "--dataset_config_path",
            type=str,
            default=None,
            help="Path to the txt file containing the dataset configs to process.",
        )
        parser.add_argument(
            "--output_dir_path",
            type=str,
            help="Path to the directory where the processed dataset will be saved.",
        )
        parser.add_argument(
            "--tokenizer_checkpoint",
            type=str,
            nargs="?",
            const=None,
            help="Name of the HuggingFace tokenizer model to be used.",
        )
        parser.add_argument(
            "--use_fast",
            type=boolean,
            nargs="?",
            const=False,
            help="Set to `true` if you want to use the Ruste-based tokenizer from HF.",
        )
        parser.add_argument(
            "--template",
            type=str,
            required=True,
            default="default",
            help="Graph template. One of {'default', 'mag', 'halph-enriched', 'halph'}.",
        )
        args, _ = parser.parse_known_args()
        return args
