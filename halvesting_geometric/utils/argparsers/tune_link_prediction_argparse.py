# halvesting_geometric/utils/argparsers/link_prediction_argparse.py

import argparse

from halvesting_geometric.utils.helpers import boolean


class TuneLinkPredictionArgparse:
    """Argument parser used to perfom link prediction."""

    @classmethod
    def parse_known_args(cls):
        """Parses arguments.

        Returns
        -------
        args: Any
            Parsed arguments.
        """
        parser = argparse.ArgumentParser(
            description="Arguments for link prediction tuning."
        )
        parser.add_argument(
            "--root_dir",
            type=str,
            required=True,
            help="Path to the data root directory.",
        )
        parser.add_argument(
            "--storage_path",
            type=str,
            required=True,
            help="Path to the directory where the tuning logs are stored.",
        )
        parser.add_argument(
            "--lang_",
            type=str,
            default="all",
            help="Language to use for the dataset.",
        )
        parser.add_argument(
            "--accelerator",
            type=str,
            default=None,
            help="Accelerator to use for training.",
        )
        parser.add_argument(
            "--num_proc",
            type=int,
            default=None,
            help="Number of processes to use.",
        )
        args, _ = parser.parse_known_args()
        return args
