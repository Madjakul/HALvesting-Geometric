# halph/utils/helpers.py

import argparse
import json
import logging
import os
from typing import List, Union

WIDTH = 139
PROJECT_ROOT = os.getcwd()
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_ROOT, exist_ok=True)


def boolean(argument: Union[str, int, bool]):
    """Custom boolean type to parse bollean arguments.

    Parameters
    ----------
    argument: Union[str, int, bool]
        Argument to parse.

    Returns
    -------
    boolean
        Parsed argument.
    """
    if isinstance(argument, bool):
        return argument
    elif isinstance(argument, str):
        if argument.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif argument.lower() in ("no", "false", "f", "n", "0"):
            return False
    elif isinstance(argument, int):
        if argument == 1:
            return True
        else:
            return False
    else:
        raise argparse.ArgumentTypeError("Boolean argumentalue expected.")


def check_dir(path: str):
    """Check if there is a directory at ``path`` and creates it if necessary.

    Parameters
    ----------
    path: str
        Path to the directory.

    Returns
    -------
    path: str
        Path to the existing directory.
    """
    if os.path.isdir(path):
        return path
    logging.warning(f"No folder at {path}: creating folders at path.")
    os.makedirs(path)
    return path


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as jsf:
        js = json.load(jsf)
    return js


def read_jsons(paths: List[str]):
    js = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as jsf:
            js.update(json.load(jsf))
    return js


def transpose_json(path: str, on: str):
    with open(path, "r", encoding="utf-8") as jsf:
        json_headers = json.load(jsf)
