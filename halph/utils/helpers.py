# halph/utils/helpers.py

import argparse
import gzip
import json
import logging
import os
from typing import List, Union

from lxml import etree

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
    """Reads a JSON file at path.

    Parameters
    ----------

    Returns
    -------
    js: List[Dict[str, Union[str, List[str]]]]
    """
    with open(path, "r", encoding="utf-8") as jsf:
        js = json.load(jsf)
    return js


def read_jsons(paths: List[str]):
    """Reads a JSON files in the directory at path.

    Parameters
    ----------

    Returns
    -------
    js: List[Dict[str, Union[str, List[str]]]]
    """
    js = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as jsf:
            js.extend(json.load(jsf))
    return js


def json_to_dict(path: str, on: str):
    data = {}
    with open(path, "r", encoding="utf-8") as jsf:
        json_headers = json.load(jsf)
    for json_header in json_headers:
        tmp_data = {}
        tmp_data[json_header[on]] = {k: v for k, v in json_header.items() if k != on}
        data.update(tmp_data)
    return data


def jsons_to_dict(paths: List[str], on: str):
    data = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as jsf:
            json_headers = json.load(jsf)
        for json_header in json_headers:
            tmp_data = {}
            tmp_data[json_header[on]] = {
                k: v for k, v in json_header.items() if k != on
            }
            data.update(tmp_data)
    return data


def str_to_xml(xml: str):
    """Function.

    Parameters
    ----------

    Returns
    -------
    root: etree.ElementTree
        Clean XML tree.
    """
    root = etree.fromstring(xml.encode("utf-8"))
    for elem in root.getiterator():
        # Skip comments and processing instructions,
        # because they do not have names
        if not (
            isinstance(elem, etree._Comment)
            or isinstance(elem, etree._ProcessingInstruction)
        ):
            # Remove a namespace URI in the element's name
            elem.tag = etree.QName(elem).localname
    # Remove unused namespace declarations
    etree.cleanup_namespaces(root)
    return root


def gzip_compress(path: str):
    with open(path, "rb") as f:
        with gzip.open(f"{path}.gz", "wb") as gzf:
            gzf.writelines(f)
