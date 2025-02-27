# halvesting_geometric/utils/helpers.py

import gzip
import json
import logging
import os
import os.path as osp
import shutil
from typing import List, Union

import dask.dataframe as dd
import pandas as pd
import yaml
from lxml import etree
from lxml.etree import XMLParser
from tqdm import tqdm

WIDTH = 88
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


def load_config_from_file(path: str):
    """Load a configuration file from a path.

    Parameters
    ----------
    path: str
        Path to the configuration file.

    Returns
    -------
    config: Dict[str, Any]
        Configuration dictionary.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    load_config_map = {
        "json": load_config_from_json,
        "yaml": load_config_from_yaml,
        "yml": load_config_from_yaml,
    }
    assert os.path.isfile(path)
    _, file_extension = osp.splitext(path)
    file_extension = file_extension[1:]
    if not file_extension in load_config_map:
        raise ValueError(f"File extension {file_extension} not supported.")
    config = load_config_map[file_extension](path)
    return config


def load_config_from_json(path: str):
    """Load a JSON configuration file from a path.

    Parameters
    ----------
    path: str
        Path to the JSON configuration file.

    Returns
    -------
    config: Dict[str, Any]
        Configuration dictionary.
    """
    config = json.load(open(path))
    return config


def load_config_from_yaml(path: str):
    """Load a YAML configuration file from a path.

    Parameters
    ----------
    path: str
        Path to the YAML configuration file.

    Returns
    -------
    config: Dict[str, Any]
        Configuration dictionary.
    """
    config = yaml.load(open(path), Loader=yaml.FullLoader)
    return config


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
    """Convert a JSON file to a dictionary.

    Parameters
    ----------
    path : str
        Path to the JSON file.
    on : str
        Key to use as the main key in the dictionary.

    Returns
    -------
    data : dict
        Dictionary with the JSON data.
    """
    data = {}
    with open(path, "r", encoding="utf-8") as jsf:
        json_headers = json.load(jsf)
    for json_header in json_headers:
        tmp_data = {}
        tmp_data[json_header[on]] = {k: v for k, v in json_header.items() if k != on}
        data.update(tmp_data)
    return data


def jsons_to_dict(paths: List[str], on: str):
    """Convert a list of JSON files to a dictionary.

    Parameters
    ----------
    paths : List[str]
        List of paths to JSON files.
    on : str
        Key to use as the main key in the dictionary.

    Returns
    -------
    data : dict
        Dictionary with the JSON data.
    """
    logging.info("Converting JSONs to Dict...")
    data = {}
    for path in tqdm(paths):
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
    """Convert a string to an XML element.

    Parameters
    ----------
    xml : str
        String to convert to an XML element.

    Returns
    -------
    root : etree.Element
        XML element.
    """
    p = XMLParser(huge_tree=True)
    root = etree.fromstring(xml.encode("utf-8"), parser=p)
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
    etree.cleanup_namespaces(root)  # type: ignore
    return root


def gzip_compress(path: str):
    """Compress a file to a gzip file.

    Parameters
    ----------
    path : str
        Path to the file to compress.
    """
    with open(path, "rb") as f:
        with gzip.open(f"{path}.gz", "wb") as gzf:
            gzf.writelines(f)


def zip_compress(path: str):
    """Compress a file to a zip file.

    Parameters
    ----------
    path : str
        Path to the file to compress.
    """
    shutil.make_archive(path, "zip", path)


def compress_csv(df: pd.DataFrame, path: str):
    """Compress a pandas DataFrame to a gzip CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame to compress.
    path : str
        Path to save the compressed file.
    """
    df.to_csv(f"{path}.gz", compression="gzip", sep="\t", index=False, header=False)
    os.remove(path)


def jsons_to_jsonls(input_paths: List[str], output_paths: List[str]):
    """Convert a list of JSON files to JSONL files.

    Parameters
    ----------
    input_paths : List[str]
        List of paths to JSON files.
    output_paths : List[str]
        List of paths to JSONL files.
    """
    logging.info("Converting JSONs to JSONLs...")
    for input_path, output_path in tqdm(list(zip(input_paths, output_paths))):
        df = pd.read_json(input_path, orient="records")
        df.to_json(output_path, orient="records", lines=True, force_ascii=False)


def pd_read_jsons(paths: List[str], lines: bool):
    """Read a list of JSONs a load them into a single pandas dataframe.

    Parameters
    ----------
    paths : List[str]
        List of paths to JSON files.
    lines : bool
        Whether the JSONs are in line format.

    Returns
    -------
    df: pandas.DataFrame
        Pandas DataFrame.
    """
    logging.info("Reading JSONs to DataFrame...")
    first_df = True
    for path in tqdm(paths):
        if first_df:
            df = pd.read_json(path, orient="records", lines=lines)
            first_df = False
            continue
        df_ = pd.read_json(path, orient="records", lines=lines)
        df = pd.concat([df, df_])
    df = df.reset_index(drop=True)
    logging.info(df)
    return df


def dd_read_jsons(paths: List[str], lines: bool):
    """Read a list of JSONs a load them into a single dask dataframe.

    Parameters
    ----------
    paths : List[str]
        List of paths to JSON files.

    Returns
    -------
    ddf: dask.dataframe.DataFrame
        Dask DataFrame.
    """
    logging.info("Reading JSONs to Dask DataFrame...")
    first_ddf = True
    for path in tqdm(paths):
        if first_ddf:
            ddf = dd.read_json(path, orient="records", lines=lines)  # type: ignore
            first_ddf = False
            continue
        ddf_ = dd.read_json(path, orient="records", lines=lines)  # type: ignore
        ddf = dd.concat([ddf, ddf_])  # type: ignore
    ddf = ddf.reset_index(drop=True)  # type: ignore
    logging.info(ddf)
    return ddf
