# HALvesting Geometric

[![arXiv](https://img.shields.io/badge/arXiv-2309.08351-b31b1b.svg)](https://arxiv.org/abs/2309.08351)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Data-yellow)](https://huggingface.co/datasets/Madjakul/HALvest-Geometric)

Harvests citation networks from HAL.
* See also [HALvesting](https://github.com/Madjakul/HALvesting)

---

HALvesting Geometric is a Python project designed to crawl data from the [HAL (Hyper Articles en Ligne) repository](https://hal.science/). It provides functionalities to build citation networks in several languages and from several years.


## Features

* [**build_metadata.py**](build_metadata.py): given languages and years, build a raw citation network.
* [**link_prediction**](link_prediction.py): train a link prediction model on a random split fo the data and evaluate the said model on another split.


## Requirements

* Python > 3.8.


## Installation

1. Clone the repository

```sh
git clone https://github.com/Madjakul/HALvesting-Geometric.git
```

2. Navigate to the project directory

```sh
cd HALvesting-Geometric
```

3. Install the required dependencies:

```sh
pip install -r requirements.txt
```


## Usage

It's easier to modify the files [`scripts/build_metadata.sh`](scripts/build_metadata.sh) and [`scripts/link_prediction.sh`](scripts/link_prediction.sh) at need before launching them. However one can launch directly the Python scripts, with the correct arguments as we will see below.


### Build Metadata

This script is used to build the raw citation network and storing it into CSV files.

```
usage: build_metadata.py [-h] [--root_dir ROOT_DIR] [--json_dir JSON_DIR] [--xml_dir XML_DIR] --raw_dir RAW_DIR --compute_nodes COMPUTE_NODES [--lang LANG]
                         --compute_edges COMPUTE_EDGES [--cache_dir CACHE_DIR] [--dataset_checkpoint DATASET_CHECKPOINT] [--dataset_config_path DATASET_CONFIG_PATH]
                         [--zip_compress ZIP_COMPRESS]

Argument parser used to build the graphs' metadata.

options:
  -h, --help            show this help message and exit
  --root_dir ROOT_DIR   Path to the directory where the processed dataset will be saved.
  --json_dir JSON_DIR   Path to the directyory where the JSON responses are stored.
  --xml_dir XML_DIR     Path to the directory where the TEI XML files are stored.
  --raw_dir RAW_DIR     Path to the raw computed graph.
  --compute_nodes COMPUTE_NODES
                        Weather to compute the nodes or not.
  --lang LANG           Language of the documents to consider.
  --compute_edges COMPUTE_EDGES
                        Weather to compute the edges or not.
  --cache_dir CACHE_DIR
                        Path to the directory where the dataset will be cached.
  --dataset_checkpoint DATASET_CHECKPOINT
                        Name of the HuggingFace dataset containing the filtered documents.
  --dataset_config_path DATASET_CONFIG_PATH
                        Path to the file containing the dataset configurations to download.
  --zip_compress ZIP_COMPRESS
                        Weather to compress the metadata or not.
```

### Link Prediction

Train a link prediction model on author <-> paper pairs and evaluate it.

```
usage: link_prediction.py [-h] --config_file CONFIG_FILE --root_dir ROOT_DIR [--lang_ LANG_] [--accelerator ACCELERATOR] [--wandb WANDB] [--num_proc NUM_PROC] --run RUN

Arguments for link prediction.

options:
  -h, --help            show this help message and exit
  --config_file CONFIG_FILE
                        Path to the configuration file.
  --root_dir ROOT_DIR   Path to the data root directory.
  --lang_ LANG_         Language to use for the dataset.
  --accelerator ACCELERATOR
                        Accelerator to use for training.
  --wandb WANDB         Enable Weights and Biases logging.
  --num_proc NUM_PROC   Number of processes to use.
  --run RUN             Index of the run. Only used during experiments to log the run.
```


## Citation

To cite HALvesting/HALvest:

```bib
@software{almanach_halvest_2024,
  author = {Kulumba, Francis and Antoun, Wissam and Vimont, Guillaume and Romary, Laurent},
  title = {HALvest: Open Scientific Papers Harvested from HAL.},
  month = April,
  year = 2024,
  company = Almanach,
  url = {https://github.com/Madjakul/HALvesting}
}
```


## License

This project is licensed under the [Apache License 2.0](LICENSE).