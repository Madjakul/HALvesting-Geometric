# HALvesting Geometric

# TODO: Changer le badge ArXiv/HAL
[![arXiv](https://img.shields.io/badge/arXiv-2309.08351-b31b1b.svg)](https://arxiv.org/abs/2309.08351)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Data-yellow)](https://huggingface.co/datasets/Madjakul/HALvest-Geometric)

Harvests citation networks from HAL.

---

HALvesting Geometric is a Python project designed to crawl data from the [HAL (Hyper Articles en Ligne) repository](https://hal.science/). It provides functionalities to build citation networks in several languages and from several years.


## Features

* [**build_metadata.py**](build_metadata.py): given languages and years, build a raw citation network.
* [**link_prediction**](link_prediction.py): harness link prediction to perform autorship attribution on the citation network and evaluate it if needed.


## Requirements

You will need Python > 3.8.


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

It's easier to modify the files [`scripts/build_metadata.sh`](scripts/build_metadata.sh) and [`scripts/link_prediction.sh`](scripts/link_prediction.sh) at need and launch them. However one can launch directly the Python scripts with the correct arguments as we will see below.


### Build Metadata

This script is used to build the raw citation network and storing it into CSV files. For each node type, each node are given a unique integer identifier. The edges are then computed as 2-uples using the said identifiers.

### Hyperparameter Tuning

**Requires [Weight and Biases](https://wandb.ai/site)!**
Runs sweeps on the link prediction model to get the best hyperparameters.

### Link Prediction

Perform link prediction on author <-> paper pairs and evaluate it. The test set is a random subgraph from the citation network with 30% of disjoint nodes.


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