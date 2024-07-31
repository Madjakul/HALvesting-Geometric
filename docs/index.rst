.. HALvesting documentation master file, created by
   sphinx-quickstart on Tue Jul 30 16:08:27 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

####################################################################################
HALvesting-Geometric: Harvests citation networks from HAL (Hyper Articles en Ligne). 
####################################################################################

.. image:: https://img.shields.io/badge/arXiv-2407.20595-b31b1b.svg
   :target: https://arxiv.org/abs/2407.20595

.. image:: https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Data-yellow
    :target: https://huggingface.co/datasets/Madjakul/HALvest-Geometric

HALvesting Geometric is a Python project designed to crawl data from the HAL (Hyper Articles en Ligne) repository.
It provides functionalities to build citation networks in several languages and from several years.


Abstract
========

HAL (Hyper Articles en Ligne) is the French national publication repository, used by most higher education and research organizations for their open science policy.
As a digital library, it is a rich repository of scholarly documents, but its potential for advanced research has been underutilized.
We present HALvest, a unique dataset that bridges the gap between citation networks and the full text of papers submitted on HAL.
We craft our dataset by filtering HAL for scholarly publications, resulting in approximately 700,000 documents, spanning 56 languages across 13 identified domains, suitable for language model training, and yielding approximately 16.5 billion tokens (with 8 billion in French and 7 billion in English, the most represented languages).
We transform the metadata of each paper into a citation network, producing a directed heterogeneous graph.
This graph includes uniquely identified authors on HAL, as well as all open submitted papers, and their citations.
We provide a baseline for authorship attribution using the dataset, implement a range of state-of-the-art models in graph representation learning for link prediction, and discuss the usefulness of our generated knowledge graph structure.


Code
====

.. toctree::
    :maxdepth: 1

    halvesting_geometric/modules/modeling_link_prediction.rst
    halvesting_geometric/modules/link_classifier.rst
    halvesting_geometric/modules/sage.rst
    halvesting_geometric/modules/rggc.rst
    halvesting_geometric/modules/gat.rst
    halvesting_geometric/utils/data/link_prediction_datamodule.rst
    halvesting_geometric/utils/data/link_prediction_metadata.rst
    halvesting_geometric/utils/helpers.rst

About
=====

.. toctree::
    :maxdepth: 1

    about.md