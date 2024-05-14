# tests/test_helpers.py

import os

import pytest

from halph.utils.helpers import json_to_dict, jsons_to_dict, read_json, read_jsons

# Define the path to the directory containing mock JSON files
MOCK_DIR = os.path.join(os.getcwd(), "data", "mock", "responses")

# Define the paths to the mock JSON files
MOCK_JSON_1 = os.path.join(MOCK_DIR, "mock_file_1.json")
MOCK_JSON_2 = os.path.join(MOCK_DIR, "mock_file_2.json")

# Define the key to transpose JSON files
TRANSPOSE_KEY = "halid"


def test_read_json():
    # Test reading a single JSON file
    data = read_json(MOCK_JSON_1)
    assert isinstance(data, list)
    assert isinstance(data[0], dict)
    # Add more assertions specific to your JSON structure if needed


def test_read_jsons():
    # Test reading multiple JSON files
    data = read_jsons([MOCK_JSON_1, MOCK_JSON_2])
    assert isinstance(data, list)
    assert isinstance(data[0], dict)
    # Add more assertions specific to your JSON structure if needed


def test_transpose_json():
    # Test transposing a single JSON file
    transposed_data = json_to_dict(MOCK_JSON_1, TRANSPOSE_KEY)
    assert isinstance(transposed_data, dict)
    # Add more assertions specific to your transposed JSON structure if needed


def test_transpose_jsons():
    # Test transposing multiple JSON files
    transposed_data = jsons_to_dict([MOCK_JSON_1, MOCK_JSON_2], TRANSPOSE_KEY)
    assert isinstance(transposed_data, dict)
    # Add more assertions specific to your transposed JSON structure if needed
