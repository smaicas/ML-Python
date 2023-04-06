import unittest
import os
import pandas as pd
import json
from NjMl import create_partial_dataframe


class TestCreatePartialDataFrame(unittest.TestCase):

    def setUp(self):
        # Create a test dataset
        data = [
            {"id": 1, "name": "Alice", "age": 25, "gender": "F"},
            {"id": 2, "name": "Bob", "age": 30, "gender": "M"},
            {"id": 3, "name": "Charlie", "age": 35, "gender": "M"},
            {"id": 4, "name": "David", "age": 40, "gender": "M"},
            {"id": 5, "name": "Eve", "age": 45, "gender": "F"}
        ]
        self.dataset_path = "test_dataset.jsonl"
        with open(self.dataset_path, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")

    def tearDown(self):
        # Remove the test dataset file
        os.remove(self.dataset_path)

    def gen(self):
        # Test the method without any filters
        ds_path = "C:/Workspace/ML/Data/Codeforces/dump-original.jsonl/dump-original.jsonl"
        create_partial_dataframe(ds_path, items=3)
        new_dataset_path = ds_path.replace('.jsonl', '.test.jsonl')
        self.assertTrue(os.path.isfile(new_dataset_path))
        df = pd.read_json(new_dataset_path, lines=True)
        self.assertEqual(len(df), 3)
        os.remove(new_dataset_path)

    def test_without_filters(self):
        # Test the method without any filters
        create_partial_dataframe(self.dataset_path, items=3)
        new_dataset_path = self.dataset_path.replace('.jsonl', '.test.jsonl')
        self.assertTrue(os.path.isfile(new_dataset_path))
        df = pd.read_json(new_dataset_path, lines=True)
        self.assertEqual(len(df), 3)
        os.remove(new_dataset_path)

    def test_with_single_value_filter(self):
        # Test the method with a single-value filter
        filter_params = {"gender": "F"}
        create_partial_dataframe(self.dataset_path, items=3, filter_params=filter_params)
        new_dataset_path = self.dataset_path.replace('.jsonl', '.test.jsonl')
        self.assertTrue(os.path.isfile(new_dataset_path))
        df = pd.read_json(new_dataset_path, lines=True)
        self.assertEqual(len(df), 2)
        self.assertTrue(all(df["gender"] == "F"))

    def test_with_multiple_values_filter(self):
        # Test the method with a multiple-values filter
        filter_params = {"gender": ["F", "M"]}
        create_partial_dataframe(self.dataset_path, items=3, filter_params=filter_params)
        new_dataset_path = self.dataset_path.replace('.jsonl', '.test.jsonl')
        self.assertTrue(os.path.isfile(new_dataset_path))
        df = pd.read_json(new_dataset_path, lines=True)
        self.assertEqual(len(df), 3)
        self.assertTrue(all(df["gender"].isin(["F", "M"])))
