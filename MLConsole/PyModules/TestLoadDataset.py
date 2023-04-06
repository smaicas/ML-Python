import unittest
import os
import pandas as pd
import json
from NjMl import *


class TestLoadDataset(unittest.TestCase):
    def setUp(self):
        self.test_data_path = "C:/Workspace/ML/Data/Codeforces/dump-original.jsonl/dump-original.jsonl"

    def test_load_all_data(self):
        df = load_dataset(self.test_data_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(1262910,len(df))

    def test_load_n_items(self):
        df = load_dataset(self.test_data_path, nitems=100)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 100)

    def test_load_filtered_data(self):
        filter_params = {"language": ["C#", "Python"]}
        df = load_dataset(self.test_data_path, filter_params=filter_params)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertTrue(df["language"].str.contains("|".join(filter_params["language"])).all())


if __name__ == '__main__':
    main()
