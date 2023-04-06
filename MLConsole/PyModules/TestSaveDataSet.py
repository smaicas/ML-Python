import unittest
import os
import pandas as pd
import json
from NjMl import *


class TestSaveDataset(unittest.TestCase):

    def setUp(self):
        self.path = "C:/Workspace/ML/Data/Codeforces/dump-original.jsonl/dump-original.jsonl"
        self.filter_params = {"language": ["C#"]}
        self.df = load_dataset(self.path, nitems=1000, filter_params=self.filter_params)

    def test_save_all(self):
        save_dataset(self.df, "test_save_all.jsonl")
        self.assertTrue(os.path.exists("test_save_all.jsonl"))
        with open("test_save_all.jsonl") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), len(self.df))

    def test_save_filtered(self):
        save_dataset(self.df, "test_save_filtered.jsonl")
        self.assertTrue(os.path.exists("test_save_filtered.jsonl"))
        with open("test_save_filtered.jsonl") as f:
            lines = f.readlines()
            self.assertLessEqual(len(lines), len(self.df))
            for line in lines:
                row = json.loads(line.strip())
                self.assertIn(self.filter_params["language"][0], row["language"])

    def tearDown(self):
        if os.path.exists("test_save_all.jsonl"):
            os.remove("test_save_all.jsonl")
        if os.path.exists("test_save_filtered.jsonl"):
            os.remove("test_save_filtered.jsonl")


if __name__ == '__main__':
    unittest.main()
