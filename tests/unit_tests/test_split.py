from unittest import TestCase

from datasets import DATASETS_PATH

import os
import numpy as np
from si.io.csv_file import read_csv

from si.model_selection.split import train_test_split, stratified_train_test_split

class TestSplits(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_train_test_split(self):

        train, test = train_test_split(self.dataset, test_size = 0.2, random_state=123)
        test_samples_size = int(self.dataset.shape()[0] * 0.2)
        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)

    def test_stratified_train_test_split(self):
        train, test = stratified_train_test_split(self.dataset, test_size=0.2, random_state=123)

        # Check Dimensions (Shapes)
        expected_test_size = int(self.dataset.shape()[0] * 0.2) # Should be 30
        self.assertEqual(test.shape()[0], expected_test_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - expected_test_size)

        # Check Stratification (Class proportions)
        labels, counts = np.unique(test.y, return_counts=True)

        # 3 unique classes in the test set
        self.assertEqual(len(labels), 3)

        # Each class has exactly 10 samples (150 total * 0.2 split / 3 classes)
        for count in counts:
            self.assertEqual(count, 10, "Stratification failed: Classes are not balanced in the test set.")