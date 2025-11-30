from unittest import TestCase

import numpy as np


from datasets import DATASETS_PATH

import os
from si.io.csv_file import read_csv

from si.models.knn_classifier import KNNClassifier
from si.models.knn_regressor import KNNRegressor

from si.model_selection.split import train_test_split

class TestKNNClassifier(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        knn = KNNClassifier(k=3)

        knn.fit(self.dataset)

        self.assertTrue(np.all(self.dataset.features == knn.dataset.features))
        self.assertTrue(np.all(self.dataset.y == knn.dataset.y))

    def test_predict(self):
        knn = KNNClassifier(k=1)

        train_dataset, test_dataset = train_test_split(self.dataset)

        knn.fit(train_dataset)
        predictions = knn.predict(test_dataset)
        self.assertEqual(predictions.shape[0], test_dataset.y.shape[0])
        self.assertTrue(np.all(predictions == test_dataset.y))

    def test_score(self):
        knn = KNNClassifier(k=3)

        train_dataset, test_dataset = train_test_split(self.dataset)

        knn.fit(train_dataset)
        score = knn.score(test_dataset)
        self.assertEqual(score, 1)


class TestKNNRegressor(TestCase):

    def setUp(self):
        self.csv_file_cpu = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset_cpu = read_csv(filename=self.csv_file_cpu, features=True, label=True)

    def test_fit(self):
        # Test if the model fits the data correctly
        knn = KNNRegressor(k=3)
        knn.fit(self.dataset_cpu)
        
        # Verify if the training dataset is stored correctly
        self.assertTrue(np.all(self.dataset_cpu.features == knn.dataset.features))
        self.assertEqual(knn.dataset.shape(), self.dataset_cpu.shape())

    def test_predict(self):
        # Test regression predictions
        knn = KNNRegressor(k=3)
        train_dataset, test_dataset = train_test_split(self.dataset_cpu, random_state=42)

        knn.fit(train_dataset)
        predictions = knn.predict(test_dataset)
        
        # Check if we have exactly one prediction per test sample
        self.assertEqual(predictions.shape[0], test_dataset.y.shape[0])
        
        # Ensure predictions are floats
        self.assertTrue(np.issubdtype(predictions.dtype, np.floating))

    def test_score(self):
        # Test RMSE score calculation
        knn = KNNRegressor(k=3)
        train_dataset, test_dataset = train_test_split(self.dataset_cpu, random_state=42)

        knn.fit(train_dataset)
        score = knn.score(test_dataset)
        
        # RMSE score must be non-negative (>= 0)
        self.assertGreaterEqual(score, 0.0)