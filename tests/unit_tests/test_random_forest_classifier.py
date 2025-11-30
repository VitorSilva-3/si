
from unittest import TestCase
import os
import numpy as np
from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.random_forest_classifier import RandomForestClassifier

class TestRandomForestClassifier(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        rf = RandomForestClassifier(n_estimators=10, seed=42)
        rf.fit(self.dataset)

        # Verify that the correct number of trees were created and stored
        self.assertEqual(len(rf.trees), 10)
        
        # Check if the first element in the trees list is a tuple (features, tree)
        self.assertIsInstance(rf.trees[0], tuple)
        self.assertEqual(len(rf.trees[0]), 2)

    def test_predict(self):
        train_dataset, test_dataset = train_test_split(self.dataset, test_size=0.3, random_state=42)
        
        rf = RandomForestClassifier(n_estimators=10, seed=42)
        rf.fit(train_dataset)
        
        predictions = rf.predict(test_dataset)

        # Verify that the number of predictions matches the number of test samples
        self.assertEqual(predictions.shape[0], test_dataset.shape()[0])
        
        # Ensure predictions are valid classes from the dataset labels
        unique_labels = np.unique(self.dataset.y)
        self.assertTrue(np.all(np.isin(predictions, unique_labels)))

    def test_score(self):
        train_dataset, test_dataset = train_test_split(self.dataset, test_size=0.3, random_state=42)
        
        rf = RandomForestClassifier(n_estimators=20, min_sample_split=2, max_depth=5, seed=42)
        rf.fit(train_dataset)
        accuracy = rf.score(test_dataset)

        # Verify that accuracy is within the valid range [0, 1]
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        
        # Check if accuracy is reasonable for Iris dataset (expecting high accuracy)
        self.assertGreater(accuracy, 0.90)