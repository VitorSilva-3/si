
from unittest import TestCase
from datasets import DATASETS_PATH
import os
import numpy as np

from si.feature_selection.select_percentile import SelectPercentile
from si.io.csv_file import read_csv
from si.statistics.f_classification import f_classification

class TestSelectPercentile(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        select_percentile = SelectPercentile(score_func=f_classification, percentile=50)
        select_percentile.fit(self.dataset)
        
        # Verify that F and p values are computed for all 4 features
        self.assertEqual(select_percentile.F.shape[0], 4)
        self.assertEqual(select_percentile.p.shape[0], 4)
        
        # Ensure that F-values are not all zero, indicating actual calculation
        self.assertTrue(np.sum(select_percentile.F) > 0)

    def test_transform(self):
        select_percentile = SelectPercentile(score_func=f_classification, percentile=50)
        select_percentile.fit(self.dataset)
        new_dataset = select_percentile.transform(self.dataset)

        # Verify that the number of features is reduced to exactly 2 (50% of 4)
        self.assertEqual(new_dataset.X.shape[1], 2)
        self.assertEqual(len(new_dataset.features), 2)
        
        # Confirm that 'petal_length', a known discriminative feature, is selected
        self.assertIn('petal_length', new_dataset.features)