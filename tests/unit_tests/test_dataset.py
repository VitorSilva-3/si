import unittest

import numpy as np

from si.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())

    def test_dropna(self):
        X = np.array([[1, 2], [3, np.nan], [5, 6]])
        y = np.array([1, 2, 3])
        dataset = Dataset(X, y)

        dataset.dropna()

        self.assertEqual((2, 2), dataset.shape())
        self.assertTrue(np.array_equal(dataset.X, np.array([[1, 2], [5, 6]])))
        self.assertTrue(np.array_equal(dataset.y, np.array([1, 3])))

    def test_fillna(self):
        X = np.array([[1, 2], [3, np.nan], [5, 6]])
        y = np.array([1, 2, 3])
        dataset = Dataset(X, y)
        
        dataset.fillna(0)
        
        expected_X = np.array([[1, 2], [3, 0], [5, 6]])
        self.assertTrue(np.array_equal(dataset.X, expected_X))

        X = np.array([[1, 2], [3, np.nan], [5, 6]]) 
        dataset_mean = Dataset(X, y)
        
        dataset_mean.fillna('mean')
        
        expected_X_mean = np.array([[1, 2], [3, 4], [5, 6]])
        self.assertTrue(np.array_equal(dataset_mean.X, expected_X_mean))

        X = np.array([[1, 10], [1, np.nan], [1, 30]])
        dataset_median = Dataset(X, y)
        
        dataset_median.fillna('median')
        
        expected_X_median = np.array([[1, 10], [1, 20], [1, 30]])
        self.assertTrue(np.array_equal(dataset_median.X, expected_X_median))

    def test_remove_by_index(self):
        X = np.array([[10, 11], [20, 21], [30, 31]])
        y = np.array(['a', 'b', 'c'])
        dataset = Dataset(X, y)

        dataset.remove_by_index(1)

        self.assertEqual((2, 2), dataset.shape())
        
        expected_X = np.array([[10, 11], [30, 31]])
        self.assertTrue(np.array_equal(dataset.X, expected_X))
        
        expected_y = np.array(['a', 'c'])
        self.assertTrue(np.array_equal(dataset.y, expected_y))