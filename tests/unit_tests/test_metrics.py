from unittest import TestCase

import numpy as np
from si.metrics.accuracy import accuracy
from si.metrics.mse import mse
from si.metrics.rmse import rmse

class TestMetrics(TestCase):

    def test_accuracy(self):

        y_true = np.array([0,1,1,1,1,1,0])
        y_pred = np.array([0,1,1,1,1,1,0])

        self.assertTrue(accuracy(y_true, y_pred)==1)

    def test_mse(self):

        y_true = np.array([0.1,1.1,1,1,1,1,0])
        y_pred = np.array([0,1,1.1,1,1,1,0])

        self.assertTrue(round(mse(y_true, y_pred), 3)==0.004)

    def test_rmse(self):
        y_true = np.array([10, 20, 30])
        y_pred = np.array([10, 20, 34])
        
        result = rmse(y_true, y_pred)
        expected = np.sqrt(np.mean((y_true - y_pred)**2))
        
        # Check if result matches expected value up to 3 decimal places
        self.assertAlmostEqual(result, expected, places=3)
        self.assertAlmostEqual(result, 2.309, places=3)
