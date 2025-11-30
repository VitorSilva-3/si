
from unittest import TestCase
import os
import numpy as np
from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.ridge_regression_least_squares import RidgeRegressionLeastSquares

class TestRidgeRegressionLeastSquares(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        # Initialize model with small L2 penalty
        model = RidgeRegressionLeastSquares(l2_penalty=0.1, scale=True)
        model.fit(self.dataset)

        # Verify that parameters are estimated
        self.assertIsNotNone(model.theta)
        self.assertIsNotNone(model.theta_zero)
        
        # Check dimensions of coefficients (must match number of features)
        self.assertEqual(model.theta.shape[0], self.dataset.shape()[1])
        
        # Check scaling parameters
        self.assertIsNotNone(model.mean)
        self.assertIsNotNone(model.std)

    def test_predict(self):
        model = RidgeRegressionLeastSquares(l2_penalty=0.1, scale=True)
        train, test = train_test_split(self.dataset, test_size=0.2, random_state=42)
        
        model.fit(train)
        predictions = model.predict(test)
        
        # Verify predictions shape matches test set size
        self.assertEqual(predictions.shape[0], test.shape()[0])
        
        # Verify predictions are float numbers
        self.assertTrue(np.issubdtype(predictions.dtype, np.floating))

    def test_score(self):
        model = RidgeRegressionLeastSquares(l2_penalty=0.1, scale=True)
        train, test = train_test_split(self.dataset, test_size=0.2, random_state=42)
        
        model.fit(train)
        mse_score = model.score(test)
        
        # Check that MSE is calculated and reasonable
        self.assertGreater(mse_score, 0.0)