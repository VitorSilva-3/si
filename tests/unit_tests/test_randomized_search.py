
from unittest import TestCase
import os
import numpy as np

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.model_selection.randomized_search import randomized_search_cv
from si.models.logistic_regression import LogisticRegression
from si.metrics.accuracy import accuracy

class TestRandomizedSearchCV(TestCase):

    def setUp(self):

        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_csv(filename=self.csv_file, label=True, sep=",")

    def test_randomized_search_k_fold_cross_validation(self):
        """
        Test the randomized_search_cv function logic and outputs.
        """
        
        np.random.seed(42)

        model = LogisticRegression()

        # Define the hyperparameter grid
        parameter_grid = {
            'l2_penalty': np.linspace(1, 10, 10),
            'alpha': np.linspace(0.001, 0.0001, 100),
            'max_iter': np.linspace(1000, 2000, 200).astype(int)
        }

        # Perform randomized search
        results = randomized_search_cv(
            model=model,
            dataset=self.dataset,
            hyperparameter_grid=parameter_grid,
            cv=3,
            n_iter=10,
            scoring=accuracy
        )

        # Validate structure
        self.assertEqual(len(results["scores"]), 10, 
                         "The number of scores does not match n_iter.")

        # Validate best params count
        best_hyperparameters = results['best_hyperparameters']
        self.assertEqual(len(best_hyperparameters), 3, 
                         "The best hyperparameters should contain 3 parameters.")

        # Validate best score
        best_score = results['best_score']
        print(f"Score obtained in test: {best_score}")
        
        # Check if the score is reasonably high (>= 0.96)
        self.assertGreaterEqual(round(best_score, 2), 0.96, 
                         f"The best score ({best_score}) is lower than expected.")