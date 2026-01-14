
from unittest import TestCase
import os
import numpy as np
from datasets import DATASETS_PATH

from si.ensemble.stacking_classifier import StackingClassifier
from si.io.csv_file import read_csv
from si.model_selection.split import stratified_train_test_split
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression

class TestStackingClassifier(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_csv(filename=self.csv_file, label=True, sep=",")
        self.train_dataset, self.test_dataset = stratified_train_test_split(self.dataset, test_size=0.2, random_state=42)

    def test_fit(self):

        # Initialize base models and final model
        knn = KNNClassifier(k=3)
        logistic_regression = LogisticRegression(l2_penalty=0.1, alpha=0.01, max_iter=1000)
        decision_tree = DecisionTreeClassifier(min_sample_split=2, max_depth=5)
        knn_final = KNNClassifier(k=3)

        # Create the stacking classifier
        stacking_classifier = StackingClassifier(models=[knn, logistic_regression, decision_tree], final_model=knn_final)

        # Call fit
        stacking_classifier.fit(self.train_dataset)

        # Validate that models are stored correctly
        self.assertEqual(len(stacking_classifier.models), 3, "Number of base models is incorrect.")
        
        # Validate hyperparameters for one of the base models (decision tree) to ensure object integrity
        self.assertEqual(stacking_classifier.models[2].min_sample_split, 2, "DecisionTreeClassifier min_sample_split is incorrect.")
        self.assertEqual(stacking_classifier.models[2].max_depth, 5, "DecisionTreeClassifier max_depth is incorrect.")

    def test_predict(self):

        # Initialize base models and final model
        knn = KNNClassifier(k=3)
        logistic_regression = LogisticRegression(l2_penalty=0.1, alpha=0.01, max_iter=1000)
        decision_tree = DecisionTreeClassifier(min_sample_split=2, max_depth=5)
        knn_final = KNNClassifier(k=3)

        # Create and train the stacking classifier
        stacking_classifier = StackingClassifier(models=[knn, logistic_regression, decision_tree], final_model=knn_final)
        stacking_classifier.fit(self.train_dataset)

        # Generate predictions for the test dataset
        predictions = stacking_classifier.predict(self.test_dataset)

        # Validate that the number of predictions matches the number of test samples
        self.assertEqual(predictions.shape[0], self.test_dataset.shape()[0], 
                         "The number of predictions does not match the number of test samples.")
        
        # Validate return type
        self.assertIsInstance(predictions, np.ndarray, "Predictions should be a numpy array.")

    def test_score(self):
        """
        Test the score method to verify that the calculated accuracy matches the expected value.
        """
        # Initialize base models and final model
        knn = KNNClassifier(k=3)
        logistic_regression = LogisticRegression(l2_penalty=0.1, alpha=0.01, max_iter=1000)
        decision_tree = DecisionTreeClassifier(min_sample_split=2, max_depth=5)
        knn_final = KNNClassifier(k=3)

        # Create and train the stacking classifier
        stacking_classifier = StackingClassifier(models=[knn, logistic_regression, decision_tree], final_model=knn_final)
        stacking_classifier.fit(self.train_dataset)

        # Calculate the accuracy of the stacking classifier
        accuracy_ = stacking_classifier.score(self.test_dataset)

        # Validate the accuracy score
        print(f"Obtained Accuracy: {accuracy_}")
        self.assertEqual(round(accuracy_, 2), 0.95, 
                         f"The accuracy of the StackingClassifier ({accuracy_}) does not match the expected value (0.95).")