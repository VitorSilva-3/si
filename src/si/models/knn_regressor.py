
from typing import Callable

import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor(Model):
    """
    Implements the k-Nearest Neighbors algorithm for regression problems.
    
    It predicts the target value for a new sample by calculating the average 
    of the target values from the 'k' closest samples in the training set.
    """

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance, **kwargs):
        """
        Initializes the KNN Regressor model.

        Parameters
        ----------
        k : int
            Number of neighbors to consider for the prediction.
            Default is 1.
        distance : Callable
            The metric used to calculate distance between samples (e.g., Euclidean).
            Default is euclidean_distance.
        """
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        Fits the model by storing the training dataset.

        Parameters
        ----------
        dataset : Dataset
            The training data (features and labels).

        Returns
        -------
        self : KNNRegressor
            The fitted model instance.
        """
        self.dataset = dataset
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts target values for the entire dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset containing samples to predict.

        Returns
        -------
        np.ndarray
            An array containing the predicted values for each sample.
        """
        n_samples = dataset.shape()[0]
        predictions = []

        # Iterate over each sample in the test dataset
        for i in range(n_samples):
            sample = dataset.X[i]

            # Calculate distances between the sample and all training samples
            dists = self.distance(sample, self.dataset.X)

            # Get indices of the k nearest neighbors
            k_indices = np.argpartition(dists, self.k)[:self.k]

            # Get the y values of the k nearest neighbors
            neighbor_values = self.dataset.y[k_indices]

            # Calculate the mean of the neighbors' values
            mean_value = np.mean(neighbor_values)
            
            predictions.append(mean_value)

        return np.array(predictions)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Evaluates the model performance using RMSE.

        Parameters
        ----------
        dataset : Dataset
            The test dataset containing true labels.
        predictions : np.ndarray
            The array of predicted values.

        Returns
        -------
        float
            The Root Mean Squared Error (RMSE).
        """
        return rmse(dataset.y, predictions)