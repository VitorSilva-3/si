
import numpy as np
from si.metrics.accuracy import accuracy
from si.data.dataset import Dataset
from si.base.model import Model

class StackingClassifier(Model):
    """
    Stacking is an ensemble method that combines predictions from several base models 
    and uses another model (meta-model) to generate the final predictions.
    """

    def __init__(self, models: list, final_model: Model, **kwargs):
        """
        Initialize the StackingClassifier with base models and a final model.

        Parameters
        ----------
        models : list
            List of base models to be trained. Each model must implement fit and predict methods.
        final_model : Model
            The meta-model used to aggregate predictions from base models. It must also implement fit and predict methods.
        """
        super().__init__(**kwargs)
        self.models = models
        self.final_model = final_model

    def _fit(self, dataset: Dataset) -> "StackingClassifier":
        """
        Train the StackingClassifier.

        Parameters
        ----------
        dataset : Dataset
            The training dataset.

        Returns
        -------
        StackingClassifier
            The trained model (self).
        """
        # Train base models
        for model in self.models:
            model.fit(dataset)

        # Get predictions 
        base_predictions = np.array([model.predict(dataset) for model in self.models]).T

        # Create dataset for meta-learner
        stacked_dataset = Dataset(
            X=base_predictions,
            y=dataset.y,
            features=[f"model_{i}" for i in range(len(self.models))],
            label=dataset.label
        )

        # Train meta-learner
        self.final_model.fit(stacked_dataset)

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        dataset : Dataset
            The dataset to predict the class labels for.

        Returns
        -------
        np.ndarray
            The predicted class labels.
        """
        # Get predictions from base models
        base_predictions = np.array([model.predict(dataset) for model in self.models]).T

        # Create dataset for meta-learner
        new_dataset = Dataset(
            X=base_predictions,
            y=None,
            features=[f"model_{i}" for i in range(len(self.models))],
            label=None
        )

        # Final prediction
        return self.final_model.predict(new_dataset)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculate accuracy.

        Parameters
        ----------
        dataset : Dataset
            The dataset used to evaluate the model.
        predictions : np.ndarray
            The predictions made by the model.

        Returns
        -------
        float
            The accuracy of the model on the dataset.
        """
        return accuracy(dataset.y, predictions)