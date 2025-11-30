
import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.models.decision_tree_classifier import DecisionTreeClassifier

class RandomForestClassifier(Model):
    """
    Random Forest Classifier.
    An ensemble learning method that fits a number of decision tree classifiers 
    on various sub-samples of the dataset and uses averaging to improve 
    the predictive accuracy and control over-fitting.
    """

    def __init__(self, n_estimators: int = 100, max_features: int = None, 
                 min_sample_split: int = 2, max_depth: int = 10, 
                 mode: str = 'gini', seed: int = 42, **kwargs):
        """
        Initializes the Random Forest Classifier.

        Parameters
        ----------
        n_estimators : int
            The number of trees in the forest.
        max_features : int
            The number of features to consider when looking for the best split.
            If None, then max_features = sqrt(n_features).
        min_sample_split : int
            The minimum number of samples required to split an internal node.
        max_depth : int
            The maximum depth of the tree.
        mode : str
            The function to measure the quality of a split.
        seed : int
            Random seed for reproducibility.
        """
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = []

    def _fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        """
        Builds a forest of trees from the training set (bootstrap samples).

        Parameters
        ----------
        dataset : Dataset
            The training dataset.

        Returns
        -------
        self : RandomForestClassifier
            The fitted model.
        """
        np.random.seed(self.seed)
        n_samples, n_features = dataset.shape()

        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        for _ in range(self.n_estimators):
            # Create bootstrap samples (random sampling with replacement) to ensure tree diversity
            sample_idxs = np.random.choice(n_samples, n_samples, replace=True)
            
            # Select random features without replacement to reduce correlation between trees
            feature_idxs = np.random.choice(n_features, self.max_features, replace=False)
            
            # Construct the subset dataset using the selected samples and features
            X_subset = dataset.X[sample_idxs][:, feature_idxs]
            y_subset = dataset.y[sample_idxs]
            bootstrap_dataset = Dataset(X_subset, 
                                        y_subset, 
                                        features=np.array(dataset.features)[feature_idxs], 
                                        label=dataset.label)

            # Initialize and train a decision tree on this specific subset of data
            tree = DecisionTreeClassifier(
                min_sample_split=self.min_sample_split,
                max_depth=self.max_depth,
                mode=self.mode
            )
            tree.fit(bootstrap_dataset)

            self.trees.append((feature_idxs, tree))

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class for X.
        The predicted class of an input sample is a vote by the trees in the forest.

        Parameters
        ----------
        dataset : Dataset
            The dataset to predict.

        Returns
        -------
        np.ndarray
            The predicted classes.
        """
        n_samples = dataset.shape()[0]
        tree_predictions = []

        # Collect predictions from each individual tree using its specific features
        for feature_idxs, tree in self.trees:
            # Filter dataset columns to match only the features used during this tree's training
            X_subset = dataset.X[:, feature_idxs]

            # Create temp dataset for prediction
            subset_dataset = Dataset(X_subset, 
                                     dataset.y, 
                                     features=np.array(dataset.features)[feature_idxs], 
                                     label=dataset.label)
            
            tree_predictions.append(tree.predict(subset_dataset))

        # Transpose to shape (n_samples, n_estimators)
        tree_predictions = np.array(tree_predictions).T

        # Apply majority voting: find the most frequent predicted class for each sample
        final_predictions = []
        for i in range(n_samples):
            # Get all votes for the current sample
            sample_votes = tree_predictions[i]

            # Count occurrences of each class label
            values, counts = np.unique(sample_votes, return_counts=True)

            # Select the class with the highest vote count
            most_common = values[np.argmax(counts)]
            final_predictions.append(most_common)

        return np.array(final_predictions)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        dataset : Dataset
            The test dataset containing true labels.
        predictions : np.ndarray
            The predicted labels.

        Returns
        -------
        float
            Accuracy score.
        """
        return accuracy(dataset.y, predictions)