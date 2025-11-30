
import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile(Transformer):
    """
    Selects the highest scoring features based on a specific percentile.
    """
    def __init__(self, percentile: float, score_func: callable = f_classification, **kwargs):
        """
        Selects features from the given percentile of a score function 
        and returns a new Dataset object with the selected features.

        Parameters
        ----------
        percentile : float
            Percentile (0-100) for selecting features.
        score_func : callable, optional
            Variance analysis function. Uses f_classification by default.
        """
        super().__init__(**kwargs)
        
        if not (0 <= percentile <= 100):
            raise ValueError("Percentile must be a float between 0 and 100")
            
        self.percentile = percentile
        self.score_func = score_func
        self.F = None
        self.p = None

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        Estimates the F and p values for each feature using the scoring function.

        Parameters
        ----------
        dataset : Dataset
            The training data.
            
        Returns
        -------
        self : SelectPercentile
            The fitted transformer.
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Selects features with the highest F value up to the specified percentile.

        Parameters
        ----------
        dataset : Dataset
            The dataset to transform.
            
        Returns
        -------
        dataset : Dataset
            A new Dataset object with the selected features.
        """
        # Calculate number of features to keep (k)
        num_features = len(dataset.features)
        k = int(num_features * (self.percentile / 100))
        
        # Ensure at least 1 feature if percentile > 0
        if k == 0 and self.percentile > 0:
            k = 1

        # Handle NaNs: replace with 0.0 so they appear last
        F_clean = np.nan_to_num(self.F, nan=0.0)
        
        # Sort indices descending (stable sort to handle ties)
        sorted_indices = np.argsort(-F_clean, kind='mergesort')
        
        # Select top k features
        best_indices = sorted_indices[:k]
        selected_features = np.array(dataset.features)[best_indices]
        
        return Dataset(
            X=dataset.X[:, best_indices], 
            y=dataset.y, 
            features=selected_features, 
            label=dataset.label
        )