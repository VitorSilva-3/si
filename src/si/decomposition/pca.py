
import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset

class PCA(Transformer):
    """
    Principal Component Analysis (PCA) implementation.
    Reduces the dimensionality of the dataset.
    """

    def __init__(self, n_components: int, **kwargs):
        """
        Initialize the PCA transformer.

        Parameters
        ----------
        n_components : int
            The number of principal components to keep.
        """
        super().__init__(**kwargs)
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def _fit(self, data: Dataset) -> 'PCA':
        """
        Fits the PCA model to the dataset by calculating the covariance matrix
        and its eigenvectors.

        Parameters
        ----------
        data : Dataset
            The training data.

        Returns
        -------
        self : PCA
            The fitted instance.
        """
        n_samples, n_features = data.shape()
        if self.n_components < 1 or self.n_components > n_features:
            raise ValueError(f"n_components must be between 1 and {n_features}")

        # Center the data
        self.mean = np.mean(data.X, axis=0)
        centered_data = data.X - self.mean

        # Calculates the covariance matrix and its eigenvalues decomposition
        cov_matrix = np.cov(centered_data, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

        # Sort eigenvalues and select top n_components
        sorted_idxs = np.argsort(eig_vals)[::-1] 
        top_idxs = sorted_idxs[:self.n_components]
        self.components = eig_vecs[:, top_idxs].T
        
        # Calculate ratio of variance explained
        total_var = np.sum(eig_vals)
        self.explained_variance = eig_vals[top_idxs] / total_var

        return self

    def _transform(self, data: Dataset) -> Dataset:
        """
        Projects the data onto the principal components.

        Parameters
        ----------
        data : Dataset
            The dataset to be transformed.

        Returns
        -------
        Dataset
            The dataset with reduced dimensionality.
        """
        # Center the new data using the training mean
        centered_X = data.X - self.mean

        # Project data (matrix multiplication)
        # Structure: (samples x features) . (features x components)
        projected_X = np.dot(centered_X, self.components.T)

        new_feat_names = [f"PC{i+1}" for i in range(self.n_components)]

        return Dataset(
            X=projected_X,
            y=data.y,
            features=np.array(new_feat_names),
            label=data.label
        )