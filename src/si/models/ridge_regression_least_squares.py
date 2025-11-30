
import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse

class RidgeRegressionLeastSquares(Model):
    """
    Implementation of ridge egression using the least squares method.
    It solves the linear problem using the normal equation with L2 regularization:
    theta = (X.T * X + lambda * I)^-1 * (X.T * y)
    """

    def __init__(self, l2_penalty: float = 1.0, scale: bool = True, **kwargs):
        """
        Initialize the ridge regression model.

        Parameters
        ----------
        l2_penalty : float
            The L2 regularization parameter (lambda).
        scale : bool
            Whether to scale the dataset features (mean 0, std 1) before fitting.
        """
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.scale = scale
        
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def _fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Estimates the model parameters using the normal equation.

        Parameters
        ----------
        dataset : Dataset
            The training dataset.

        Returns
        -------
        self : RidgeRegressionLeastSquares
            The fitted model.
        """
        X = dataset.X
        
        # Scale the data
        if self.scale:
            self.mean = np.nanmean(X, axis=0)
            self.std = np.nanstd(X, axis=0)
            self.std[self.std == 0] = 1.0 
            X = (X - self.mean) / self.std

        # Add intercept term (column of ones) to X
        m = X.shape[0]
        X_augmented = np.c_[np.ones(m), X]

        # Compute the penalty matrix (lambda * I)
        n_features = X_augmented.shape[1]
        penalty_matrix = np.eye(n_features) * self.l2_penalty
        
        # Change the first position to 0 to avoid penalizing the intercept (theta_zero)
        penalty_matrix[0, 0] = 0

        # Compute the model parameters: theta = (X.T * X + penalty)^-1 * (X.T * y)
        # Transpose of X
        X_T = X_augmented.T
        
        # Matrix multiplication X.T * X
        XtX = X_T.dot(X_augmented)
        
        # Add penalty
        A = XtX + penalty_matrix
        
        # Calculate inverse
        A_inv = np.linalg.inv(A)
        
        # Calculate X.T * y
        Xty = X_T.dot(dataset.y)
        
        # Final calculation of all thetas
        thetas_full = A_inv.dot(Xty)

        # Separate theta_zero (intercept) and theta (coefficients)
        self.theta_zero = thetas_full[0]
        self.theta = thetas_full[1:]

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the dependent variable (y) using the estimated coefficients.

        Parameters
        ----------
        dataset : Dataset
            The dataset to predict.

        Returns
        -------
        np.ndarray
            The predicted values.
        """
        X = dataset.X
        
        # Scale the data using the training mean/std
        if self.scale:
            X = (X - self.mean) / self.std

        # 2Add intercept term
        m = X.shape[0]
        X_augmented = np.c_[np.ones(m), X]

        # Compute predicted Y -> X * thetas
        # Concatenate theta_zero and theta to match X_augmented dimensions
        thetas_full = np.r_[self.theta_zero, self.theta]
        
        return X_augmented.dot(thetas_full)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculates the mean squared error (MSE) between real and predicted values.

        Parameters
        ----------
        dataset : Dataset
            The dataset containing real labels.
        predictions : np.ndarray
            The predicted values.

        Returns
        -------
        float
            The MSE value.
        """
        return mse(dataset.y, predictions)