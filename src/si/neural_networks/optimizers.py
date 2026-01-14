from abc import abstractmethod

import numpy as np


class Optimizer:

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Initialize the optimizer.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        momentum:
            The momentum to use for updating the weights.
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        return w - self.learning_rate * self.retained_gradient
    

class Adam(Optimizer):
    """
    The Adam optimizer combines features of SGD with momentum and RMSprop, 
    adjusting learning rates based on first and second moments of gradients.
    """

    def __init__(self, learning_rate: float, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize the Adam optimizer.

        Parameters
        ----------
        learning_rate : float
            The learning rate to use for updating the weights.
        beta_1 : float
            Exponential decay rate for the first moment estimate (default is 0.9).
        beta_2 : float
            Exponential decay rate for the second moment estimate (default is 0.999).
        epsilon : float
            Small value to prevent division by zero (default is 1e-8).
        """
        super().__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time stamp (epoch)

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights using the Adam optimization algorithm.

        Parameters
        ----------
        w : np.ndarray
            Current weights.
        grad_loss_w : np.ndarray
            Gradient of the loss function with respect to the weights.

        Returns
        -------
        np.ndarray
            The updated weights.
        """
        # Verify if m and v are initialized
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)

        # Update t (t += 1)
        self.t += 1

        # Compute and update m (Momentum) - m_t = beta1 * m_{t-1} + (1 - beta1) * g
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad_loss_w

        # Compute and update v (RMSprop) - v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (grad_loss_w ** 2)

        # Compute m_hat (Bias Correction) - m_hat = m_t / (1 - beta1^t)
        m_hat = self.m / (1 - self.beta_1 ** self.t)

        # Compute v_hat (Bias Correction) - v_hat = v_t / (1 - beta2^t)
        v_hat = self.v / (1 - self.beta_2 ** self.t)

        # Return the updated weights - w = w - lr * (m_hat / (sqrt(v_hat) + epsilon))
        return w - self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon))