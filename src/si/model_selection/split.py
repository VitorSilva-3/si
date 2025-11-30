from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test

def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> tuple[Dataset, Dataset]:
    """
    Splits the dataset into training and testing sets, preserving the class proportions (stratification).

    Parameters
    ----------
    dataset : Dataset
        The dataset object to split.
    test_size : float
        The proportion of the dataset to include in the test split (e.g., 0.2 for 20%).
    random_state : int
        Seed for the random number generator to ensure reproducibility.

    Returns
    -------
    tuple[Dataset, Dataset]
        A tuple containing the (train_dataset, test_dataset).
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Get unique labels and their counts
    labels, counts = np.unique(dataset.y, return_counts=True)
    
    train_indices = []
    test_indices = []
    
    # Loop through each class label to split data proportionally
    for label in labels:
        # Get indices corresponding to the current class
        curr_indices = np.where(dataset.y == label)[0]
        
        # Shuffle indices for this class
        np.random.shuffle(curr_indices)
        
        # Calculate number of test samples for this class
        n_test_samples = int(len(curr_indices) * test_size)
        
        # Select indices: first n_test_samples go to test, rest to train
        test_indices.extend(curr_indices[:n_test_samples])
        train_indices.extend(curr_indices[n_test_samples:])
        
    train_dataset = Dataset(X=dataset.X[train_indices], 
                            y=dataset.y[train_indices], 
                            features=dataset.features, 
                            label=dataset.label)
    
    test_dataset = Dataset(X=dataset.X[test_indices], 
                           y=dataset.y[test_indices], 
                           features=dataset.features, 
                           label=dataset.label)
    
    return train_dataset, test_dataset