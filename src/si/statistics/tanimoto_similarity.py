import numpy as np

def tanimoto_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculates the Tanimoto distance between a single binary sample x 
    and multiple binary samples y.

    Similarity: T(A, B) = (A . B) / (|A|^2 + |B|^2 - A . B)
    Where '.' is the dot product (intersection for binary data).

    Parameters
    ----------
    x : np.ndarray
        A single binary sample (1D array).
    y : np.ndarray
        Multiple binary samples (2D array).

    Returns
    -------
    np.ndarray
        An array containing the Tanimoto distances between x and each sample in y.
    """
    x = np.array(x)
    y = np.array(y)
    
    # Calculate intersection (A . B)
    xy = np.dot(y, x)
    
    # Calculate magnitude of X (|A|^2)
    xx = np.sum(x)
    
    # Calculate magnitude of y (|B|^2) for each row
    yy = np.sum(y, axis=1)
    
    # Calculate denominator (union: A + B - intersection)
    denominator = xx + yy - xy
    
    # Similarity
    with np.errstate(divide='ignore', invalid='ignore'):
        similarity = xy / denominator
    
    similarity = np.nan_to_num(similarity, nan=1.0) 
    
    # Distance (1 - Similarity)
    return 1.0 - similarity