
import unittest
import numpy as np
from si.statistics.tanimoto_similarity import tanimoto_similarity

class TestTanimotoSimilarity(unittest.TestCase):

    def setUp(self):
        # Initialize a single binary sample (query vector)
        self.x = np.array([1, 0, 1])
        
        # Initialize a set of binary samples with specific overlap scenarios
        self.y = np.array([
            [1, 0, 1],  # Identical to x
            [0, 1, 0],  # No overlap
            [1, 1, 1]   # Partial overlap
        ])

    def test_tanimoto_similarity_values(self):
        # Calculate Tanimoto distances between vector x and all samples in y
        distances = tanimoto_similarity(self.x, self.y)
        
        # Verify that the output array has one distance value per sample in y
        self.assertEqual(distances.shape[0], 3)
        
        # Check distance for identical vectors (should be 0.0)
        self.assertAlmostEqual(distances[0], 0.0)
        
        # Check distance for disjoint vectors (should be 1.0)
        self.assertAlmostEqual(distances[1], 1.0)
        
        # Check distance for partially overlapping vectors (1 - intersection/union)
        self.assertAlmostEqual(distances[2], 1 - (2/3))

    def test_tanimoto_similarity_types(self):
        # Define inputs as standard Python lists instead of NumPy arrays
        x_list = [1, 0, 1]
        y_list = [[1, 0, 1], [0, 1, 0]]
        
        # Compute distances using list inputs to test automatic type conversion
        distances = tanimoto_similarity(x_list, y_list)
        
        # Ensure the function returns a NumPy array
        self.assertIsInstance(distances, np.ndarray)
        
        # Verify that calculation is correct even when inputs were lists
        self.assertAlmostEqual(distances[0], 0.0)