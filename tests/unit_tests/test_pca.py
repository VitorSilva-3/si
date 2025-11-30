
from unittest import TestCase
import os
import numpy as np
from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.decomposition.pca import PCA

class TestPCA(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        # Initialize PCA with 2 components and fit to the dataset
        n_components = 2
        pca = PCA(n_components=n_components)
        pca.fit(self.dataset)

        # Check if the estimated mean vector has the correct shape (4 features)
        self.assertEqual(pca.mean.shape[0], 4)
        
        # Check if components matrix has correct shape (2 components x 4 features)
        self.assertEqual(pca.components.shape, (n_components, 4))
        
        # Check if explained variance has correct shape (2 values)
        self.assertEqual(pca.explained_variance.shape[0], n_components)
        
        # Verify that the first component explains more variance than the second (PC1 > PC2)
        self.assertGreater(pca.explained_variance[0], pca.explained_variance[1])

    def test_transform(self):
        # Initialize and fit PCA with 2 components
        n_components = 2
        pca = PCA(n_components=n_components)
        pca.fit(self.dataset)
        
        # Transform the dataset to the reduced space
        new_dataset = pca.transform(self.dataset)

        # Verify that the number of samples remains 150
        self.assertEqual(new_dataset.shape()[0], 150)
        
        # Verify that the number of features is reduced to 2
        self.assertEqual(new_dataset.shape()[1], n_components)
        
        # Ensure the feature names are updated correctly (PC1, PC2)
        self.assertTrue(np.all(new_dataset.features == ['PC1', 'PC2']))

    def test_full_variance(self):
        # Initialize PCA keeping all components (4) to check total variance
        pca = PCA(n_components=4)
        pca.fit(self.dataset)
        
        # Sum of explained variance ratio must be approximately 1.0 (100%)
        self.assertAlmostEqual(np.sum(pca.explained_variance), 1.0)

    def test_invalid_n_components(self):
        # Check that requesting more components than features raises ValueError
        with self.assertRaises(ValueError):
            pca = PCA(n_components=5)
            pca.fit(self.dataset)