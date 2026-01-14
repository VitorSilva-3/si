
import os
import numpy as np
from unittest import TestCase

from datasets import DATASETS_PATH
from si.io.data_file import read_data_file
from si.neural_networks.optimizers import Adam

class TestAdamOptimizer(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")
        
        # Initialize dummy weights and gradients matching dataset features
        self.w = np.random.rand(self.dataset.X.shape[1], 1)
        self.grad_loss_w = np.random.rand(self.dataset.X.shape[1], 1)

    def test_adam_update(self):
        
        adam = Adam(learning_rate=0.01)
        
        # Perform update
        new_w = adam.update(self.w, self.grad_loss_w)

        # Check if the output shape is preserved
        self.assertEqual(new_w.shape, self.w.shape)
        
        # Check if weights were actually updated (not equal to old weights)
        self.assertFalse(np.allclose(new_w, self.w))

        # Check if internal state (memory) was initialized correctly
        self.assertEqual(adam.t, 1, "Time step should be 1 after first update")
        self.assertIsNotNone(adam.m, "Momentum (m) not initialized")
        self.assertIsNotNone(adam.v, "Velocity (v) not initialized")