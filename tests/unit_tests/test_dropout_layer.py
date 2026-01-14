
from unittest import TestCase
import numpy as np
from si.neural_networks.layers import Dropout

class TestDropoutLayer(TestCase):

    def setUp(self):
        # 600 samples, 100 features
        self.X = np.random.rand(600, 100)

    def test_forward_propagation(self):
        dropout_layer = Dropout(probability=0.2)
        
        # Test training mode
        training_mode_output = dropout_layer.forward_propagation(self.X, training=True)
        
        # Check if the shape of the output matches the input shape
        self.assertEqual(training_mode_output.shape, self.X.shape,
                         "Output shape does not match input shape in training mode.")
        
        # Check if some values are dropped out (set to 0)
        self.assertTrue(np.any(training_mode_output == 0),
                        "No neurons were dropped in training mode (mask failed).")
        
        expected_output = self.X * dropout_layer.mask
        
        # Use allclose to handle small floating-point precision differences
        self.assertTrue(np.allclose(training_mode_output, expected_output),
                        "Inverted dropout scaling calculation is incorrect.")

        # Test inference mode
        inference_mode_output = dropout_layer.forward_propagation(self.X, training=False)
        
        # Check if the output matches the input exactly in inference mode
        self.assertTrue(np.array_equal(inference_mode_output, self.X),
                        "Output must be identical to input in inference mode.")

    def test_backward_propagation(self):
        dropout_layer = Dropout(probability=0.2)
        
        # Run forward propagation first to generate the mask
        dropout_layer.forward_propagation(self.X, training=True)
        
        # Create a random output error (simulating gradient from next layer)
        output_error = np.random.random((self.X.shape))
        
        # Compute input error
        input_error = dropout_layer.backward_propagation(output_error)
        
        # Check if the shape matches
        self.assertEqual(input_error.shape, output_error.shape,
                         "Gradient shape mismatch.")
        
        # Check if the input error is masked correctly
        expected_error = output_error * dropout_layer.mask
        
        self.assertTrue(np.allclose(input_error, expected_error),
                        "Error was not correctly propagated (masked) during backward pass.")