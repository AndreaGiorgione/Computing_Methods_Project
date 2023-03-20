
"""File containing some tests for the check of
the subnetworks builder functions."""

import sys
import os

import unittest
import numpy as np

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from subnetworks import  build_extractor, build_classificator, build_regressor

class TestNetworks(unittest.TestCase):

    def test_extractor(self):
        input_dim = 10
        layers = 5
        neurons = 5
        model = build_extractor(input_dim, layers, neurons)
        sample = np.random.rand(1, input_dim)
        prediction = model.predict(sample)
        self.assertEqual(sample.shape, prediction.shape)
        self.assertEqual(len(model.layers), layers + 3) # Three extra layers (input, output, add)

    def test_classificator(self):
        input_dim = 10
        layers = 5
        neurons = 5
        output_dim = 1
        model = build_classificator(input_dim, layers, neurons, output_dim)
        sample = np.random.rand(1, input_dim)
        prediction = model.predict(sample)
        self.assertEqual(output_dim, prediction.shape[1])
        self.assertEqual(len(model.layers), layers + 2) # Two extra layers (input, output)

    def test_regressor(self):
        input_dim = 10
        layers = 5
        neurons = 5
        output_dim = 3
        model = build_regressor(input_dim, layers, neurons, output_dim)
        sample = np.random.rand(1, input_dim)
        prediction = model.predict(sample)
        self.assertEqual(output_dim, prediction.shape[1])
        self.assertEqual(len(model.layers), layers + 2) # Two extra layers (input, output)

if __name__ == "__main__":
    unittest.main()
