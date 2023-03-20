
"""File containing some tests for the check of
the correlation related loss functions."""

import sys
import os

import unittest
import tensorflow as tf

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from custom_loss_functions import correlation_coefficient_loss, inverse_correlation_coefficient_loss

class TestCorrelation(unittest.TestCase):
    """Unittesting for the loss functions."""

    def test_correlation_coefficient_loss(self):
        """Testing the correlation coefficient squared."""
        first_tensor = tf.constant([1.0, 2.0, 3.0])
        second_tensor = tf.constant([1.0, 2.0, 3.0])
        correlation = correlation_coefficient_loss(first_tensor, second_tensor)
        self.assertEqual(tf.get_static_value(correlation), 1.0)

        first_tensor = tf.constant([4.0, 5.0, 6.0])
        second_tensor = tf.constant([-4.0, -5.0, -6.0])
        correlation = correlation_coefficient_loss(first_tensor, second_tensor)
        self.assertEqual(tf.get_static_value(correlation), 1.0)

        first_tensor = tf.constant([1.0, 3.0, 1.0])
        second_tensor = tf.constant([1.0, 3.0, 5.0])
        correlation = correlation_coefficient_loss(first_tensor, second_tensor)
        self.assertEqual(tf.get_static_value(correlation), 0.0)

    def test_invers_correlation_coefficient_loss(self):
        """Testing one minus the correlation coefficient squared."""
        first_tensor = tf.constant([1.0, 2.0, 3.0])
        second_tensor = tf.constant([1.5, 2.5, 3.5])
        correlation = inverse_correlation_coefficient_loss(first_tensor, second_tensor)
        self.assertEqual(tf.get_static_value(correlation), 0.0)

        first_tensor = tf.constant([4.0, 5.0, 6.0])
        second_tensor = tf.constant([-4.5, -5.5, -6.5])
        correlation = inverse_correlation_coefficient_loss(first_tensor, second_tensor)
        self.assertEqual(tf.get_static_value(correlation), 0.0)

        first_tensor = tf.constant([1.0, 4.0, 1.0])
        second_tensor = tf.constant([1.0, 4.0, 7.0])
        correlation = inverse_correlation_coefficient_loss(first_tensor, second_tensor)
        self.assertEqual(tf.get_static_value(correlation), 1.0)

if __name__ == "__main__":
    unittest.main()
