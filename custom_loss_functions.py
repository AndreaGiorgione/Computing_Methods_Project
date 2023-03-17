
"""File containing losses for neural networks."""

from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp

from keras import backend

def correlation_coefficient_loss(y_true: tf.Tensor, y_predicted: tf.Tensor) -> tf.Tensor:
    """Pearson correlation squared between two batches.

    Arguments
    ---------
    y_true : tf.Tensor
    Taget values of the used network.

    y_predicted : tf.Tensor
    Predictions made by the used model.

    Returns
    ---------
    Correlation squared (range between 0 and 1) as TensorFlow tensor.
    """
    mean_y_true = backend.mean(y_true)
    mean_y_predicted  = backend.mean(y_predicted)

    shifted_y_true = y_true - mean_y_true
    shifted_y_predicted = y_predicted - mean_y_predicted

    numerator = backend.sum(tf.multiply(shifted_y_true,shifted_y_predicted))
    denominator = backend.sqrt(tf.multiply(backend.sum(backend.square(shifted_y_true)),
                                           backend.sum(backend.square(shifted_y_predicted))))

    correlation_coefficient = numerator / denominator
    correlation_coefficient = backend.maximum(backend.minimum(correlation_coefficient, 1.0), -1.0)

    return backend.square(correlation_coefficient)

def inverse_correlation_coefficient_loss(y_true: tf.Tensor, y_predicted: tf.Tensor) -> tf.Tensor:
    """Pearson correlation squared between two batches.

    Arguments
    ---------
    y_true : tf.Tensor
    Taget values of the used network.

    y_predicted : tf.Tensor
    Predictions made by the used model.

    Returns
    ---------
    One minuse correlation squared (range between 0 and 1) as TensorFlow tensor.
    """
    mean_y_true = backend.mean(y_true)
    mean_y_predicted  = backend.mean(y_predicted)

    shifted_y_true = y_true - mean_y_true
    shifted_y_predicted = y_predicted - mean_y_predicted

    numerator = backend.sum(tf.multiply(shifted_y_true,shifted_y_predicted))
    denominator = backend.sqrt(tf.multiply(backend.sum(backend.square(shifted_y_true)),
                                           backend.sum(backend.square(shifted_y_predicted))))

    correlation_coefficient = numerator / denominator
    correlation_coefficient = backend.maximum(backend.minimum(correlation_coefficient, 1.0), -1.0)

    return 1 - backend.square(correlation_coefficient)

def correlation_loss(y_true: Any, y_pred: Any) -> tf.Tensor:
    """Pearson correlation squared between two batches.

    Arguments
    ---------
    y_true : Any
    Taget values of the used network.

    y_predicted : Any
    Predictions made by the used model.

    Returns
    ---------
    Correlation squared (range between 0 and 1) as TensorFlow tensor.
    """
    return backend.square(tfp.stats.correlation(y_true, y_pred))

def inverse_correlation_loss(y_true: Any, y_pred: Any) -> tf.Tensor:
    """Pearson correlation squared between two batches.

    Arguments
    ---------
    y_true : Any
    Taget values of the used network.

    y_predicted : Any
    Predictions made by the used model.

    Returns
    ---------
    One minus correlation squared (range between 0 and 1) as TensorFlow tensor.
    """
    return 1 - backend.square(tfp.stats.correlation(y_true, y_pred))
