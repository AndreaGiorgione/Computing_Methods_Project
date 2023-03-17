
"""File containing some function for the creation of some
neural networks. Here three possibilities of models: a features
extractor in the form of a residual networks, a classificator and
a regressor. The extractor has the same shape in input and output
in order to exploit the concept of correction vectors."""

import keras
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Dense, Add

def build_extractor(input_dim: int, layers: int, neurons: int) -> keras.Model:
    """Function for the construction of an extractor. The network has a skip
    connection linking the input layer and the output layer.

    Arguments
    ---------
    input_dim : int
    Number of input neurons.

    layers : int
    Number of hidden layers in the network.

    neurons : int
    Number of neurons in each layer.

    Returns
    ---------
    extractor : keras.Model
    Extraction model (not compiled).
    """
    inputs = Input(shape=(input_dim))
    hidden = Dense(units=neurons, activation='relu', kernel_initializer= \
                   tf.random_uniform_initializer(minval=-0.3, maxval=0.3))(inputs)
    for _ in range(layers - 1):
        hidden = Dense(units=neurons, activation='relu', kernel_initializer= \
                       tf.random_uniform_initializer(minval=-0.3, maxval=0.3))(hidden)
    outputs = Dense(units=input_dim, activation='linear', kernel_initializer= \
                    tf.random_uniform_initializer(minval=-0.3, maxval=0.3))(hidden)
    added = Add()([inputs, outputs])
    extractor = Model(inputs, added)
    return extractor

def build_classificator(input_dim: int, layers: int, neurons: int, output_dim) -> keras.Model:
    """Function for the construction of a classificator.

    Arguments
    ---------
    input_dim : int
    Number of input neurons.

    layers : int
    Number of hidden layers in the network.

    neurons : int
    Number of neurons in each layer.

    output_dim : int
    Number of output neurons.

    Returns
    ---------
    extractor : keras.Model
    Extraction model (not compiled).
    """
    inputs = Input(shape=(input_dim))
    hidden = Dense(units=neurons, activation='relu', kernel_initializer= \
                   tf.random_uniform_initializer(minval=-0.3, maxval=0.3))(inputs)
    for _ in range(layers - 1):
        hidden = Dense(units=neurons, activation='relu', kernel_initializer= \
                       tf.random_uniform_initializer(minval=-0.3, maxval=0.3))(hidden)
    outputs = Dense(units=output_dim, activation='sigmoid', kernel_initializer= \
                    tf.random_uniform_initializer(minval=-0.3, maxval=0.3))(hidden)
    classificator = Model(inputs, outputs)
    return classificator

def build_regressor(input_dim: int, layers: int, neurons: int, output_dim) -> keras.Model:
    """Function for the construction of a regressor.

    Arguments
    ---------
    input_dim : int
    Number of input neurons.

    layers : int
    Number of hidden layers in the network.

    neurons : int
    Number of neurons in each layer.

    output_dim : int
    Number of output neurons.

    Returns
    ---------
    regressor : keras.Model
    Regression model (not compiled).
    """
    inputs = Input(shape=(input_dim))
    hidden = Dense(units=neurons, activation='relu', kernel_initializer= \
                   tf.random_uniform_initializer(minval=-0.3, maxval=0.3))(inputs)
    for _ in range(layers - 1):
        hidden = Dense(units=neurons, activation='relu', kernel_initializer= \
                       tf.random_uniform_initializer(minval=-0.3, maxval=0.3))(hidden)
    outputs = Dense(units=output_dim, activation='linear', kernel_initializer= \
                    tf.random_uniform_initializer(minval=-0.3, maxval=0.3))(hidden)
    regressor = Model(inputs, outputs)
    return regressor
