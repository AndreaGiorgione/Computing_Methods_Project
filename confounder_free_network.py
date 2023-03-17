
'''Module containing the model for confounder free
machine learning application.'''

from pathlib import Path

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.optimizers import Adam

from subnetworks import build_classificator, build_extractor, build_regressor
from custom_loss_functions import correlation_coefficient_loss

class ConfounderFreeNetwork():
    """Deep model for a classification approach independent form biases. It is
    composed by a features extractor network, a classifator and a bias predictor.

    The first is a sort of autoencoder calculating a modified version of the input
    data, while the second is the actual model for the classification problem. The
    latest is a regressor attempting to recreate a subset of the input variables,
    supposed to be the misleading features.

    The computed features of the first subnetwork are the input for the other.
    """
    def __init__(self, input_dim: int, extractor_layers: int,
                 classificator_layers: int, regressor_layers: int,
                 extractor_neurons: int, classificator_neurons: int,
                 regressor_neurons: int, classificator_output_dim: int,
                 regressor_output_dim: int, verbose=0):
        """Network constructor.

        Arguments
        ---------
        input_dim : int
        Number of input neurons.

        extractor_layers : int
        Number of hidden layers of the extractor network.

        classification_layers : int
        Number of hidden layers of the classification network.

        predictor_layers : int
        Number of hidden layers of the predictor network.

        extractor_neurons : int
        Number of neurons in the hidden layers of the extractor network.

        classificator_neurons : int
        Number of neurons in the hidden layers of the classificator network.

        predictor_neurons : int
        Number of neurons in the hidden layers of the predictor network.
        """
        # Building of the subnewtworks
        self.extractor = build_extractor(input_dim, extractor_neurons,
                                         extractor_layers)

        self.classificator = build_classificator(input_dim, classificator_neurons,
                                                 classificator_layers, classificator_output_dim)

        self.regressor = build_regressor(input_dim, regressor_neurons,
                                                 regressor_layers, regressor_output_dim)

        # Definition of the submodels
        self.classification_network = Sequential()
        self.classification_network.add(self.extractor) # Trainable
        self.classification_network.add(self.classificator) # Trainable
        self.classification_network.compile(loss='binary_crossentropy',
                                            optimizer=Adam(learning_rate=5.0e-3),
                                            metrics=['accuracy'])

        self.resgression_network = Sequential()
        self.resgression_network.add(self.extractor) # Non-Trainable
        self.resgression_network.add(self.regressor) # Trainable
        self.resgression_network.compile(loss='mse',
                                        optimizer=Adam(learning_rate=5.0e-3))

        self.extraction_network = Sequential()
        self.extraction_network.add(self.extractor) # Trainable
        self.extraction_network.add(self.regressor) # Non-Trainable
        self.extraction_network.compile(loss=correlation_coefficient_loss,
                                        optimizer=Adam(learning_rate=5.0e-3))

        if verbose == 1:
            self.classification_network.summary()
            self.resgression_network.summary()
            self.extraction_network.summary()

    def train(self, dataset: np.ndarray, epochs: int, batch_size: int, test_fraction=0.25):
        """Method for the training of the model.

        Arguments
        ---------
        dataset : np.ndarray
        Dataset for the machine learning application (last columns must be the labels vector).

        epochs : int
        Number of epochs.

        batch_dimension : int
        Number of samples in every batch.

        test_fraction : float
        Percentage of the dataset for the model assesment (test set).
        """

        # Number of batches
        batches_number = int(dataset.shape[0] / batch_size)

        # Splitting of the dataset
        train_data = dataset[:int(len(dataset)*(1-test_fraction)), :]
        test_data = dataset[int(len(dataset)*test_fraction):, :]

        train_set = train_data[:, 0:-1]
        test_set = test_data[:, 0:-1]

        train_labels = train_data[:, -1]
        test_labels = test_data[:, -1]

        train_confounders = train_data[:, -4:-1]
        test_confounders = test_data[:, -4:-1]

        # Definition of the batches
        train_batches = np.array_split(train_set, batches_number)
        train_labels_batches = np.array_split(train_labels, batches_number)
        train_confounders_batches = np.array_split(train_confounders, batches_number)

        # Pretraining of the classification network
        self.classification_network.fit(train_set, train_labels,
                                        validation_data=[test_set, test_labels],
                                        epochs=200,
                                        batch_size=100)

        # Training of each model for the selected number of epoch
        for epoch in range(epochs):

            # Training each subnetwork on every batch
            for train_batch, train_labels_batch, train_confounders_batch \
                in zip(train_batches, train_labels_batches, train_confounders_batches):

                self.extractor.trainable = True
                self.classificator.trainable = True
                self.classification_network.train_on_batch(train_batch, train_labels_batch)

                self.extractor.trainable = False
                self.regressor.trainable = True
                self.resgression_network.train_on_batch(train_batch, train_confounders_batch)

                self.extractor.trainable = True
                self.regressor.trainable = False
                self.extraction_network.train_on_batch(train_batch, train_confounders_batch)

            # Summary of the results for every epoch
            class_train_results = self.classification_network.evaluate(train_set, train_labels,
                                                                       verbose=0)
            class_test_results = self.classification_network.evaluate(test_set, test_labels,
                                                                      verbose=0)
            pred_train_results = self.resgression_network.evaluate(train_set, train_confounders,
                                                                   verbose=0)
            pred_test_results = self.resgression_network.evaluate(test_set, test_confounders,
                                                                  verbose=0)

            # Show results
            print(f'Epoch {epoch+1} of {epochs}: {class_train_results}, {class_test_results}')
            print(f'                             {pred_train_results}, {pred_test_results}')

if __name__ == "__main__":

    # Import of the dataset (first as pandas dataframe and then as numpy array)
    data = pd.read_excel(Path('Data/ABIDE_modified_dataset.xlsx'))
    data = np.array(data)

    # Changing the -1 labels with 0 dor the binary crossentropy usage
    data[data == -1] = 0

    # Prepare the model
    model = ConfounderFreeNetwork(input_dim=(data.shape[1]-1), extractor_layers=35,
                                  classificator_layers=15, regressor_layers=8,
                                  extractor_neurons=10, classificator_neurons=10,
                                  regressor_neurons=5, classificator_output_dim=1,
                                  regressor_output_dim=3, verbose=1)

    # Train model
    model.train(dataset=data, epochs=400, batch_size=100, test_fraction=0.2)
