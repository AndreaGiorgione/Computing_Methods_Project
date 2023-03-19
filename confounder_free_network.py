
'''Module containing the model for confounder free
machine learning application.'''

from pathlib import Path

import argparse
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
                 regressor_output_dim: int, learning_rate: float,
                 verbose=False):
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
                                            optimizer=Adam(learning_rate=learning_rate),
                                            metrics=['accuracy'])

        self.resgression_network = Sequential()
        self.resgression_network.add(self.extractor) # Non-Trainable
        self.resgression_network.add(self.regressor) # Trainable
        self.resgression_network.compile(loss='mse',
                                        optimizer=Adam(learning_rate=learning_rate))

        self.extraction_network = Sequential()
        self.extraction_network.add(self.extractor) # Trainable
        self.extraction_network.add(self.regressor) # Non-Trainable
        self.extraction_network.compile(loss=correlation_coefficient_loss,
                                        optimizer=Adam(learning_rate=learning_rate))

        if verbose:
            self.classification_network.summary()
            self.resgression_network.summary()
            self.extraction_network.summary()

    def train(self, dataset: np.ndarray, epochs: int, batch_size: int,
              validation_fraction: float, test_fraction: float,
              labels_indeces: int, confound_indeces: int,
              test=False, verbose=False):
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
        developement_data = dataset[:int(len(dataset)*(1-test_fraction)), :]
        test_data = dataset[int(len(dataset)*test_fraction):, :]

        train_data = developement_data[:int(len(dataset)*(1-validation_fraction)), :]
        validation_data = developement_data[int(len(dataset)*validation_fraction):, :]

        train_set = train_data[:, 0:labels_indeces[0]]
        validation_set = validation_data[:, 0:labels_indeces[0]]
        test_set = test_data[:, 0:labels_indeces[0]]

        train_labels = train_data[:, labels_indeces]
        validation_labels = validation_data[:, labels_indeces]
        test_labels = test_data[:, labels_indeces]

        train_confounders = train_data[:, confound_indeces]
        validation_confounders = validation_data[:, confound_indeces]
        test_confounders = test_data[:, confound_indeces]

        print(train_confounders[1:10])

        # Definition of the batches
        train_batches = np.array_split(train_set, batches_number)
        train_labels_batches = np.array_split(train_labels, batches_number)
        train_confounders_batches = np.array_split(train_confounders, batches_number)

        # Pretraining of the classification network
        self.classification_network.fit(x=train_set, y=train_labels,
                                        validation_data=[validation_set, validation_labels],
                                        epochs=int(epochs / 5), batch_size=batch_size)

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
            class_train_results = self.classification_network.evaluate(train_set,
                                                                       train_labels,
                                                                       verbose=0)
            class_validation_results = self.classification_network.evaluate(validation_set,
                                                                            validation_labels,
                                                                            verbose=0)
            pred_train_results = self.resgression_network.evaluate(train_set,
                                                                   train_confounders,
                                                                   verbose=0)
            pred_validation_results = self.resgression_network.evaluate(validation_set,
                                                                        validation_confounders,
                                                                        verbose=0)

            # Show results
            print(f'Epoch {epoch+1} of {epochs}: {class_train_results}, {class_validation_results}')
            if verbose:
                print(f'                   {pred_train_results}, {pred_validation_results}')
        
        if test:
            class_test_results = self.classification_network.evaluate(test_set,
                                                                      test_labels,
                                                                      verbose=0)
            pred_test_results = self.resgression_network.evaluate(test_set,
                                                                  test_confounders,
                                                                  verbose=0)
            print(f'Final assesment on test: {class_test_results}, {pred_test_results}')

if __name__ == "__main__":

    # Definition of the parser for the user arguments
    parser = argparse.ArgumentParser(description='Program for a confounder free network.',
                                     epilog='Be sure choosen params are suited.')

    parser.add_argument('-dp', type=str, metavar='', required=True,
                        help='Pathname of the dataset (format xlsx).')

    parser.add_argument('-el', type=int, metavar='', required=True,
                        help='Number of layers for the features extractor.')
    parser.add_argument('-cl', type=int, metavar='', required=True,
                        help='Number of layers for the classificator.')
    parser.add_argument('-rl', type=int, metavar='', required=True,
                        help='Number of layers for the regressor.')

    parser.add_argument('-en', type=int, metavar='', required=True,
                        help='Number of neurons for the features extractor.')
    parser.add_argument('-cn', type=int, metavar='', required=True,
                        help='Number of neurons for the classificator.')
    parser.add_argument('-rn', type=int, metavar='', required=True,
                        help='Number of neurons for the regressor.')

    parser.add_argument('-co', type=int, metavar='', required=True,
                        help='Output dimension for the classificator.')
    parser.add_argument('-ro', type=int, metavar='', required=True,
                        help='Output dimension for the regressor.')

    parser.add_argument('-e', type=int, metavar='',  required=True, 
                        help='Number of epochs.')
    parser.add_argument('-b', type=int, metavar='',  required=True, 
                        help='Batch size.')
    parser.add_argument('-v', type=float, metavar='', required=True,
                        help='Validation fraction.')
    parser.add_argument('-t', type=float, metavar='', required=True,
                        help='Test fraction.')
    parser.add_argument('-l', type=int, nargs='+', metavar='', required=True,
                        help='Labels indeces.')
    parser.add_argument('-c', type=int, nargs='+', metavar='', required=True,
                        help='Cnfounders indeces.')
    parser.add_argument('-lr', type=float, metavar='', required=True,
                        help='Leraning rate.')

    parser.add_argument('-ts', action='store_true',
                        help='Final assesment over the test set.')
    parser.add_argument('-vr', action='store_true',
                        help='Print some info during the run.')

    args = parser.parse_args()

    # Import of the dataset (first as pandas dataframe and then as numpy array)
    data = pd.read_excel(Path(args.dp))
    data = np.array(data)

    # Changing the -1 labels with 0 dor the binary crossentropy usage
    data[data == -1] = 0

    # Prepare the model
    model = ConfounderFreeNetwork(input_dim=(data.shape[1]-args.co), extractor_layers=args.el,
                                  classificator_layers=args.cl, regressor_layers=args.rl,
                                  extractor_neurons=args.en, classificator_neurons=args.cn,
                                  regressor_neurons=args.rn, classificator_output_dim=args.co,
                                  regressor_output_dim=args.ro, learning_rate=args.lr,
                                  verbose=args.v)

    # Train model
    model.train(dataset=data, epochs=args.e, batch_size=args.b,
                validation_fraction=args.v, test_fraction=args.t,
                labels_indeces=args.l, confound_indeces=args.c,
                test=args.ts, verbose=args.v)
