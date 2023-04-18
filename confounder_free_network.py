
"""Module containing the model for confounder free
machine learning application."""

from pathlib import Path

import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.optimizers import Adam
from keras.metrics import AUC

from sklearn.model_selection import KFold

from statistics import mean

from subnetworks import build_classificator, build_extractor, build_regressor
from custom_loss_functions import correlation_coefficient_loss

class ConfounderFreeNetwork():
    """Deep model for a classification approach independent fromm biases. It is
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
                 regressor_output_dim: int, verbose=False):
        """Network constructor.

        Arguments
        ---------
        input_dim : int
        Number of input neurons.

        extractor_layers : int
        Number of hidden layers of the extractor network.

        classification_layers : int
        Number of hidden layers of the classification network.

        regressor_layers : int
        Number of hidden layers of the predictor network.

        extractor_neurons : int
        Number of neurons in the hidden layers of the extractor network.

        classificator_neurons : int
        Number of neurons in the hidden layers of the classificator network.

        regressor_neurons : int
        Number of neurons in the hidden layers of the regressor network.

        classificator_output_dim : int
        Number of output neurons in the classificator.

        regressor_output_dim : int
        Number of output neurons in the regressor.

        verbose : bool
        If True print some useful information about the model.
        """
        # Building of the subnewtworks
        self.extractor = build_extractor(input_dim, extractor_layers,
                                         extractor_neurons)

        self.classificator = build_classificator(input_dim, classificator_layers,
                                                 classificator_neurons,
                                                 classificator_output_dim)

        self.regressor = build_regressor(input_dim, regressor_layers,
                                         regressor_neurons,
                                         regressor_output_dim)

        # Definition of the submodels
        self.classification_network = Sequential()
        self.classification_network.add(self.extractor) # Trainable
        self.classification_network.add(self.classificator) # Trainable
        self.classification_network.compile(loss='binary_crossentropy',
                                            optimizer=Adam(),
                                            metrics=AUC())

        self.resgression_network = Sequential()
        self.resgression_network.add(self.extractor) # Non-Trainable
        self.resgression_network.add(self.regressor) # Trainable
        self.resgression_network.compile(loss='mse',
                                        optimizer=Adam())

        self.extraction_network = Sequential()
        self.extraction_network.add(self.extractor) # Trainable
        self.extraction_network.add(self.regressor) # Non-Trainable
        self.extraction_network.compile(loss=correlation_coefficient_loss,
                                        optimizer=Adam())

        if verbose:
            self.classification_network.summary()
            self.resgression_network.summary()
            self.extraction_network.summary()

    def train(self, train_data: np.ndarray, validation_data: np.ndarray,
              epochs: int, batch_size: int, labels_indexes: int,
              confound_indexes: int, verbose=False):
        """Method for the training of the model.

        Arguments
        ---------
        train_data : np.ndarray
        Dataset for the model training (last columns must be the labels).

        validation_data : np.ndarry
        Dataset for the model validation (last columns must be the labels).

        epochs : int
        Number of epochs.

        batch_size : int
        Number of samples in every batch.

        labels_indexes: int
        Indexes of the classification labels.

        confound_indexes: int
        Indexes of the confounder variables.

        verbose : bool
        If True print the results of the regressor.
        """

        # Number of batches
        batches_number = int(train_data.shape[0] / batch_size)

        # Splitting of the data
        train_set = train_data[:, 0:labels_indexes[0]]
        validation_set = validation_data[:, 0:labels_indexes[0]]

        train_labels = train_data[:, labels_indexes]
        validation_labels = validation_data[:, labels_indexes]

        train_confounders = train_data[:, confound_indexes]
        validation_confounders = validation_data[:, confound_indexes]

        # Definition of the batches
        train_batches = np.array_split(train_set, batches_number)
        train_labels_batches = np.array_split(train_labels, batches_number)
        train_confounders_batches = np.array_split(train_confounders, batches_number)

        # Pretraining of the classification network
        self.classification_network.fit(x=train_set, y=train_labels,
                                        validation_data=[validation_set, validation_labels],
                                        epochs=int(epochs / 5), batch_size=batch_size)

        train_results = []
        validation_results = []

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
            
            train_results.append(class_train_results[1])
            validation_results.append(class_validation_results[1])

            # Show results
            if (epoch+1) % 1 == 0:
                print(f'Epoch {epoch+1} of {epochs}:', end=' ')
                print(f'loss {round(class_train_results[0], 7)}', end=' - ')
                print(f'auc {round(class_train_results[1], 7)}', end=' - ')
                print(f'val_loss {round(class_validation_results[0], 7)}', end=' - ')
                print(f'val_auc {round(class_validation_results[1], 7)}')
                if verbose:
                    print('                 ', end=' ')
                    print(f'regr_loss {round(pred_train_results, 7)}', end=' - ')
                    print(f'regr_val_loss {round(pred_validation_results, 7)}', end='')
                print('\n')

        # Showing preformances
        print(f'Train AUC max and mean: {max(train_results)}, {mean(train_results)}')
        print(f'Validation AUC max and mean: {max(validation_results)}, {mean(validation_results)}')

        plt.plot(np.linspace(1, epochs, epochs),
                 train_results, color='blue',
                 label='Train AUC')
        plt.plot(np.linspace(1, epochs, epochs),
                 validation_results, color='red',
                 label='Validation AUC')
        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        plt.title('Learning curves')
        plt.legend(loc='lower right')
        plt.show()

    def assesment(self, test_set, labels_indexes: int, confound_indexes: int):
        """Method for the model assesment on test set.

        Arguments
        ---------
        test_set : np.array
        Test set for the evalutaion phase.

        labels_indexes : int
        Indexes of the classification lables.

        confound_indexes : int
        Indexes of the confounder variables.
        """

        # Splitting the data
        test_set = test_set[:, 0:labels_indexes[0]]
        test_labels = test_set[:, labels_indexes]
        test_confounders = test_set[:, confound_indexes]

        # Prediction of the network
        class_test_results = self.classification_network.evaluate(test_set,
                                                                  test_labels,
                                                                  verbose=0)
        pred_test_results = self.resgression_network.evaluate(test_set,
                                                              test_confounders,
                                                              verbose=0)

        # Print of results
        print(f'Final assesment on test: {class_test_results}, {pred_test_results}')

    def extracted_features(self, input_data: np.ndarray) -> np.ndarray:
        """Method for the visualization of the
        features extractor behaviour.
        
        Arguments
        ---------
        input_data : npndarray
        Samples to be investigated.

        Returns
        ---------
        modified_input : np.ndarray
        New version of the samples generated by theextractor
        and used by classificator and regressor.
        """

        # Expolit predict method of the extractor
        modified_input = self.extractor.predict(input_data)
        return modified_input
    
    def prediction(self, input_data: np.ndarray) -> np.ndarray:
        """Method for the prediction of the
        classification network on new samples
        
        Arguments
        ---------
        input_data : np.ndarray
        Samples to be predicted.

        Returns
        predictions : np.ndarray
        Labesl computed by the model.
        """

        # Expolit predict method of the classification network
        predictions = self.classification_network.predict(input_data)
        return predictions

if __name__ == "__main__":

    # Definition of the parser for the user arguments
    parser = argparse.ArgumentParser(description='Confounder free network program.',
                                     epilog='Be sure choosen params are suited.')

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                    help='Show some help information and terminate.')

    parser.add_argument('-dp', type=str, metavar='', required=True,
                        help='Pathname of the dataset (format xlsx).')

    parser.add_argument('-el', type=int, metavar='', required=True,
                        help='Number of layers of the features extractor.')
    parser.add_argument('-cl', type=int, metavar='', required=True,
                        help='Number of layers of the classificator.')
    parser.add_argument('-rl', type=int, metavar='', required=True,
                        help='Number of layers of the regressor.')

    parser.add_argument('-en', type=int, metavar='', required=True,
                        help='Number of neurons per layer of the features extractor.')
    parser.add_argument('-cn', type=int, metavar='', required=True,
                        help='Number of neurons per layer of the classificator.')
    parser.add_argument('-rn', type=int, metavar='', required=True,
                        help='Number of neurons per layer of the regressor.')

    parser.add_argument('-co', type=int, metavar='', required=True,
                        help='Output dimension of the classificator.')
    parser.add_argument('-ro', type=int, metavar='', required=True,
                        help='Output dimension of the regressor.')

    parser.add_argument('-e', type=int, metavar='',  required=True,
                        help='Number of epochs.')
    parser.add_argument('-b', type=int, metavar='',  required=True,
                        help='Batch size.')
    parser.add_argument('-t', type=float, metavar='', required=True,
                        help='Test fraction.')
    parser.add_argument('-l', type=int, nargs='+', metavar='', required=True,
                        help='Labels indexes.')
    parser.add_argument('-c', type=int, nargs='+', metavar='', required=True,
                        help='Confounders indexes.')

    parser.add_argument('-ts', action='store_true',
                        help='Final assesment on test.')
    parser.add_argument('-vr', action='store_true',
                        help='Print some info during the run.')

    args = parser.parse_args()

    # Import of the dataset (first as pandas dataframe and then as numpy array)
    data = pd.read_excel(Path(args.dp))
    data = np.array(data)

    # Changing the -1 labels with 0 dor the binary crossentropy usage
    data[data == -1] = 0

    # Splitting the dataset
    developement_data = data[:int(len(data)*(1-args.t)), :]
    test_data = data[int(len(data)*(1-args.t)):, :]

    # If required check test performances
    if args.ts:
        # Prepare the model
        model = ConfounderFreeNetwork(input_dim=(data.shape[1]-args.co), extractor_layers=args.el,
                                  classificator_layers=args.cl, regressor_layers=args.rl,
                                  extractor_neurons=args.en, classificator_neurons=args.cn,
                                  regressor_neurons=args.rn, classificator_output_dim=args.co,
                                  regressor_output_dim=args.ro, verbose=args.vr)

        # Training
        model.train(train_data=developement_data,
                    validation_data=test_data,
                    epochs=args.e, batch_size=args.b, labels_indexes=args.l,
                    confound_indexes=args.c, verbose=args.vr)
        
        extracted_features = model.extracted_features(developement_data[[0, 4, 7], :-1])
        print(f'Raw samples: {developement_data[[0, 4, 7], :-1]}')
        print(f'Computation of the extractor: {extracted_features}')
        input('Press enter to perform the K-fold...')

    # Start of the computation time
    start = time.time()

    # K-Fold cross validation
    FOLDS = 4
    kf = KFold(n_splits=FOLDS, shuffle=False)
    for fold, (train_index, validation_index) in enumerate(kf.split(developement_data)):
        print(f'Fold {fold+1} of {FOLDS}')

        # Prepare the model
        model = ConfounderFreeNetwork(input_dim=(data.shape[1]-args.co), extractor_layers=args.el,
                                  classificator_layers=args.cl, regressor_layers=args.rl,
                                  extractor_neurons=args.en, classificator_neurons=args.cn,
                                  regressor_neurons=args.rn, classificator_output_dim=args.co,
                                  regressor_output_dim=args.ro, verbose=args.vr)

        # Training
        model.train(train_data=developement_data[train_index],
                    validation_data=developement_data[validation_index],
                    epochs=args.e, batch_size=args.b, labels_indexes=args.l,
                    confound_indexes=args.c, verbose=args.vr)

    # End of computation time
    end = time.time()
    print(f'Total time: {round((end - start) / 60)} minutes.')
