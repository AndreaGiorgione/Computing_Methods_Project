
"""Module containing a classic fully
connected neural network."""

from pathlib import Path

import argparse
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.optimizers import Adam

from confounder_free_network import build_extractor, build_classificator

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

    parser.add_argument('-en', type=int, metavar='', required=True,
                        help='Number of neurons for the features extractor.')
    parser.add_argument('-cn', type=int, metavar='', required=True,
                        help='Number of neurons for the classificator.')

    parser.add_argument('-co', type=int, metavar='', required=True,
                        help='Output dimension for the classificator.')

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

    # Splitting of the dataset
    developement_data = data[:int(len(data)*(1-args.t)), :]
    test_data = data[int(len(data)*(1-args.t)):, :]

    train_data = developement_data[:int(len(data)*(1-args.v)), :]
    validation_data = developement_data[int(len(data)*(1-args.v)):, :]

    train_set = train_data[:, 0:args.l[0]]
    validation_set = validation_data[:, 0:args.l[0]]
    test_set = test_data[:, 0:args.l[0]]

    train_labels = train_data[:, args.l]
    validation_labels = validation_data[:, args.l]
    test_labels = test_data[:, args.l]

    # Building of the subnewtworks
    extractor = build_extractor(input_dim=(data.shape[1]-args.co),
                                layers=args.el,
                                neurons=args.en)

    classificator = build_classificator(input_dim=(data.shape[1]-args.co),
                                layers=args.cl,
                                neurons=args.cn,
                                output_dim=args.co)

    # Prepare the model
    classification_network = Sequential()
    classification_network.add(extractor) # Trainable
    classification_network.add(classificator) # Trainable
    classification_network.compile(loss='binary_crossentropy',
                                    optimizer=Adam(learning_rate=args.lr),
                                    metrics=['accuracy'])
    
    # If required show model details
    if args.vr:
        classification_network.summary()

    # Training
    classification_network.fit(x=train_set, y=train_labels,
                               validation_data=[validation_set, validation_labels],
                               epochs=args.e, batch_size=args.b)
    
    # If required show the test results
    if args.ts:
        class_test_results = classification_network.evaluate(test_set,
                                                             test_labels,
                                                             verbose=0)
        
        print(f'Final assesment on test: {class_test_results}')
