# Computing_Methods_Project
Project for the exam of Computing Methods for Experimental Physics and Data Science. It consist in a machiene learning application over a medical dataset: the task is predicting if a subject is affetced by autism or not on the base of some numerical features.

## General description
The project consist in the creation in Python of a model known in leterature as confounder-free network. It has been implemented with the OOP paradigm, creating a new class representing the network itself. The reason behind this choice is the necessity to hold together the different component of the model and manage its functionalities. The aim of the implementation is to show the improvement of the proposed model with reference to a simpler neural network. The used data are stored in Data folder, and the original dataset is the FS_features_ABIDE_male.csv file.

## Python code
The actual implementation of the model of interest is in confounder_free_network.py, while the comparison simple model is in simple_network.py file. The confounder-free network is composed by three subnetworks defined in subnetworks.py, with the definition in custom_loss_function.py of a couple of target function for one of the subnetworks. Some unittests can be found in the folder Tests. Data pre-processing is also needed for the application and in harmonization.py the hrmonization of the dataset is implemented. This last script works together with the matlab livescripts descibed in the following section.

## Matlab code
In Data_preparation.mlx is defined a pipeline for the preparation of the dataset for the machine learning implementation. It consist in outliers removal and features reduction with usage of statistical tools. In it is called the harmonizaion.py script. This generated: ABIDE_prepared_dataset.xlsx, ABIDE_harmonized_dataset.xlsx (by the Python script) and ABIDE_final_dataset.xlsx (the actual one used for the analysis). Also another file called Data_exploration.mlx is present. It is just a preliminar check over the original dataset, with a first attempt of data preparation (it is just a test file not mandatory for the application).

## Requirements
  Here a list of installed Python packages:
  - pathlib 1.0.1
  - numpy 1.23.3
  - pandas 1.5.2
  - tensorflow 2.11.0
  - keras 2.11.0
  - scikit-learn 1.2.1
  - matplotlib 3.6.0
  - pylint 2.15.3
 
  Matlab version: 
  - R2022b

## How to run he code
To run the training of the model, user pass thanks to argparse the choosen dataset (excel format with samples corresponding to lines and fetures to columns) and the architectures of the network in terms of number of inputs, number of outputs, number of hidden layers, number of neurons for them, indexes of confounder variables and indexes of labels (dataset should have in the last part of the columns confounders and then labels). 
![ray-so-export](https://user-images.githubusercontent.com/113907653/232740719-8e1cccb4-b2d0-4000-b78a-384e60700d33.png)
