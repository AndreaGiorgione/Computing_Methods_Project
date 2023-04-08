
"""File for the harmonization of the dataset
between acquisition sites."""

import numpy as np
import pandas as pd

from neuroHarmonize import harmonizationLearn

# Import the dataset as pandas dataframe
data = pd.read_excel('Data\ABIDE_prepared_dataset.xlsx')

# Prepare the numpy array of fetures and pandas dataframe of covariates
covariate_matrix = data[['SITE', 'AGE_AT_SCAN', 'DX_GROUP']]
features_dataframe = data.drop(['SITE', 'AGE_AT_SCAN', 'DX_GROUP'], axis=1)
features_array = np.array(features_dataframe)

# Harmonization of the data
model, harmonized_array = harmonizationLearn(features_array, covariate_matrix)

# Go back to dataframe
harmonized_features_dataframe = pd.DataFrame(harmonized_array, columns=list(features_dataframe.columns))

# Restore the dataset in the original format
newdata = pd.concat([covariate_matrix, harmonized_features_dataframe], axis=1)
newdata.to_excel('Data\ABIDE_harmonized_dataset.xlsx', index=False)
