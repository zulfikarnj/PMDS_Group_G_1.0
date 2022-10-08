# Python package
import numpy as np
import pandas as pd
import joblib
import yaml
import pycountry_convert

# File package
from src.read_data import read_data, split_input_output, split_train_validation
import src.preprocess_data as ppr
import src.training as train
import src.testing as tests

# Open yaml
f = open("src/params/params.yaml", "r")
params = yaml.load(f, Loader = yaml.SafeLoader)
f.close()

# Reading, rename columns, and splitting data
data_covid = read_data(params['DATA_PATH'])
data_covid = data_covid.replace(np.nan, 0)

input_df, output_df = split_input_output(data_covid,
                                         params['TARGET_COLUMN'])
X_train, X_valid, y_train, y_valid = split_train_validation(input_df,
                                                            output_df,
                                                            True,
                                                            params['TEST_SIZE'])

temp = ['TRAIN', 'VALID']

for subgroup in temp:
    print(f"Runnuing on feature engineering {subgroup}...")
    xpath = params[f"X_PATH_{subgroup}"]
    ypath = params[f"Y_PATH_{subgroup}"]
    dump_path = params[f"DUMP_{subgroup}"]
    
    if subgroup == 'TRAIN':
        state = 'fit'
    else:
        state = 'transform'
        
    ppr.run(params, xpath, ypath, dump_path, state)
    
# Training and Tuning
print(f"Running on training and hyperparameter tuning...")
train.main(params)

# Predicting and Last Evaluation
print(f"Last evaluation on test data")
tests.main(params)