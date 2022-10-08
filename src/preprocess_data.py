"""
This script tends to have any wrangling features
includes impute nan value and standard scaling
"""

# Import package
import pandas as pd
import numpy as np
from numpy import nan
import joblib
import yaml

from sklearn.preprocessing import StandardScaler

f = open("src/params/params.yaml", "r")
params = yaml.load(f, Loader = yaml.SafeLoader)
f.close()

def numerical_imputer(numerical, state = 'transform'):
    '''
    Function to impute the nan value with 0
    Since most of the nan data came from
    The covid information that doesn't
    Available
    '''
    
    if state == 'fit':
        numerical[params['NUM_COLUMN']].replace(np.nan, 0)
        joblib.dump(numerical, "output/preprocess_data/estimator/numerical_imputer.pkl")
    elif state == 'transform':
        numerical = joblib.load("output/preprocess_data/estimator/numerical_imputer.pkl")
        numerical[params['NUM_COLUMN']].replace(np.nan, 0)
    
    return numerical

def normalize_input(input_data, state = "fit", save_file=True, return_file=True):
    """
    Function to do normalization

    Parameters
    -----------
    input_data      : pandas dataframe  - input data which we want to standardize
    state           : str               - fitting or transformation process
    save_file       : bool              - if True, will save to new dataframe
    return_file     : bool              - if True, will do return

    Return
    -------
    output_data     : pandas dataframe  - standardisasi result dataframe
    """
    # Save column
    column_ = input_data.columns

    # Make scaler
    
    if state == 'fit':
        scaler = StandardScaler()
        scaler.fit(input_data)
        joblib.dump(scaler,
                    "output/preprocess_data/estimator/data_normalize.pkl")
        
    elif state == 'transform':
        scaler = joblib.load("output/preprocess_data/estimator/data_normalize.pkl")

    # Do scaling
    output_data = scaler.transform(input_data)
    output_data = pd.DataFrame(data=output_data, columns=column_)

    if return_file:
        return output_data
    
def map_year(x):
    '''
    Mapping each year into categorical object
    
    parameter
    --------------
    x: int    - year
    
    output
    -------------
    x: str    - categorical
    '''
    if x == 2020:
        row = "0"
    elif x == 2021:
        row = "1"
    elif x == 2022:
        row = "2"
    return row

location_map = {'EU' : "1",
                'AS' : "2",
                'AF' : "3",
                'OC' : "4",
                'SA' : "5",
                'World' : "6",
                'NA' : "7",
                'High income' : "8",
                'Low income' : "9",
                'Upper middle income' : "10",
                'Lower middle income' : "11"
                }

def run(params, xpath, ypath, dump_path, state='fit'):
    '''
    Main function of wrangling and feature engineering.
    This function will applied in data training, testing and validation.
    
    Parameters
    ----------
    params: .yaml file
        File containing necessary variables as constant variable such as location file and features name 
        - PREDICT_COLUMN(str) : list of features to be used   
    xpath: string
        Location of features pickle file

    ypath: string
        Location of target pickle file

    dump_path: string
        Location to save the result of preprocessing

    state: string
        Data state for leakage handling. fit for training data, transform for validation and testing data

    '''
    
    # Load variables and target pickle file
    covid_variables = joblib.load(xpath)
    covid_target = joblib.load(ypath)
    
    # All features other than target and id will be included
    covid = covid_variables[params['TRAIN_COLUMN']]
    
    # Handling missing value
    covid_imputed = numerical_imputer(covid, state = state)
    
    # Transform entry for Location and Year columns
    covid_imputed['Location'] = covid_imputed['Location'].map(location_map)
    covid_imputed['Year'] = covid_imputed['Year'].apply(map_year)
    
    # Normalization
    covid_normalized = normalize_input(covid_imputed, state = state)
    joblib.dump(covid_normalized, dump_path)