# Import libraries and packages

import numpy as np
import pandas as pd
import src.model_lib as mlib
import time
import joblib
import yaml

def main(params):
    '''
    Main function of modelling
    
    Parameters
    ----------
    params: .yaml file contain (dict) of general parameters for the read_data and model_lib function
        - DUMP_TRAIN (str)  : location of preprocessed training data pickle
        - Y_PATH_TRAIN (str): location of target column pickle for training data
        - DUMP_VALID (str)  : location of preprocessed validation data pickle
        - Y_PATH_VALID (str): location of target column  pickle validation data

        - target(str) : y column to be used   
        - scoring(str) : sklearn cross-val scoring scheme
        - n_iter_search : RandomizedSearchCV number of iteration

    '''
    # Make a dictionary "train_log_dict" to be saved later as pickle containing model information in training stage
    train_log_dict = {'model': ['xgb', 'rf'],
                      'model_name': [],
                      'model_fit': [],
                      'model_report': [],
                      'model_score': [],
                      'fit_time': []}
    
    # Read data after preprocessing
    x_train, y_train, x_valid, y_valid  = mlib.read_data(params)
    
    # Iterate list model
    #for model in train_log_dict['model']:
    '''first model'''
    # initiate the model
    param_model_1, base_model_1 = mlib.model_xgb()
        # logging model name
    train_log_dict['model_name'].append(base_model_1.__class__.__name__)
    print(
       f'Fitting {base_model_1.__class__.__name__}')

        # Training
    t0 = time.time()
        
    # Searching best parameter using Random Search CV
    fitted_model, best_estimator = mlib.fitting(
        x_train, y_train, base_model_1, param_model_1, params)
    elapsed_time = time.time() - t0
    print(f'elapsed time: {elapsed_time} s \n')
    train_log_dict['fit_time'].append(elapsed_time)
    train_log_dict['model_fit'].append(best_estimator.__class__.__name__)
        
    # Fitting model with best params to data training
    best_estimator.fit(x_train, y_train)
    train_log_dict['model_report'].append(best_estimator)

        
    # Validate model to validation data
    score = mlib.validation_score( x_valid, y_valid, best_estimator)
    train_log_dict['model_score'].append(score)
        
    ''' second model'''
    # initiate the model
    param_model_2, base_model_2 = mlib.model_rf()
    # logging model name
    train_log_dict['model_name'].append(base_model_2.__class__.__name__)
    print(
       f'Fitting {base_model_2.__class__.__name__}')

    # Training
    t1 = time.time()
        
    # Searching best parameter using Random Search CV
    fitted_model, best_estimator = mlib.fitting(
        x_train, y_train, base_model_2, param_model_2, params)
    elapsed_time = time.time() - t1
    print(f'elapsed time: {elapsed_time} s \n')
    train_log_dict['fit_time'].append(elapsed_time)
    train_log_dict['model_fit'].append(best_estimator.__class__.__name__)
        
    # Fitting model with best params to data training
    best_estimator.fit(x_train, y_train)
    train_log_dict['model_report'].append(best_estimator)

        
    # Validate model to validation data
    score = mlib.validation_score( x_valid, y_valid, best_estimator)
    train_log_dict['model_score'].append(score)

    # Select which model in model list has best score evaluation (minimum rmse) in validation data
    best_model, best_estimator, best_report = mlib.select_model(
        train_log_dict)
    print(
        f"Model: {best_model}, Score: {best_report}, Parameter: {best_estimator}")
    
    # Dump model name
    joblib.dump(best_model, f'output/model/train/model_name_v1.1.pkl')
    # Dump best model estimator with best param
    joblib.dump(best_estimator, 'output/model/train/best_estimator_v1.1.pkl')
    # Dump training log
    joblib.dump(train_log_dict, 'output/model/train/train_log_v1.1.pkl')