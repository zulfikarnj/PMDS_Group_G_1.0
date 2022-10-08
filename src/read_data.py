# Import related packages
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

"""
Script with the purpose of reading data, 
changing the column name, and separate data
since the column name of the raw data itself is
really annoying    
"""

def read_data(path,
              save_file = True,
              return_file = True):
    """
    Function to open .csv files and rename the column's name

    Parameters
    -----------
    path        : str   - Dataset path
    save_file   : bool  - If true, will save dataframe file in pickle
    return_file : bool  - If true, will do data return              
    
    Return
    -------
    data    : pandas dataframe  - dataframe from pandas environment
    """
    
    # Read data
    data = pd.read_csv(path)
    
    # Changing columns name
    col_names_change = {
    "Weekly Cases" : "WeekCase",
    "Weekly Cases per Million" : "WeekCasePerMil",
    "Weekly Deaths" : "WeekDeath",
    "Weekly Deaths per Million" : "WeekDeathPerMil",
    "Total Vaccinations" : "TotalVac",
    "People Vaccinated" : "PeopleVac",
    "People Fully Vaccinated" : "PeopleFullVac",
    "Total Boosters" : "TotalBoost",
    "Daily Vaccinations" : "DailyVac",
    "Total Vaccinations per Hundred" : "TotalVacPerHun",
    "People Vaccinated per Hundred" : "PeopleVacPerHun",
    "People Fully Vaccinated per Hundred" : "PeopleFullVacPerHun",
    "Total Boosters per Hundred" : "TotalBoostPerHun",
    "Daily Vaccinations per Hundred" : "DailyVacPerHun",
    "Daily People Vaccinated" : "DailyPeopleVac",
    "Daily People Vaccinated per Hundred" : "DailyPeopleVacPerHun",
    "Next Week's Deaths" : "NWD"
    }
    
    data = data.rename(columns = col_names_change)
    
    #Bagian dump ini bisa tidak diikutkan
    if save_file:
        joblib.dump(data, "output/read_data/data_original.pkl")
    
    if return_file:
        return data
    
def split_input_output(dataset, target_column, save_file=True, return_file=True):
    """
    Function to separate dataset to input & output (based on target_column)

    Parameters
    -----------
    dataset         : pandas dataframe  - Dataset
    target_column   : str               - nama kolom yang jadi output
    save_file       : bool              - Apabila true, akan melakukan saving file dataframe dalam pickle
    return_file     : bool              - Apabila true, akan melakukan return data              
    
    Return
    -------
    input_df        : pandas dataframe  - dataframe input
    output_df       : pandas dataframe  - dataframe output
    """
    output_df = dataset[target_column]
    input_df = dataset.drop([target_column], axis=1)    # drop kolom target

    
    #Bagian dump ini bisa tidak diikutkan
    if save_file:
        joblib.dump(input_df, "output/read_data/input_df.pickle")
        joblib.dump(output_df, "output/read_data/output_df.pickle")

    if return_file:
        return input_df, output_df
    
def split_train_validation(input_df, output_df, save_file=True, return_file=True, test_size=0.2):
    """
    Fungsi untuk memisahkan dataset training menjadi training dataset & validation dataset
    untuk kebutuhan validasi, dengan perbandingan test_size = validation_dataset/total_dataset

    Parameters
    -----------
    input_df    : pandas dataframe  - dataframe input
    output_df   : pandas dataframe  - dataframe output
    save_file   : bool              - Apabila true, akan melakukan saving file dataframe dalam pickle
    return_file : bool              - Apabila true, akan melakukan return data  

    Return
    -------
    X_train           : pandas dataframe  - dataframe training input
    X_validation      : pandas dataframe  - dataframe validation input
    y_train           : pandas dataframe  - dataframe training output
    y_validation      : pandas dataframe  - dataframe validation output
    """
    # Copy data biar tidak terjadi aliasing
    X = input_df.copy()
    y = output_df.copy()

    # Split data
    # Random state = 123 untuk mempermudah duplikasi riset
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, 
                                                                    test_size=test_size,
                                                                    random_state=123)

    #Bagian dump ini bisa tidak diikutkan
    if save_file:
        joblib.dump(X_train, "output/split_data/X_train.pkl")
        joblib.dump(X_validation, "output/split_data/X_valid.pkl")
        joblib.dump(y_train, "output/split_data/y_train.pkl")
        joblib.dump(y_validation, "output/split_data/y_valid.pkl")

    if return_file:
        return X_train, X_validation, y_train, y_validation