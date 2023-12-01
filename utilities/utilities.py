import numpy as np
import argparse
import pandas as pd
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor as vif)


""" Data cleanup utilities """
def fillna_by_number_dtype(df):
    """
    This is handles the WHO Life Expectancy data specifically. 
    It could definitely be generalized see if a dtype is any 
    type of int, etc.
    """
    for col in df:
        data_type = df[col].dtype
        if data_type == np.int64:
            df[col] = df[col].fillna(
                df[col].median().round(0).astype(np.int64))
        elif data_type == np.float64:
            df[col] = df[col].fillna(df[col].median())


""" Data exploration utilities """
# Modifed from: https://towardsdatascience.com/
# targeting-multicollinearity-with-python-3bd3b4088d0b
# Compute the vif for all given features
def compute_vif(df, considered_features):
    X = df.reindex(considered_features, axis=1)
    fillna_by_number_dtype(X)
    # The calculation of variance inflation requires a constant
    X['intercept'] = 1
    # Create dataframe to store vif values
    df_vif = pd.DataFrame()
    df_vif["Variable"] = X.columns
    df_vif["VIF"] = [vif(X.values, i) for i in range(X.shape[1])]
    df_vif = df_vif[df_vif['Variable']!='intercept']
    return df_vif

""" Model training utilities """
def get_model_training_args():
    """
    Get arguments for training the models
    --selected_models abbreviations:
        rr: Ridge regression
        rfr: Random forest regression
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_dataset', default='who_life_expectancy',
                        help='Choose a dataset for the models to learn.')
    parser.add_argument('--target_variable', default='Life expectancy',
                        help='Choose the target variable.')
    parser.add_argument('--variables_to_drop', default='',
                        help='Variables to exclude during training.')
    parser.add_argument('--selected_models', default='rfr',
                        help='Choose which models to train.')
    parser.add_argument('--train_with_covar', action='store_true',
                        help='Train a model including the covariates.')
    parser.add_argument('--train_with_shuffled_data', action='store_true',
                        help='Train models on a shuffled dataset.')
    parser.add_argument('--output_suffix', default='',
                        help='Add suffix to output file')
    parser.add_argument('--timestamp', action='store_true',
                        help='Timestamp the results directory.')
    parser.add_argument('--n_jobs', default=1, type=int,
                        help='Number of threads to use during training.')
    args = parser.parse_args()
    return args