import sys
import os
import datetime
import json
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, minmax_scale
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import SGDRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score
from joblib import dump

sys.path.append('utilities')
import utilities as utils

def main():
    args = utils.get_model_training_args()
    training_dataset = args.training_dataset
    target_variable = args.target_variable
    variables_to_drop = args.variables_to_drop.split(',')
    # selected_models = args.selected_models.split(',')
    train_with_covar = args.train_with_covar
    train_with_shuffled_data = args.train_with_shuffled_data
    output_suffix = args.output_suffix
    timestamp = args.timestamp
    n_jobs = args.n_jobs

    if not variables_to_drop[0].strip():
        variables_to_drop = []
    # if not selected_models[0].strip():
    #     selected_models = ['rfr']
    output_dir = training_dataset
    if output_suffix.strip():
        output_dir = f'{output_dir}_{output_suffix}'
    if timestamp:
        run_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        output_dir = f'{output_dir}_{run_datetime}'
    
    project_dir = os.path.abspath(os.getcwd())
    dataset_config_path = os.path.join(
        project_dir, 'config', 'dataset_config.json')
    with open (dataset_config_path) as json_config:
        dataset_config_options = json.load(json_config)
    dataset_config = dataset_config_options[training_dataset]
    ml_config_path = os.path.join(project_dir, 'config', 'ml_config.json')
    with open (ml_config_path) as json_config:
        ml_config_options = json.load(json_config)
    ml_config = ml_config_options['grid_search_cv']

    TRAIN_TEST_SPLIT_RANDOM_STATE = 0
    CV_RANDOM_STATE = 21
    MODEL_RANDOM_STATE = 42
    VERBOSE = 1

    home_dir = os.path.expanduser('~')
    input_dir = os.path.join(
        home_dir, dataset_config['io_dir'].replace('/', os.sep), 'input')
    results_dir = os.path.join(project_dir, 'results', output_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create X, X_cov, and y matrices
    df = pd.read_csv(os.path.join(input_dir, 'df.csv'), index_col=0)
    variables_to_drop.append(target_variable)
    y = df[target_variable]
    X = df.drop(variables_to_drop, axis=1)
    # Use when creating table of feature importances
    # Make sure there will be no duplicate column names if 
    # excluding a prefix
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=TRAIN_TEST_SPLIT_RANDOM_STATE)
    Xy_train_test_matrices = {'X': (X_train, X_test, y_train, y_test)}
    if train_with_covar:
        df_covar = pd.read_csv(
            os.path.join(input_dir, 'df_covar.csv'), index_col=0)
        covar = pd.get_dummies(df_covar, prefix='', prefix_sep='')
        X_with_covar = X.join(covar)
        X_train_with_covar = X_with_covar.reindex(X_train.index)
        X_test_with_covar = X_with_covar.reindex(X_test.index)
        Xy_train_test_matrices['X_with_covar'] = (
            X_train_with_covar, X_test_with_covar,
            y_train, y_test,
            )
    if train_with_shuffled_data:
        X_train_shuffled = shuffle(X_train).reset_index(drop=True)
        X_test_shuffled = shuffle(X_test).reset_index(drop=True)
        Xy_train_test_matrices['X_shuffled'] = (
            X_train_shuffled, X_test_shuffled,
            y_train, y_test,
            )
    
    # Setup pipelines, hyperparameters, etc.
    cv = KFold(n_splits=5, shuffle=True, random_state=CV_RANDOM_STATE)
    r2_scorer = make_scorer(r2_score)
    # Ridge regression
    sgd_ridge_regressor = SGDRegressor(random_state=MODEL_RANDOM_STATE)
    sgd_ridge_regressor_pipeline = Pipeline(
        steps=[
            ('imputer', SimpleImputer()),
            ('scaler', StandardScaler()),
            ('regressor', sgd_ridge_regressor)
            ])
    sgd_ridge_regressor_param_grid = {
        **ml_config['simple_imputer'],
        **ml_config['sgd_ridge_regression']
        }
    sgd_ridge_regressor_cv = GridSearchCV(
        estimator=sgd_ridge_regressor_pipeline,
        param_grid=sgd_ridge_regressor_param_grid, cv=cv, refit=True,
        scoring=r2_scorer, n_jobs=n_jobs, verbose=VERBOSE)
    # Random forest
    rf_regressor = RandomForestRegressor(random_state=MODEL_RANDOM_STATE)
    # Use a passthrough scaler to keep pipeline steps consistent
    rf_regressor_pipeline = Pipeline(
        steps=[
            ('imputer', SimpleImputer()),
            ('scaler', None),
            ('regressor', rf_regressor)
            ])
    rf_regressor_param_grid = {
        **ml_config['simple_imputer'],
        **ml_config['rf_regression']
        }
    rf_regressor_cv = GridSearchCV(
        estimator=rf_regressor_pipeline, param_grid=rf_regressor_param_grid,
        cv=cv, refit=True, scoring=r2_scorer, n_jobs=n_jobs, verbose=VERBOSE)
    models_to_train = {
        'rr': ('Ridge regression', sgd_ridge_regressor_cv),
        'rfr': ('Random forest regression', rf_regressor_cv)
        }
    
    all_cv_results = pd.DataFrame()
    test_result_columns = [
        'model_key', 'model_name', 'training_x',
        'r2_score', 'shuffled_r2_score', 'best_hyperparameters'
        ]
    all_test_results = pd.DataFrame(columns=test_result_columns)
    for model_cv_key in models_to_train:
        model_cv_name, model_cv = models_to_train[model_cv_key]
        for Xy_key, Xy_matrices in Xy_train_test_matrices.items():
            model_results_dir = os.path.join(
                results_dir, f'{model_cv_key}_{Xy_key}')
            if not os.path.exists(model_results_dir):
                os.makedirs(model_results_dir)
            X_train_i, X_test_i, y_train_i, y_test_i = Xy_matrices
            # Train model
            print(f'*****\n{model_cv_name} training on {Xy_key}')
            model_cv.fit(X=X_train_i, y=y_train_i)
            # Results of hyperparameter search
            cv_results = pd.DataFrame(model_cv.cv_results_)
            cv_results.insert(0, 'model_key', model_cv_key)
            cv_results.insert(1, 'model_name', model_cv_name)
            cv_results.insert(2, 'training_x', Xy_key)
            all_cv_results = pd.concat(
                [
                    all_cv_results,
                    cv_results.sort_values(by=['rank_test_score']).iloc[0:10]
                    ],
                sort=True, ignore_index=True)
            # Test on unseen data
            y_test_predictions = model_cv.predict(X_test_i)
            y_test_predictions = pd.DataFrame(
                {'true': y_test_i, 'prediction': y_test_predictions})
            test_score = r2_score(
                y_test_predictions['true'], y_test_predictions['prediction'])
            shuffled_test_score = np.nan
            if Xy_key == 'X':
                shuffled_y_test_predictions = model_cv.predict(
                    shuffle(X_test_i).reset_index(drop=True))
                shuffled_y_test_predictions = pd.DataFrame(
                    data={
                        'true': y_test_i,
                        'prediction': shuffled_y_test_predictions
                        })
                shuffled_test_score = r2_score(
                    shuffled_y_test_predictions['true'],
                    shuffled_y_test_predictions['prediction'])
            test_results = [
                model_cv_key, model_cv_name, Xy_key,
                test_score, shuffled_test_score, str(model_cv.best_params_)
                ]
            test_results = pd.DataFrame(
                index=[0], data=dict(zip(test_result_columns, test_results)))
            all_test_results = pd.concat(
                [all_test_results, test_results], ignore_index=True)
            feature_importances = {}
            if model_cv_key in ['rr']:
                feature_importances['coefficient'] = (
                    model_cv.best_estimator_.named_steps['regressor'].coef_
                    .flatten().tolist())
                feature_importances['normalized_abs_importance'] = (
                    minmax_scale(
                        [abs(i) for i in feature_importances['coefficient']]))
            elif model_cv_key in ['rfr']:
                feature_importances['importance'] = (
                    model_cv.best_estimator_.named_steps['regressor']
                    .feature_importances_.tolist())
                feature_importances['normalized_abs_importance'] = (
                    minmax_scale(
                        [abs(i) for i in feature_importances['importance']]))
            feature_importances = pd.DataFrame(data=feature_importances)
            feature_importances.insert(0, 'model_key', model_cv_key)
            feature_importances.insert(1, 'model_name', model_cv_name)
            feature_importances.insert(2, 'training_x', Xy_key)
            feature_importances.insert(
                3, 'feature', model_cv.feature_names_in_.tolist())
            
            dump(
                model_results_dir,
                os.path.join(model_results_dir, 'model.joblib'))
            y_test_predictions.to_csv(
                os.path.join(model_results_dir, 'y_test_predictions.csv'))
            feature_importances.to_csv(
                os.path.join(model_results_dir, 'feature_importances.csv'))

    all_cv_results.to_csv(os.path.join(results_dir, 'cv_results.csv'))
    all_test_results.to_csv(os.path.join(results_dir, 'test_results.csv'))

if __name__ == '__main__':
    main()