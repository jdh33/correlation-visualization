ml = {
    'grid_search_cv': {
        'simple_imputer': {
            'imputer__strategy': [
                'mean', 'median',
                'most_frequent', 'constant'
                ]
        },
        'sgd_ridge_regression': {
            'regressor__alpha': [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0]
        },
        'rf_regression_': {
        'regressor__max_features': [0.1, 0.33, 0.5, 0.66, 1.0]
        }
    }
}