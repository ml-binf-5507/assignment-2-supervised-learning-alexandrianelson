"""
Classification functions for logistic regression and k-nearest neighbors.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def train_logistic_regression_grid(X_train, y_train, param_grid=None):
    """
    Train logistic regression models with grid search over hyperparameters.
    
    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Training feature matrix
    y_train : np.ndarray or pd.Series
        Training target vector (binary)
    param_grid : dict, optional
        Parameter grid for GridSearchCV. 
        Default: {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'penalty': ['l2'],
                  'solver': ['lbfgs']}
        
    Returns
    -------
    sklearn.model_selection.GridSearchCV
        Fitted GridSearchCV object with best model
    """
    if param_grid is None:
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }
    
    # TODO: Implement grid search for logistic regression
    # - Create LogisticRegression with max_iter=1000
    # - Use GridSearchCV with cv=5
    # - Fit on training data
    # - Return fitted GridSearchCV object

    
    log_regression = LogisticRegression(max_iter=1000)
    grid_search = GridSearchCV(log_regression, param_grid, cv=5)

    grid_search.fit(X_train, y_train)

    return grid_search


def train_knn_grid(X_train, y_train, param_grid=None):
    """
    Train k-NN models with grid search over hyperparameters.
    
    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Training feature matrix (should be scaled)
    y_train : np.ndarray or pd.Series
        Training target vector (binary)
    param_grid : dict, optional
        Parameter grid for GridSearchCV.
        Default: {'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
                  'weights': ['uniform', 'distance'],
                  'metric': ['euclidean', 'manhattan']}
        
    Returns
    -------
    sklearn.model_selection.GridSearchCV
        Fitted GridSearchCV object with best model
    """
    if param_grid is None:
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    
    # TODO: Implement grid search for k-NN
    # - Create KNeighborsClassifier
    # - Use GridSearchCV with cv=5
    # - Fit on training data
    # - Return fitted GridSearchCV object

    knn_model = KNeighborsClassifier()
    grid_search = GridSearchCV(knn_model, param_grid, cv=5)

    grid_search.fit(X_train, y_train)

    return grid_search
    


def get_best_logistic_regression(X_train, y_train, X_test, y_test, param_grid=None):
    """
    Get best logistic regression model with test R² evaluation.
    
    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Training features
    y_train : np.ndarray or pd.Series
        Training target
    X_test : np.ndarray or pd.DataFrame
        Test features
    y_test : np.ndarray or pd.Series
        Test target
    param_grid : dict, optional
        Parameter grid for GridSearchCV
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'model': best fitted LogisticRegression model
        - 'best_params': best parameters found
        - 'cv_results_df': DataFrame of all CV results
    """
    # TODO: Implement best model retrieval
    # - Use train_logistic_regression_grid
    # - Extract best model
    # - Return dictionary

    # Run the grid search
    model = train_logistic_regression_grid(X_train, y_train, param_grid)

    y_pred = model.predict(X_test)

    score = model.score(X_test, y_test)

    # Best model
    best_model = model.best_estimator_

    # Create dictionary to return results of the best model
    results_dict = {"model": model.best_estimator_, 
                    "best_params": model.best_params_, 
                    "cv_results_df": pd.DataFrame(model.cv_results_)}

    return results_dict


def get_best_knn(X_train, y_train, X_test, y_test, param_grid=None):
    """
    Get best k-NN model with test R² evaluation.
    
    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Training features (scaled)
    y_train : np.ndarray or pd.Series
        Training target
    X_test : np.ndarray or pd.DataFrame
        Test features (scaled)
    y_test : np.ndarray or pd.Series
        Test target
    param_grid : dict, optional
        Parameter grid for GridSearchCV
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'model': best fitted KNeighborsClassifier model
        - 'best_params': best parameters found
        - 'best_k': best n_neighbors value
        - 'cv_results_df': DataFrame of all CV results
    """
    # TODO: Implement best model retrieval
    # - Use train_knn_grid
    # - Extract best model and best_k
    # - Return dictionary
    
    # Run grid search
    model = train_knn_grid(X_train, y_train, param_grid)

    y_pred = model.predict(X_test)

    score = model.score(X_test, y_test)

    # Extract best model
    best_model = model.best_estimator_

    results_dict = {"model": model.best_estimator_, # best fitted model
                    "best_params": model.best_params_, # best parameters found
                    "best_k": model.best_params_["n_neighbors"], # best_k
                    "cv_results_df": pd.DataFrame(model.cv_results_)}

    return results_dict
