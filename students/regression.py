"""
Linear regression functions for predicting cholesterol using ElasticNet.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score


def train_elasticnet_grid(X_train, y_train, l1_ratios, alphas):
    """
    Train ElasticNet models over a grid of hyperparameters.
    
    Parameters
    ----------
    X_train : np.ndarray or pd.DataFrame
        Training feature matrix
    y_train : np.ndarray or pd.Series
        Training target vector
    l1_ratios : list or np.ndarray
        L1 ratio values to test (0 = L2 only, 1 = L1 only)
    alphas : list or np.ndarray
        Regularization strength values to test
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['l1_ratio', 'alpha', 'r2_score', 'model']
        Contains R² scores for each parameter combination on training data
    """
    # TODO: Implement grid search
    # - Create results list
    # - For each combination of l1_ratio and alpha:
    #   - Train ElasticNet model with max_iter=5000
    #   - Calculate R² score on training data
    #   - Store results
    # - Return DataFrame with results

    # Create results list
    results = []

    for ratio in l1_ratios: # Iterate over L1 ratios
        for alpha in alphas: # Iterate over alphas
            model = ElasticNet(
                l1_ratio=ratio,
                alpha=alpha,
                random_state=42,
                max_iter=5000
            )

            model.fit(X_train, y_train)

            score = model.score(X_train, y_train) # Calculate R2 score

            # Store results as a dictionary in the results list
            results.append({"l1_ratio": ratio, 
                            "alpha" : alpha,
                            "r2_score" : score,
                            "model" : model})
    
    df = pd.DataFrame(results)
    
    return df


def create_r2_heatmap(results_df, l1_ratios, alphas, output_path=None):
    """
    Create a heatmap of R² scores across l1_ratio and alpha parameters.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results from train_elasticnet_grid
    l1_ratios : list or np.ndarray
        L1 ratio values used in grid
    alphas : list or np.ndarray
        Alpha values used in grid
    output_path : str, optional
        Path to save figure. If None, returns figure object
        
    Returns
    -------
    matplotlib.figure.Figure
        The heatmap figure
    """
    # TODO: Implement heatmap creation
    # - Pivot results_df to create matrix with l1_ratio on x-axis, alpha on y-axis
    # - Create heatmap using seaborn
    # - Set labels: "L1 Ratio", "Alpha", "R² Score"
    # - Add colorbar
    # - Save to output_path if provided
    # - Return figure object

    # Pivot results_df to create the matrix
    heatmap_data = results_df.pivot(
        index="alpha", 
        columns="l1_ratio", 
        values="r2_score")

    # Create figure
    figure = plt.figure(figsize=(8,6))

    # Create a heatmap
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis")
    
    # Add labels for the axes and the figure
    plt.xlabel("L1 Ratio")
    plt.ylabel("Alpha")
    plt.title("ElasticNet R² Scores")

    # Create conditional for the output path
    if output_path:
        plt.savefig(output_path)

    return figure


def get_best_elasticnet_model(X_train, y_train, X_test, y_test, 
                               l1_ratios=None, alphas=None):
    """
    Find and train the best ElasticNet model on test data.
    
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
    l1_ratios : list, optional
        L1 ratio values to test. Default: [0.1, 0.3, 0.5, 0.7, 0.9]
    alphas : list, optional
        Alpha values to test. Default: [0.001, 0.01, 0.1, 1.0, 10.0]
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'model': fitted ElasticNet model
        - 'best_l1_ratio': best l1 ratio
        - 'best_alpha': best alpha
        - 'train_r2': R² on training data
        - 'test_r2': R² on test data
        - 'results_df': full results DataFrame
    """
    if l1_ratios is None:
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
    
    # TODO: Implement best model selection
    # - Train models using train_elasticnet_grid
    # - Select model with highest test R² (not training R²)
    # - Return dictionary with best model and parameters

    # Train models using train_elastic_grid
    results_df = train_elasticnet_grid(X_train, y_train, l1_ratios=l1_ratios, alphas=alphas)

    # Store model's r2 results in the data frame under a new column
    results_df["test_r2"] = results_df["model"].apply(lambda x: x.score(X_test, y_test))

    # Select the row with the best r2 score
    index = results_df["test_r2"].idxmax()
    best_row = results_df.loc[index]

    # Return dictionary with best model and parameters
    best_model = {
        "model":best_row["model"],
        "best_l1_ratio":best_row["l1_ratio"],
        "best_alpha":best_row["alpha"],
        "train_r2":best_row["r2_score"],
        "test_r2":best_row["test_r2"],
        "results_df":results_df
    }

    return best_model
