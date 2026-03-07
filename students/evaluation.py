"""
Model evaluation functions: metrics and ROC/PR curves.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    roc_auc_score, auc as compute_auc, r2_score
)


def calculate_r2_score(y_true, y_pred):
    """
    Calculate R² score for regression.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True target values
    y_pred : np.ndarray or pd.Series
        Predicted target values
        
    Returns
    -------
    float
        R² score (between -inf and 1, higher is better)
    """
    # TODO: Implement R² calculation
    # Use sklearn's r2_score
    from sklearn.metrics import r2_score

    r2 = r2_score(y_true, y_pred)

    return float(r2)
    


def calculate_classification_metrics(y_true, y_pred):
    """
    Calculate classification metrics.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred : np.ndarray or pd.Series
        Predicted binary labels
        
    Returns
    -------
    dict
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # TODO: Implement metrics calculation
    # Return dictionary with all four metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Create dictionary for the metrics
    classification_metrics = {"accuracy": accuracy,
                              "precision": precision,
                              "recall": recall,
                              "f1": f1}

    return classification_metrics
    


def calculate_auroc_score(y_true, y_pred_proba):
    """
    Calculate Area Under the ROC Curve (AUROC).
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
        
    Returns
    -------
    float
        AUROC score (between 0 and 1)
    """
    # TODO: Implement AUROC calculation
    # Use sklearn's roc_auc_score
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y_true, y_pred_proba)

    return float(auc)


def calculate_auprc_score(y_true, y_pred_proba):
    """
    Calculate Area Under the Precision-Recall Curve (AUPRC).
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
        
    Returns
    -------
    float
        AUPRC score (between 0 and 1)
    """
    # TODO: Implement AUPRC calculation
    # Use sklearn's average_precision_score
    from sklearn.metrics import average_precision_score

    ap = average_precision_score(y_true, y_pred_proba)
    
    return float(ap)


def generate_auroc_curve(y_true, y_pred_proba, model_name="Model", 
                        output_path=None, ax=None):
    """
    Generate and plot ROC curve.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
    model_name : str
        Name of the model (for legend)
    output_path : str, optional
        Path to save figure
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
        
    Returns
    -------
    tuple
        (figure, ax) or (figure,) if ax provided
    """
    # TODO: Implement ROC curve plotting
    # - Calculate ROC curve using roc_curve()
    # - Calculate AUROC using auc()
    # - Plot curve with label showing AUROC score
    # - Add diagonal reference line
    # - Set labels: "False Positive Rate", "True Positive Rate"
    # - Save to output_path if provided
    # - Return figure and/or axes

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auroc = auc(fpr, tpr)

    # If no axis is provided, make an axis
    # Otherwise, use the axis provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        fig = ax.figure
    
    # Plotting the graph
    ax.plot(fpr, tpr, label=f"{model_name} ROC Curve (AUC={auroc:.3f})")
    ax.plot([0,1], [0,1], "k--", label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} ROC Curve")
    ax.legend()
    ax.grid(alpha=0.3)

    if output_path != None:
        plt.savefig(output_path)

    # If axis was provided, return a tuple with both the figure and ax
    # Otherwise, only return the figure
    if ax is None:
        return fig, ax
    else:
        return fig


def generate_auprc_curve(y_true, y_pred_proba, model_name="Model",
                        output_path=None, ax=None):
    """
    Generate and plot Precision-Recall curve.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba : np.ndarray or pd.Series
        Predicted probabilities for positive class
    model_name : str
        Name of the model (for legend)
    output_path : str, optional
        Path to save figure
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure
        
    Returns
    -------
    tuple
        (figure, ax) or (figure,) if ax provided
    """
    # TODO: Implement PR curve plotting
    # - Calculate precision-recall curve using precision_recall_curve()
    # - Calculate AUPRC using average_precision_score()
    # - Plot curve with label showing AUPRC score
    # - Add horizontal baseline (prevalence)
    # - Set labels: "Recall", "Precision"
    # - Save to output_path if provided
    # - Return figure and/or axes

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)

    # If no axis is provided, make an axis
    # Otherwise, use the axis provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        fig = ax.figure

    # Plotting the graph
    ax.plot(recall, precision, label=f"{model_name} PR Curve (AP={ap:.3f})")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{model_name} Precision-Recall Curve')
    ax.legend()
    ax.grid(alpha=0.3)

    if output_path != None:
        plt.savefig(output_path)

    # If axis was provided, return a tuple with both the figure and ax
    # Otherwise, only return the figure
    if ax is None:
        return fig, ax
    else:
        return fig


def plot_comparison_curves(y_true, y_pred_proba_log, y_pred_proba_knn,
                          output_path=None):
    """
    Plot ROC and PR curves for both logistic regression and k-NN side by side.
    
    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        True binary labels
    y_pred_proba_log : np.ndarray or pd.Series
        Predicted probabilities from logistic regression
    y_pred_proba_knn : np.ndarray or pd.Series
        Predicted probabilities from k-NN
    output_path : str, optional
        Path to save figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure with 2 subplots (ROC and PR curves)
    """
    # TODO: Implement comparison plotting
    # - Create figure with 1x2 subplots
    # - Left: ROC curves for both models
    # - Right: PR curves for both models
    # - Add legends with AUROC/AUPRC scores
    # - Save to output_path if provided
    # - Return figure

    figure, axes = plt.subplot(1, 2, figsize=(12,5))

    fpr_log, tpr_log, _ = roc_curve(y_true, y_pred_proba_log)
    fpr_knn, tpr_knn, _ = roc_curve(y_true, y_pred_proba_knn)
    auc_log = roc_auc_score(y_true, y_pred_proba_log)
    auc_knn = roc_auc_score(y_true, y_pred_proba_knn)

    axes[0].plot(fpr_log, tpr_log)
    axes[0].plot(fpr_knn, tpr_knn)
    axes[0].set_title("ROC Curves")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()


    precision_log, recall_log = precision_recall_curve(y_true, y_pred_proba_log)
    precision_knn, recall_knn = precision_recall_curve(y_true, y_pred_proba_knn)
    avg_precision_log = average_precision_score(y_true, y_pred_proba_log)
    avg_precision_knn = average_precision_score(y_true, y_pred_proba_knn)

    axes[1].plot(precision_log, recall_log)
    axes[1].plot(precision_knn, recall_knn)
    axes[1].set_title("Precision-Recall Curves")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    plt.tight_layout()

    if output_path:
        figure.savefig(output_path)

    return figure
