"""
Data loading and preprocessing functions for heart disease dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_heart_disease_data(filepath):
    """
    Load the heart disease dataset from CSV.
    
    Parameters
    ----------
    filepath : str
        Path to the heart disease CSV file
        
    Returns
    -------
    pd.DataFrame
        Raw dataset with all features and targets
        
    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist
    ValueError
        If the CSV is empty or malformed
        
    Examples
    --------
    >>> df = load_heart_disease_data('data/heart_disease_uci.csv')
    >>> df.shape
    (270, 15)
    """
    # Hint: Use pd.read_csv()
    # Hint: Check if file exists and raise helpful error if not
    # TODO: Implement data loading
    
    df = pd.read_csv(filepath)
    
    shape = df.shape
    
    if shape[0] == 0:
        raise ValueError
    
    return df


def preprocess_data(df):
    """
    Handle missing values, encode categorical variables, and clean data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset
        
    Returns
    -------
    pd.DataFrame
        Cleaned and preprocessed dataset
    """
    # TODO: Implement preprocessing
    # - Handle missing values
    # - Encode categorical variables (e.g., sex, cp, fbs, etc.)
    # - Ensure all columns are numeric
    
    print(df.isnull().sum())

    # Replacing missing values
    df = df.replace(["NA", "N/A", "na", "n/a", "NaN", "nan", ""], np.nan)

    # Grabbing column names and sorting them
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()

    # Need to impute the median for missing numeric values 
    for col in num_cols:
            column_median = df[col].median()
            df[col] = df[col].fillna(column_median)

    # Need to impute the mode for missing categorical values
    for col in cat_cols:
        column_mode = df[col].mode()
        df[col] = df[col].fillna(column_mode)

    # Removing duplicates
    df_duplicated = df.duplicated().sum()
    if df_duplicated > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    # Encode categorical columns
    encoded_cat_columns = []

    for col in cat_cols:
        encoded = pd.get_dummies(df[col], prefix=col, dtype=int)

        encoded_cat_columns.extend(encoded.columns.tolist())
    
        df = df.drop(col, axis=1)
        df = pd.concat([df, encoded], axis=1)

    # Checking that all columns are numeric
    for col in df.columns:
        if df[col].dtype == "object":
            print(f"{col} is not numeric!")
        elif df[col].dtype in ["int", "float64"]:
            continue
    
    print(df.isnull().sum())
    return df


def prepare_regression_data(df, target='chol'):
    """
    Prepare data for linear regression (predicting serum cholesterol).
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset
    target : str
        Target column name (default: 'chol')
        
    Returns
    -------
    tuple
        (X, y) feature matrix and target vector
    """
    # TODO: Implement regression data preparation
    # - Remove rows with missing chol values
    # - Exclude chol from features
    # - Return X (features) and y (target)
    
    # Remove rows with missing chol values. Mainly handled in the preprocessing function
    df = df.dropna(subset=[target])

    # Exclude chol from features
    X = df.drop(columns=[target], axis=1)

    # Create the target dataset using chol/target
    y = df[target]

    return X, y


def prepare_classification_data(df, target='num'):
    """
    Prepare data for classification (predicting heart disease presence).
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset
    target : str
        Target column name (default: 'num')
        
    Returns
    -------
    tuple
        (X, y) feature matrix and target vector (binary)
    """
    # TODO: Implement classification data preparation
    # - Binarize target variable
    # - Exclude target from features
    # - Exclude chol from features
    # - Return X (features) and y (target)

    # If the target variable is not in the columns,
    # but target is equal to target and num (the actual target variable),
    # is in the columns, target object must be "num"
    # Handles KeyError during testing
    if target not in df.columns:
        if target == "target" and "num" in df.columns:
            target = "num"
    
    # Binarize target variable
    y = (df[target] > 0).astype(int)

    # Exclude target and chol from features
    X = df.drop(columns=[target, "chol"])

    return X, y


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Split data into train/test sets and scale features.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Target vector
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
        where scaler is the fitted StandardScaler
    """
    # TODO: Implement train/test split and scaling
    # - Use train_test_split with provided parameters
    # - Fit StandardScaler on training data only
    # - Transform both train and test data
    # - Return scaled data and scaler object

    # Split data into test and train data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state)
   
    # Create the scaler for the training data transformation
    scaler = StandardScaler()

    # Fit scaler on training set
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform testing set
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
