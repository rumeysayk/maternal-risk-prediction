# ==========================================================
# FILE: preprocessing.py
# PURPOSE:
# Data preprocessing pipeline for maternal risk prediction
#
# FEATURES:
# - Load dataset
# - Handle missing values
# - Encode target variable
# - Feature / target separation
# - Train-test split
# - Feature scaling
#
# OUTPUT:
# - X_train_scaled
# - X_test_scaled
# - y_train
# - y_test
# - scaler object
#
# AUTHOR:
# Maternal Risk Prediction Project
# ==========================================================


# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset(file_path):
    """
    Loads the dataset from a CSV file.

    Parameters:
    ----------
    file_path : str
        Path to dataset CSV file

    Returns:
    -------
    df : pandas.DataFrame
        Loaded dataset
    """

    df = pd.read_csv(file_path)

    print("Dataset loaded successfully.")

    return df


# ==========================================================
# HANDLE MISSING VALUES
# ==========================================================

def handle_missing_values(df):
    """
    Handles missing values in the dataset.

    Numerical columns:
    - Filled using median values

    Target column:
    - Filled using mode value

    Parameters:
    ----------
    df : pandas.DataFrame

    Returns:
    -------
    df : pandas.DataFrame
        Cleaned dataset
    """

    # ------------------------------------------
    # Numerical Columns
    # ------------------------------------------

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    for column in numerical_columns:

        df[column] = df[column].fillna(
            df[column].median()
        )

    # ------------------------------------------
    # Target Column
    # ------------------------------------------

    df["Risk Level"] = df["Risk Level"].fillna(
        df["Risk Level"].mode()[0]
    )

    print("Missing values handled successfully.")

    return df


# ==========================================================
# ENCODE TARGET VARIABLE
# ==========================================================

def encode_target_variable(df):
    """
    Encodes target variable into numerical format.

    Encoding:
    ----------
    Low  -> 0
    High -> 1

    Parameters:
    ----------
    df : pandas.DataFrame

    Returns:
    -------
    df : pandas.DataFrame
        Encoded dataset
    """

    df["Risk Level"] = df["Risk Level"].map(
        {
            "Low": 0,
            "High": 1
        }
    )

    print("Target variable encoded successfully.")

    return df


# ==========================================================
# FEATURE / TARGET SEPARATION
# ==========================================================

def split_features_target(df):
    """
    Separates features and target variable.

    Parameters:
    ----------
    df : pandas.DataFrame

    Returns:
    -------
    X : pandas.DataFrame
        Feature matrix

    y : pandas.Series
        Target vector
    """

    X = df.drop("Risk Level", axis=1)

    y = df["Risk Level"]

    print("Features and target variable separated successfully.")

    return X, y


# ==========================================================
# TRAIN-TEST SPLIT
# ==========================================================

def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    Splits dataset into training and testing sets.

    Parameters:
    ----------
    X : pandas.DataFrame
        Feature matrix

    y : pandas.Series
        Target vector

    test_size : float
        Test size ratio

    random_state : int
        Random seed

    Returns:
    -------
    X_train, X_test, y_train, y_test
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    print("Train-test split completed successfully.")

    return X_train, X_test, y_train, y_test


# ==========================================================
# FEATURE SCALING
# ==========================================================

def scale_features(X_train, X_test):
    """
    Applies standardization to feature values.

    StandardScaler:
    - mean = 0
    - standard deviation = 1

    Parameters:
    ----------
    X_train : pandas.DataFrame

    X_test : pandas.DataFrame

    Returns:
    -------
    X_train_scaled
    X_test_scaled
    scaler
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    print("Feature scaling completed successfully.")

    return X_train_scaled, X_test_scaled, scaler


# ==========================================================
# COMPLETE PREPROCESSING PIPELINE
# ==========================================================

def preprocess_data(file_path):
    """
    Executes the complete preprocessing pipeline.

    Pipeline Steps:
    ----------------
    1. Load dataset
    2. Handle missing values
    3. Encode target variable
    4. Feature-target separation
    5. Train-test split
    6. Feature scaling

    Parameters:
    ----------
    file_path : str

    Returns:
    -------
    X_train_scaled
    X_test_scaled
    y_train
    y_test
    scaler
    """

    # ------------------------------------------
    # Load Dataset
    # ------------------------------------------

    df = load_dataset(file_path)

    # ------------------------------------------
    # Missing Values
    # ------------------------------------------

    df = handle_missing_values(df)

    # ------------------------------------------
    # Encode Target
    # ------------------------------------------

    df = encode_target_variable(df)

    # ------------------------------------------
    # Split Features / Target
    # ------------------------------------------

    X, y = split_features_target(df)

    # ------------------------------------------
    # Train-Test Split
    # ------------------------------------------

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y
    )

    # ------------------------------------------
    # Feature Scaling
    # ------------------------------------------

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train,
        X_test
    )

    print("Preprocessing pipeline completed successfully.")

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    )


# ==========================================================
# MAIN TEST
# ==========================================================

if __name__ == "__main__":

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(
        "data/Dataset.csv"
    )

    print("\nX_train shape:", X_train_scaled.shape)
    print("X_test shape:", X_test_scaled.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)