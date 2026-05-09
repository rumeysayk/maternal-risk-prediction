# ==========================================================
# FILE: evaluation.py
# PURPOSE:
# Model evaluation pipeline for maternal risk prediction
#
# FEATURES:
# - Accuracy calculation
# - ROC-AUC evaluation
# - Classification reports
# - Confusion matrices
# - Cross validation
# - Feature importance analysis
# - Final model comparison
#
# OUTPUT:
# - evaluation metrics
# - comparison tables
# - best performing model
#
# AUTHOR:
# Maternal Risk Prediction Project
# ==========================================================


# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from sklearn.model_selection import cross_val_score

# ------------------------------------------
# Import project modules
# ------------------------------------------

from preprocessing import preprocess_data
from train_models import train_all_models


# ==========================================================
# CALCULATE ACCURACY
# ==========================================================

def calculate_accuracy(y_test, y_pred):
    """
    Calculates model accuracy.

    Parameters:
    ----------
    y_test : pandas.Series

    y_pred : ndarray

    Returns:
    -------
    accuracy : float
    """

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    return accuracy


# ==========================================================
# CALCULATE ROC-AUC
# ==========================================================

def calculate_roc_auc(y_test, y_pred):
    """
    Calculates ROC-AUC score.

    Parameters:
    ----------
    y_test : pandas.Series

    y_pred : ndarray

    Returns:
    -------
    auc_score : float
    """

    auc_score = roc_auc_score(
        y_test,
        y_pred
    )

    return auc_score


# ==========================================================
# GENERATE CLASSIFICATION REPORT
# ==========================================================

def generate_classification_report(y_test, y_pred):
    """
    Generates classification report.

    Parameters:
    ----------
    y_test : pandas.Series

    y_pred : ndarray

    Returns:
    -------
    report : str
    """

    report = classification_report(
        y_test,
        y_pred
    )

    return report


# ==========================================================
# GENERATE CONFUSION MATRIX
# ==========================================================

def generate_confusion_matrix(y_test, y_pred):
    """
    Generates confusion matrix.

    Parameters:
    ----------
    y_test : pandas.Series

    y_pred : ndarray

    Returns:
    -------
    cm : ndarray
    """

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    return cm


# ==========================================================
# CROSS VALIDATION
# ==========================================================

def perform_cross_validation(model, X_train, y_train):
    """
    Performs cross validation.

    Parameters:
    ----------
    model : sklearn model

    X_train : ndarray

    y_train : pandas.Series

    Returns:
    -------
    cv_scores : ndarray
    """

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5
    )

    return cv_scores


# ==========================================================
# EVALUATE SINGLE MODEL
# ==========================================================

def evaluate_single_model(
    model_name,
    model,
    y_test,
    y_pred,
    X_train,
    y_train
):
    """
    Evaluates a single machine learning model.

    Metrics:
    --------
    - Accuracy
    - ROC-AUC
    - Classification Report
    - Confusion Matrix
    - Cross Validation

    Parameters:
    ----------
    model_name : str

    model : sklearn model

    y_test : pandas.Series

    y_pred : ndarray

    X_train : ndarray

    y_train : pandas.Series

    Returns:
    -------
    results : dict
    """

    # ------------------------------------------
    # Accuracy
    # ------------------------------------------

    accuracy = calculate_accuracy(
        y_test,
        y_pred
    )

    # ------------------------------------------
    # ROC-AUC
    # ------------------------------------------

    roc_auc = calculate_roc_auc(
        y_test,
        y_pred
    )

    # ------------------------------------------
    # Classification Report
    # ------------------------------------------

    report = generate_classification_report(
        y_test,
        y_pred
    )

    # ------------------------------------------
    # Confusion Matrix
    # ------------------------------------------

    cm = generate_confusion_matrix(
        y_test,
        y_pred
    )

    # ------------------------------------------
    # Cross Validation
    # ------------------------------------------

    cv_scores = perform_cross_validation(
        model,
        X_train,
        y_train
    )

    cv_mean = cv_scores.mean()

    # ------------------------------------------
    # Print Evaluation Results
    # ------------------------------------------

    print("\n==================================================")
    print(f"{model_name.upper()} EVALUATION")
    print("==================================================")

    print(f"\nAccuracy: {accuracy:.4f}")

    print(f"ROC-AUC: {roc_auc:.4f}")

    print(f"CV Mean: {cv_mean:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    # ------------------------------------------
    # Store Results
    # ------------------------------------------

    results = {
        "Model": model_name,
        "Accuracy": accuracy,
        "ROC-AUC": roc_auc,
        "CV Mean": cv_mean
    }

    return results


# ==========================================================
# EVALUATE ALL MODELS
# ==========================================================

def evaluate_all_models(
    models,
    predictions,
    y_test,
    X_train,
    y_train
):
    """
    Evaluates all machine learning models.

    Parameters:
    ----------
    models : dict

    predictions : dict

    y_test : pandas.Series

    X_train : ndarray

    y_train : pandas.Series

    Returns:
    -------
    results_df : pandas.DataFrame
    """

    evaluation_results = []

    # ------------------------------------------
    # Evaluate Each Model
    # ------------------------------------------

    for model_name, model in models.items():

        y_pred = predictions[model_name]

        result = evaluate_single_model(
            model_name=model_name,
            model=model,
            y_test=y_test,
            y_pred=y_pred,
            X_train=X_train,
            y_train=y_train
        )

        evaluation_results.append(result)

    # ------------------------------------------
    # Create Results DataFrame
    # ------------------------------------------

    results_df = pd.DataFrame(
        evaluation_results
    )

    # ------------------------------------------
    # Sort Results
    # ------------------------------------------

    results_df = results_df.sort_values(
        by="Accuracy",
        ascending=False
    )

    print("\n==================================================")
    print("FINAL MODEL COMPARISON")
    print("==================================================")

    print(results_df)

    return results_df


# ==========================================================
# BEST MODEL IDENTIFICATION
# ==========================================================

def get_best_model(results_df):
    """
    Returns best performing model.

    Parameters:
    ----------
    results_df : pandas.DataFrame

    Returns:
    -------
    best_model : pandas.Series
    """

    best_model = results_df.iloc[0]

    print("\n==================================================")
    print("BEST PERFORMING MODEL")
    print("==================================================")

    print(best_model)

    return best_model


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def get_feature_importance(model, feature_names):
    """
    Extracts Random Forest feature importance.

    Parameters:
    ----------
    model : RandomForestClassifier

    feature_names : list

    Returns:
    -------
    feature_importance_df : pandas.DataFrame
    """

    importance_scores = model.feature_importances_

    feature_importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance_scores
        }
    )

    feature_importance_df = feature_importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n==================================================")
    print("FEATURE IMPORTANCE")
    print("==================================================")

    print(feature_importance_df)

    return feature_importance_df


# ==========================================================
# MAIN TEST
# ==========================================================

if __name__ == "__main__":

    # ------------------------------------------
    # Preprocessing Pipeline
    # ------------------------------------------

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(
        "data/Dataset.csv"
    )

    # ------------------------------------------
    # Train Models
    # ------------------------------------------

    models, predictions = train_all_models(
        X_train_scaled,
        y_train,
        X_test_scaled
    )

    # ------------------------------------------
    # Evaluate Models
    # ------------------------------------------

    results_df = evaluate_all_models(
        models=models,
        predictions=predictions,
        y_test=y_test,
        X_train=X_train_scaled,
        y_train=y_train
    )

    # ------------------------------------------
    # Best Model
    # ------------------------------------------

    best_model = get_best_model(
        results_df
    )

    # ------------------------------------------
    # Feature Importance
    # ------------------------------------------

    feature_names = [
        "Age",
        "Systolic BP",
        "Diastolic",
        "BS",
        "Body Temp",
        "BMI",
        "Previous Complications",
        "Preexisting Diabetes",
        "Gestational Diabetes",
        "Mental Health",
        "Heart Rate"
    ]

    rf_model = models["Random Forest"]

    feature_importance_df = get_feature_importance(
        rf_model,
        feature_names
    )

    print("\nEvaluation pipeline completed successfully.")