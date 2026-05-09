# ==========================================================
# FILE: train_models.py
# PURPOSE:
# Model training pipeline for maternal risk prediction
#
# FEATURES:
# - Logistic Regression
# - KNN
# - SVM
# - Decision Tree
# - Random Forest
# - Prediction generation
#
# OUTPUT:
# - trained models
# - model predictions
#
# AUTHOR:
# Maternal Risk Prediction Project
# ==========================================================


# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------
# Import preprocessing pipeline
# ------------------------------------------

from preprocessing import preprocess_data


# ==========================================================
# TRAIN LOGISTIC REGRESSION
# ==========================================================

def train_logistic_regression(X_train, y_train):
    """
    Trains Logistic Regression model.

    Parameters:
    ----------
    X_train : ndarray
        Scaled training features

    y_train : pandas.Series
        Training target labels

    Returns:
    -------
    model : LogisticRegression
        Trained Logistic Regression model
    """

    model = LogisticRegression(
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    print("Logistic Regression model trained successfully.")

    return model


# ==========================================================
# TRAIN KNN
# ==========================================================

def train_knn(X_train, y_train, n_neighbors=5):
    """
    Trains KNN classifier.

    Parameters:
    ----------
    X_train : ndarray

    y_train : pandas.Series

    n_neighbors : int
        Number of neighbors

    Returns:
    -------
    model : KNeighborsClassifier
        Trained KNN model
    """

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors
    )

    model.fit(
        X_train,
        y_train
    )

    print("KNN model trained successfully.")

    return model


# ==========================================================
# TRAIN SVM
# ==========================================================

def train_svm(X_train, y_train):
    """
    Trains Support Vector Machine classifier.

    Parameters:
    ----------
    X_train : ndarray

    y_train : pandas.Series

    Returns:
    -------
    model : SVC
        Trained SVM model
    """

    model = SVC(
        kernel="rbf",
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    print("SVM model trained successfully.")

    return model


# ==========================================================
# TRAIN DECISION TREE
# ==========================================================

def train_decision_tree(X_train, y_train):
    """
    Trains Decision Tree classifier.

    Parameters:
    ----------
    X_train : ndarray

    y_train : pandas.Series

    Returns:
    -------
    model : DecisionTreeClassifier
        Trained Decision Tree model
    """

    model = DecisionTreeClassifier(
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    print("Decision Tree model trained successfully.")

    return model


# ==========================================================
# TRAIN RANDOM FOREST
# ==========================================================

def train_random_forest(X_train, y_train):
    """
    Trains Random Forest classifier.

    Parameters:
    ----------
    X_train : ndarray

    y_train : pandas.Series

    Returns:
    -------
    model : RandomForestClassifier
        Trained Random Forest model
    """

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    print("Random Forest model trained successfully.")

    return model


# ==========================================================
# GENERATE PREDICTIONS
# ==========================================================

def generate_predictions(model, X_test):
    """
    Generates predictions using trained model.

    Parameters:
    ----------
    model : trained sklearn model

    X_test : ndarray

    Returns:
    -------
    y_pred : ndarray
        Predicted labels
    """

    y_pred = model.predict(X_test)

    return y_pred


# ==========================================================
# TRAIN ALL MODELS
# ==========================================================

def train_all_models(X_train, y_train, X_test):
    """
    Trains all machine learning models and
    generates predictions.

    Models:
    --------
    - Logistic Regression
    - KNN
    - SVM
    - Decision Tree
    - Random Forest

    Parameters:
    ----------
    X_train : ndarray

    y_train : pandas.Series

    X_test : ndarray

    Returns:
    -------
    models : dict
        Trained models

    predictions : dict
        Model predictions
    """

    # ------------------------------------------
    # Train Models
    # ------------------------------------------

    log_model = train_logistic_regression(
        X_train,
        y_train
    )

    knn_model = train_knn(
        X_train,
        y_train
    )

    svm_model = train_svm(
        X_train,
        y_train
    )

    dt_model = train_decision_tree(
        X_train,
        y_train
    )

    rf_model = train_random_forest(
        X_train,
        y_train
    )

    # ------------------------------------------
    # Store Models
    # ------------------------------------------

    models = {
        "Logistic Regression": log_model,
        "KNN": knn_model,
        "SVM": svm_model,
        "Decision Tree": dt_model,
        "Random Forest": rf_model
    }

    # ------------------------------------------
    # Generate Predictions
    # ------------------------------------------

    predictions = {}

    for model_name, model in models.items():

        predictions[model_name] = generate_predictions(
            model,
            X_test
        )

    print("All models trained successfully.")

    return models, predictions


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
    # Print Results
    # ------------------------------------------

    print("\nTrained Models:")

    for model_name in models.keys():

        print("-", model_name)

    print("\nPrediction Preview:")

    for model_name, y_pred in predictions.items():

        print(f"\n{model_name} Predictions:")
        print(y_pred[:10])