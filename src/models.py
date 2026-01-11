"""
2.3 Machine Learning Implementation
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, confusion_matrix, precision_score, recall_score
)


def train_linear_regression(X: pd.DataFrame, y: pd.Series, save_path: str = None):
    """
    Train a Linear Regression model to predict a numerical target variable (final grade).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix used for training the model
    y : pd.Series
        Target variable representing student final grade (G3)
    save_path : str, optional
        Directory path where model outputs and evaluation metrics are saved

    Returns
    -------
    model : LinearRegression
        Trained Linear Regression model
    metrics : dict
        Dictionary containing R-squared (R²) and Mean Squared Error (MSE)

    Saved Outputs
    -------------
    - linear_regression_metrics.csv:
        Contains R² and MSE values
    - linear_regression_predictions.csv:
        Contains actual values, predicted values, and residuals
    """

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred)
    }

    # Save predictions with residuals
    pred_df = pd.DataFrame({
        "actual_G3": y_test,
        "predicted_G3": y_pred,
        "residual": y_test - y_pred
    })

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        pred_df.to_csv(os.path.join(save_path, "linear_regression_predictions.csv"), index=False)
        metrics_file = os.path.join(save_path, "linear_regression_metrics.csv")
        pd.DataFrame([metrics]).to_csv(metrics_file, index=False)

    return model, metrics


def train_decision_tree_classifier(X: pd.DataFrame, y: pd.Series, save_path: str = None):
    """
    Train a Decision Tree Classifier to predict student pass/fail outcomes.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix used for training the model
    y : pd.Series
        Binary target variable indicating pass (1) or fail (0)
    save_path : str, optional
        Directory path where model outputs and evaluation metrics are saved

    Returns
    -------
    model : DecisionTreeClassifier
        Trained Decision Tree classification model
    metrics : dict
        Dictionary containing accuracy, precision, recall, and confusion matrix

    Saved Outputs
    -------------
    - decision_tree_classifier_metrics.csv:
        Contains accuracy, precision, recall, and confusion matrix values
    - decision_tree_predictions.csv:
        Contains actual and predicted pass/fail labels
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "Confusion_Matrix": confusion_matrix(y_test, y_pred).tolist()  # convert to list for saving
    }

    pred_df = pd.DataFrame({
        "actual_pass_fail": y_test,
        "predicted_pass_fail": y_pred
    })

    pred_df.to_csv(
        os.path.join(save_path, "decision_tree_predictions.csv"),
        index=False
    )

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        metrics_file = os.path.join(save_path, "decision_tree_classifier_metrics.csv")
        # Save confusion matrix as separate columns
        cm = metrics["Confusion_Matrix"]
        cm_df = pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])
        metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items() if k != "Confusion_Matrix"})
        metrics_df = pd.concat([metrics_df, cm_df.reset_index(drop=True)], axis=1)
        metrics_df.to_csv(metrics_file, index=False)

    return model, metrics


def train_logistic_regression(X: pd.DataFrame, y: pd.Series, save_path: str = None):
    """
    Train a Logistic Regression model to predict student pass/fail outcomes.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix used for training the model
    y : pd.Series
        Binary target variable indicating pass (1) or fail (0)
    save_path : str, optional
        Directory path where model outputs and evaluation metrics are saved

    Returns
    -------
    model : LogisticRegression
        Trained Logistic Regression model
    metrics : dict
        Dictionary containing accuracy, precision, recall, and confusion matrix

    Saved Outputs
    -------------
    - logistic_regression_metrics.csv:
        Contains accuracy, precision, recall, and confusion matrix values
    - logistic_regression_predictions.csv:
        Contains actual labels, predicted labels, and predicted probabilities
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "Confusion_Matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    y_prob = model.predict_proba(X_test)[:, 1]

    pred_df = pd.DataFrame({
        "actual_pass_fail": y_test,
        "predicted_pass_fail": y_pred,
        "predicted_probability": y_prob
    })

    pred_df.to_csv(
        os.path.join(save_path, "logistic_regression_predictions.csv"),
        index=False
    )

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        metrics_file = os.path.join(save_path, "logistic_regression_metrics.csv")
        cm = metrics["Confusion_Matrix"]
        cm_df = pd.DataFrame(cm, index=["Actual_0", "Actual_1"], columns=["Pred_0", "Pred_1"])
        metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items() if k != "Confusion_Matrix"})
        metrics_df = pd.concat([metrics_df, cm_df.reset_index(drop=True)], axis=1)
        metrics_df.to_csv(metrics_file, index=False)

    return model, metrics
