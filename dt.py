from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

"""
Author: Lucas(Jeonghoo) Yoo
Simple Decision Tree Project for AWS
"""

# Reads a dataset and returns features and labels.
def read_data(filename, encoders=None):
    """
    Reads a dataset, encodes categorical features, and returns features and labels.

    Parameters:
        filename (str): The path to the CSV file.
        encoders (list of LabelEncoder or None): The encoders for each feature column.

    Returns:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Label vector.
        encoders (list of LabelEncoder): The fitted encoders.
        label_encoder (LabelEncoder): The fitted label encoder.
    """
    data = pd.read_csv(filename).astype(str)
    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values

    # Initialize encoders for the feature columns if not provided
    if encoders is None:
        encoders = [LabelEncoder() for _ in range(X.shape[1])]

    # Fit and transform the feature columns using the provided or new encoders
    for i, encoder in enumerate(encoders):
        X[:, i] = encoder.fit_transform(X[:, i])

    # Fit and transform the label column
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X.astype(np.float32), y, encoders, label_encoder

def train_decision_tree(X, y, max_depth=5):
    """
    Trains a decision tree classifier.

    Parameters:
        X (np.ndarray): A 2D array of shape (n_samples, n_features) containing the training data.
        y (np.ndarray): A 1D array of shape (n_samples,) containing the training labels.
        max_depth (int): The maximum depth of the tree.

    Returns:
        DecisionTreeClassifier: A fitted decision tree classifier.
    """
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X, y)
    return dt

def evaluate_model(dt, X_test, y_test, target_names):
    """
    Evaluates the decision tree model on the test set.

    Parameters:
        dt (DecisionTreeClassifier): The trained decision tree classifier.
        X_test (np.ndarray): The feature matrix for the test set.
        y_test (np.ndarray): The labels for the test set.
        target_names (list of str): The class names for the classification report.

    Returns:
        None
    """
    predictions = dt.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, predictions, target_names=target_names, zero_division=0))

if __name__ == "__main__":
    """
    The main function to train and evaluate the decision tree model.
    """

    train_file = "aws_train_data.csv"
    test_file = "aws_test_data.csv"

    # Load the training and testing datasets
    X_train, y_train, feature_encoders, label_encoder = read_data(train_file)
    X_test, y_test, _, _ = read_data(test_file, feature_encoders)

    # Train the decision tree model
    dt_train = train_decision_tree(X_train, y_train, max_depth=5)

    # Evaluate the model on the test set
    target_names = label_encoder.classes_
    evaluate_model(dt_train, X_test, y_test, target_names)