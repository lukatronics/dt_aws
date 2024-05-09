from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np

# Reads a dataset and returns features and labels.
def read_data(filename):
    
    data = pd.read_csv(filename).fillna("NA").astype(str)
    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values
    return X, y

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

def evaluate_model(dt, X_test, y_test):
    """
    Evaluates the decision tree model on the test set.

    Parameters:
        clf (DecisionTreeClassifier): The trained decision tree classifier.
        X_test (np.ndarray): The feature matrix for the test set.
        y_test (np.ndarray): The labels for the test set.

    Returns:
        None
    """
    predictions = dt.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    """
    The main function to train and evaluate the decision tree model.
    """
    # Replace with your actual data file path
    train_file = "path/to/your/training/data.csv"
    test_file = "path/to/your/testing/data.csv"

    # Load the training and testing datasets
    X_train, y_train = read_data(train_file)
    X_test, y_test = read_data(test_file)

    # Train the decision tree model
    dt_train = train_decision_tree(X_train, y_train, max_depth=5)

    # Evaluate the model on the test set
    evaluate_model(dt_train, X_test, y_test)