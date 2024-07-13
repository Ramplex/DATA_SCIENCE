from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on the test data.

    Parameters:
    model: The trained model to evaluate.
    X_test (DataFrame): The feature matrix for the test set.
    y_test (Series): The true labels for the test set.

    Returns:
    tuple: The accuracy score and classification report.
    """
    # Predict the labels for the test set
    y_pred = model.predict(X_test)
    
    # Calculate the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate a classification report
    report = classification_report(y_test, y_pred)
    
    return accuracy, report
