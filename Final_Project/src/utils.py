def print_evaluation(accuracy, report):
    """
    Print the evaluation results, including accuracy and classification report.

    Parameters:
    accuracy (float): The accuracy score of the model.
    report (str): The classification report of the model.
    """
    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(report)
