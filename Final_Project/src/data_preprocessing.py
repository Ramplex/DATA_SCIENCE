import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    DataFrame: The loaded data.
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """
    Preprocess the data by removing the 'Unnamed: 0' column,
    separating features and labels, and applying SMOTE for 
    handling class imbalance.

    Parameters:
    data (DataFrame): The raw data.

    Returns:
    tuple: The resampled feature matrix (X_res) and target vector (y_res).
    """
    # Remove the 'Unnamed: 0' column
    data = data.drop(columns=['Unnamed: 0'])
    
    # Separate features and the target label
    X = data.drop(columns=['quality'])
    y = data['quality']
    
    # Apply SMOTE for balancing the classes
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    return X_res, y_res

def split_data(X, y, test_size=0.4, random_state=42):
    """
    Split the data into training and testing sets.

    Parameters:
    X (DataFrame): The feature matrix.
    y (Series): The target vector.
    test_size (float): The proportion of the data to include in the test split.
    random_state (int): Random seed for reproducibility.

    Returns:
    tuple: Training and testing sets for features and labels.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
