from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib

def train_model(X_train, y_train):
    """
    Train a RandomForestClassifier using GridSearchCV to find the best hyperparameters.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
    
    Returns:
        RandomForestClassifier: The best estimator found by GridSearchCV.
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [8, 10, 12],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [4, 6],
        'bootstrap': [True],
        'class_weight': ['balanced', 'balanced_subsample', {3: 1, 4: 1, 5: 2, 6: 2, 7: 1, 8: 1, 9: 1}]
    }
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def save_model(model, file_path):
    """
    Save the trained model to a file.
    
    Args:
        model (RandomForestClassifier): The trained model to be saved.
        file_path (str): The file path where the model will be saved.
    """
    joblib.dump(model, file_path)

def load_model(file_path):
    """
    Load a trained model from a file.
    
    Args:
        file_path (str): The file path from where the model will be loaded.
    
    Returns:
        RandomForestClassifier: The loaded model.
    """
    model = joblib.load(file_path)
    return model
