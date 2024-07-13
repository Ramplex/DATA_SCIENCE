from src.data_preprocessing import load_data, preprocess_data, split_data
from src.model import train_model, save_model, load_model
from src.evaluation import evaluate_model
from src.utils import print_evaluation

def main():
    """
    Main function to load data, preprocess it, train a model, save the model,
    evaluate the model, and print the evaluation results.
    """
    # Load and preprocess the data
    # Load data from the CSV file
    data = load_data('../data/train.csv')
    
    # Preprocess the data to remove unnecessary columns and handle class imbalance
    X, y = preprocess_data(data)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train the model using the training data
    model = train_model(X_train, y_train)
    
    # Save the trained model to a file for future use
    save_model(model, '../src/model/wine_quality_model.pkl')
    
    # Evaluate the model using the testing data
    accuracy, report = evaluate_model(model, X_test, y_test)
    
    # Print the evaluation results
    print_evaluation(accuracy, report)

if __name__ == '__main__':
    # Execute the main function if this script is run directly
    main()
