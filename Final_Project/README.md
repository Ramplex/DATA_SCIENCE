# Wine Quality Prediction Project

This project aims to predict the quality of wine based on various chemical characteristics using machine learning techniques.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [GUI](#gui)

## Introduction

This project uses a Random Forest classifier to predict wine quality based on features such as acidity, residual sugar, chlorides, and alcohol content. The project includes data preprocessing, model training, evaluation, and a graphical user interface (GUI) for prediction.

## Dataset

The dataset used for this project is a wine quality dataset, typically available on platforms like Kaggle or UCI Machine Learning Repository. It includes the following features:

- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol
- Quality (target variable)

## Project Structure

wine-quality-prediction/
│
├── data/
│ ├── train.csv
│ ├── user_data.csv
│ └── inventory.csv
│
├── notebooks/
│ ├── EDA.ipynb
│
├── src/
│ ├── model/
│   └── wine_quality_model.pkl
│ ├── data_preprocessing.py
│ ├── evaluation.py
│ ├── model.py
│ ├── utils.py
│ └── main.py
│
├── gui/
│ └── gui.py
│
├── README.md
└── requirements.txt
## Usage

First, open the console inside the `final_project` folder. Then, execute the following commands to start the poetry virtual environment, optionally retrain the model, and launch the graphical user interface:

```sh
# Navigate to the final_project folder
cd final_project

# Start the poetry virtual environment
poetry shell

# (Optional) Retrain the model
python ..\main.py

# Launch the graphical user interface
python ..\gui.py
```

## Model Training
The model is trained using a Random Forest classifier with GridSearchCV for hyperparameter tuning. The training process involves:

Loading and preprocessing the data
Splitting the data into training and testing sets
Training the model with optimal hyperparameters

## Evaluation
The model's performance is evaluated using metrics such as accuracy and a classification report. The evaluation results are printed in the console.

## GUI
The project includes a Tkinter-based GUI that allows users to input wine features and predict the quality of wine. Users can also save the input data and view the wine inventory.