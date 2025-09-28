# algorithms I've elected to use: Support Vector Machine & Logisitic Regression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collections import defaultdict

# Configurations
RandomState = 42
Features = ['year', 'temp', 'humidity', 'rainfall', 'drought_code', 'buildup_index', 'day', 'month', 'wind_speed']
Target = 'fire'

training_file = 'wildfires_training.csv'
test_file = 'wildfires_test.csv'

#Loading Data
def load_data ():
    #Data Loading Portion
    try:
        if not os.path.exists(training_file) or not os.path.exists(test_file):
            raise FileNotFoundError("One of the data files are missing")
        
        print("Loading data")
        training_data = pd.read_csv(training_file)
        testing_data = pd.read_csv(test_file)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    # Data Successfully Loaded - Ending Loading Portion
    print("Data loaded successfully. {len(training_data)} training samples and {len(testing_data)} testing samples.")
    print("-" * 50)

    # Checking
    required_columns = Features + [Target]
    if not all(col in training_data.columns for col in required_columns) or \
         not all(col in testing_data.columns for col in required_columns):
        print("Error: One or more columns are missing in the loaded files.")
        print(f"The Number of Required Columns is {required_columns}")
        exit(1)

    # Conversion to target variable
    # Fire: Yes or No
    # Yes = 1; No = 0

    le = LabelEncoder()
    y_train = le.fit_transform(training_data[Target])
    y_test = le.transform(testing_data[Target])

    # Features
    x_train = training_data[Features]
    x_test = testing_data[Features]

    # Starting StandingScaler for Feature Scaling
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, y_train, x_test_scaled, y_test


def evaluate_model(model, x_train, y_train, x_test, y_test, model_name, parameters):
    model.fit(x_train, y_train)

    y_train_prediction = model.predict(x_train)
    y_test_prediction = model.predict(x_test)

    training_accuracy = accuracy_score(y_train, y_train_prediction)
    testing_accuracy = accuracy_score(y_test, y_test_prediction)

    result = {
        'Model': model_name,
        'Hyperparameters': str(parameters),
        "Training Accuracy": f'{training_accuracy:.4f}',
        'Testing Accuracy': f'{testing_accuracy:.4f}'
    }

    return result