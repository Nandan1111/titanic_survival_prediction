# titanic_survival_prediction

                                                       Project Overview

This project aims to predict whether a passenger survived the Titanic disaster using machine learning classification models. The dataset includes features like age, gender, ticket class, fare, and cabin information. The project follows a structured machine learning workflow, including data preprocessing, model training, and evaluation.

                                                     Dataset Information

Source: Titanic dataset (e.g.,Kaggle)

Features Used:

Pclass - Ticket class (1st, 2nd, 3rd)
Sex - Gender (Male/Female)
Age - Passenger's age
SibSp - Number of siblings/spouses aboard
Parch - Number of parents/children aboard
Fare - Ticket fare
Embarked - Port of Embarkation

                        Project structure

titanic_survival_prediction/
│── data/                  # Dataset (if included)
│── src/                   # Source code
│── notebooks/             # Jupyter Notebooks (if any)
│── models/                # Saved trained model
│── README.md              # Project Documentation
│── requirements.txt       # Dependencies
│── .gitignore             # Ignore unnecessary files


                                   Python Code

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])  # Drop irrelevant columns
    return df
# Preprocessing pipeline
def preprocess_data(df):
    num_features = ["Age", "Fare"]
    cat_features = ["Sex", "Embarked", "Pclass", "SibSp", "Parch"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor
# Train model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }
# Main execution
if __name__ == "__main__":
    file_path = "/content/tested.csv"  # Update with actual file path
    df = load_data(file_path)
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    model = train_model(X_train, y_train)
    results = evaluate_model(model, X_test, y_test)

    print("Model Performance:", results)






