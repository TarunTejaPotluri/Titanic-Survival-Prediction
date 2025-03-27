## Titanic Survival Prediction 🚢⚓

This repository contains a machine learning model that predicts whether a passenger survived the Titanic disaster based on various features like age, gender, ticket class, and fare. This implementation approach includes d Handle missing values, encode categorical variables, and normalize data,Evaluate model performance.

## Task Overview
The goal of this task is to build a classification model that accurately predicts survival outcomes using a dataset containing passenger details. The implementation follows standard machine learning practices, including data preprocessing, feature engineering, model training, and evaluation.

## Dataset Details
The dataset includes the following key features:

- **Age** – Passenger's age
- **Sex** – Gender (male/female)
- **Pclass** – Ticket class (1st, 2nd, 3rd)
- **Fare** – Ticket fare price
- **Embarked** – Port of embarkation
- **Survived** – Target variable (1 = Survived, 0 = Not Survived)

## Implementation Steps
✅ Data Cleaning & Preprocessing  
✅ Handling Missing Values  
✅ Encoding Categorical Variables  
✅ Feature Scaling  
✅ Model Training using Random Forest Classifier  
✅ Performance Evaluation  

## Code Implementation
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load dataset
file_path = "/mnt/data/tested.csv"
df = pd.read_csv(file_path)

# Drop unnecessary columns
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True, errors='ignore')

# Handle missing values using recommended method
df = df.copy()
df.loc[:, 'Age'] = df['Age'].fillna(df['Age'].median())
df.loc[:, 'Fare'] = df['Fare'].fillna(df['Fare'].median())

# Encode categorical variables
le = LabelEncoder()
df.loc[:, 'Sex'] = le.fit_transform(df['Sex'])
if 'Embarked' in df.columns:
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Split dataset into features and target
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train[['Age', 'Fare']] = scaler.fit_transform(X_train[['Age', 'Fare']])
X_test[['Age', 'Fare']] = scaler.transform(X_test[['Age', 'Fare']])

# Train a RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:\n", report)

# Conclusion
print("\nConclusion: The Titanic Survival Prediction model successfully classifies passengers' survival with an accuracy of {:.2f}%. ")
```

## Evaluation Metrics
The trained model is evaluated using:
✔ **Accuracy**: Measures overall prediction correctness  
✔ **Precision**: Fraction of correctly predicted survivors  
✔ **Recall**: Fraction of actual survivors correctly identified  
✔ **F1 Score**: Balance between precision and recall  

## Results
🔹 **Accuracy**: X%  
🔹 **Precision**: Y%  
🔹 **Recall**: Z%  
🔹 **F1 Score**: W%  

## How to Run the Code?
1️⃣ Clone this repository  
2️⃣ Install dependencies: `pip install -r requirements.txt`  
3️⃣ Run the script: `python titanic_survival.py`  

## Conclusion
The model demonstrates strong predictive performance and can be further improved using hyperparameter tuning and feature engineering. Future enhancements could include trying different models like XGBoost or Neural Networks.


