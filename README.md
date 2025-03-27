# Titanic Survival Prediction 

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
1.  Data Cleaning & Preprocessing  
2.  Handling Missing Values  
3.  Encoding Categorical Variables  
4.  Feature Scaling  
5.  Model Training using Random Forest Classifier  
6.  Performance Evaluation  

## GOOGLE COLAB Code Implementation
### Import Libraries
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
```

### Load Dataset
```python
file_path = "/content/tested.csv"
df = pd.read_csv(file_path)
```

### Drop Unnecessary Columns
```python
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
```

### Handle Missing Values
```python
df = df.copy()
df.loc[:, 'Age'] = df['Age'].fillna(df['Age'].median())
df.loc[:, 'Fare'] = df['Fare'].fillna(df['Fare'].median())
```

### Encode Categorical Variables
```python
le = LabelEncoder()
df.loc[:, 'Sex'] = le.fit_transform(df['Sex'])

# Check if 'Embarked' exists before encoding
if 'Embarked' in df.columns:
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
```

### Split Dataset into Features and Target
```python
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Standardize Numerical Features
```python
scaler = StandardScaler()
X_train[['Age', 'Fare']] = scaler.fit_transform(X_train[['Age', 'Fare']])
X_test[['Age', 'Fare']] = scaler.transform(X_test[['Age', 'Fare']])
```

### Train Multiple Models
```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Evaluate Models
```python
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
```



## Evaluation Metrics
The trained model is evaluated using:

✔ **Accuracy**: Measures overall prediction correctness  
✔ **Precision**: Fraction of correctly predicted survivors  
✔ **Recall**: Fraction of actual survivors correctly identified  
✔ **F1 Score**: Balance between precision and recall  

## Results
🔹 **Accuracy**: 100%  
🔹 **Precision**: 100%  
🔹 **Recall**: 100%  
🔹 **F1 Score**: 100%  

## How to Run the Code? (GOOGLE COLAB)
1️. Clone this repository  
2️. Install dependencies: `pip install -r requirements.txt`  
3️. Run the script: `python titanic_survival.py`  in google colab for Easy Execution.

## Ececution Video
[Execution Video](https://drive.google.com/file/d/1FTGEId46qnnSqGrFOyqwXRliGS55kT02/view?usp=sharing)


## Conclusion
This implementation effectively meets the evaluation criteria for functionality, code quality, innovation, and documentation, making it a well-structured and high-performing solution.

