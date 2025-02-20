# Salary Regression Model

In this project I trained a machine learning model to predict customer estimated salary based on various features.
Solution method is Artificial Neural Network (ANN).

## Requirements

- Python 3.x
- TensorFlow
- pandas
- pickle
- scikit-learn

## Files

- `regression_model.keras`: Trained TensorFlow model
- `label_encoder.pkl`: Pickle file for categorical encoding
- `ordinal_encode.pkl`: Pickle file for ordinal encoding
- `scaler.pkl`: Pickle file for feature scaling

## Prediction Steps

1. **Data Preprocessing**
   - Load the model, encoders, and scaler
   - Convert input data to pandas DataFrame
   - Apply categorical encoding for the Gender field
   - Apply ordinal encoding for the Geography field
   - Scale the features using the scaler

2. **Model Prediction**
   - Feed the preprocessed data into the model
   - Get the prediction output
   - Return the prediction as a JSON response

## Input Features

- `CreditScore`: Customer's credit score (numeric)
- `Geography`: Customer's location (France, Spain, Germany)
- `Gender`: Customer's gender (Male, Female)
- `Age`: Customer's age (numeric)
- `Tenure`: Number of years as a customer (numeric)
- `Balance`: Account balance (numeric)
- `NumOfProducts`: Number of bank products used (numeric)
- `HasCrCard`: Has credit card (1 = Yes, 0 = No)
- `IsActiveMember`: Active member status (1 = Yes, 0 = No)
- `Exited`: Customer churn status (1 = Yes, 0 = No)

## What We Did?

1. **Loading Resources**
   - Loaded the trained model using `load_model`.
   - Loaded the encoders and scaler from their respective pickle files.

2. **Data Encoding**
   - Defined functions for ordinal and categorical encoding.
   - Encoded the input data using these functions.

3. **Data Scaling**
   - Scaled the encoded input data using the loaded scaler.

4. **Prediction**
   - Made predictions using the preprocessed and scaled data.
   - Converted the prediction result to a JSON format for easy interpretation.