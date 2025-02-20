import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import json

#statics
label_pkl_file_name = "label_encoder.pkl"
ordinal_pkl_file_name = "ordinal_encode.pkl"
scaler_pkl_file_name = "scaler.pkl"
model_file_name = "regression_model.keras"
categorical_fields = ['Gender']
ordinal_fields = ['Geography']

# Example input data
input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    "Exited": 0
}


### Load the trained model, scaler pickle,onehot
model=load_model(model_file_name)

## load the encoders and scaler
with open(ordinal_pkl_file_name,'rb') as file:
    ordinal_encoder=pickle.load(file)

with open(label_pkl_file_name, 'rb') as file:
    categorical_encoder= pickle.load(file)

with open(scaler_pkl_file_name, 'rb') as file:
    scaler = pickle.load(file)

def log(text):
    print("")
    print(text)
    print("")

def ordinal_encode(input):
     dataFrames = []
     for field in ordinal_fields:
        value = input[field]
        encodedField = ordinal_encoder.transform([value]).toarray()

        columns= ordinal_encoder.get_feature_names_out([field])

        encodedDataFrame=pd.DataFrame(encodedField,columns=columns)
        dataFrames.append(encodedDataFrame)
     return encodedDataFrame

def categorical_encode(input):
    for field in categorical_fields:
        input[field] = categorical_encoder.transform(input[field])
    return input


def predict(input_data):
    input_df = pd.DataFrame([input_data])

    ## categorical encode the fields
    categorical_encode(input_df)

    ## Encode the categorical fields
    ordinal_encoded_frames = ordinal_encode(input_df)

    ## Drop the original ordinal fields
    input_df = pd.concat([input_df.drop(ordinal_fields,axis=1),ordinal_encoded_frames],axis=1)
    #log(input_df.head())

    ## Scale the input
    input_scaled = scaler.transform(input_df)
    #log(input_scaled)

    ## predict
    prediction = model.predict(input_scaled)
    #log(prediction)

    predictionRate = prediction[0][0]
    floatPrediction = float(predictionRate)


    print(json.dumps({"prediction": floatPrediction}))


    


if __name__ == "__main__":
    predict(input_data)
    