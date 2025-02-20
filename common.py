import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime

def read_data(file_to_train):
    data=pd.read_csv(file_to_train)
    return data

def clean_csv(data,fields,axis):
    return data.drop(fields,axis=axis)

def categorical_encode(data,fields,label_encoder):
    for field in fields:
        data[field]=label_encoder.fit_transform(data[field])
    return data

def ordinal_encode(data,fields,onehotencoder):
    return onehotencoder.fit_transform(data[fields]).toarray()

def concat_ordinal(data,ordinal_encoded_fields,ordinal_fields,encoder):
    encoded_ordinal_columns= encoder.get_feature_names_out(ordinal_fields)
    encoded_df = pd.DataFrame(data=ordinal_encoded_fields,columns=encoded_ordinal_columns)

    ## Drop the original ordinal fields
    data = data.drop(ordinal_fields,axis=1)

    ## concat the encoded ordinal fields
    data = pd.concat([data,encoded_df],axis=1)

    return data

def log(text):
    print("")
    print(text)
    print("")

def save_pkl(encoder,filename):
    with open(filename,'wb') as file:
        pickle.dump(encoder,file)

