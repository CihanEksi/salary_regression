import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import datetime
import common
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard


## Static variables
csv_file = "salary_excel.csv"
unwanted_columns = ['RowNumber', 'CustomerId', 'Surname']
cat_fields = ['Gender']
ordinal_fields = ['Geography']
target= 'EstimatedSalary'

# model settings
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss = tf.keras.losses.MeanAbsoluteError()
metrics=['mae']
patience=10


## Encoders and scalers
label_encoder=LabelEncoder()
onehotencoder=OneHotEncoder()
scaler = StandardScaler()

## directories
log_dir = "regressionlogs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
label_pkl_file_name = "label_encoder.pkl"
ordinal_pkl_file_name = "ordinal_encode.pkl"
scaler_pkl_file_name = "scaler.pkl"

def train():
    data = common.read_data(csv_file)
    data = common.clean_csv(data, fields=unwanted_columns, axis=1)
    common.log(data.head())

    ## Encode categorical fields
    data = common.categorical_encode(data, fields=cat_fields,label_encoder=label_encoder)
    common.log(data.head())

    ## Encode ordinal fields
    ordinal_encoded_fields = common.ordinal_encode(data, fields=ordinal_fields,onehotencoder=onehotencoder)
    data = common.concat_ordinal(data=data,ordinal_encoded_fields=ordinal_encoded_fields,ordinal_fields=ordinal_fields,encoder=onehotencoder)
    common.log(data.head())

    ## save the encoders
    common.save_pkl(label_encoder,label_pkl_file_name)
    common.save_pkl(onehotencoder,ordinal_pkl_file_name)

    ## target and feature definition
    features_x = data.drop(target,axis=1) ## Features
    common.log(features_x.shape)
    
    target_y = data[target] ## Target
    common.log(target_y.shape)

    ## Split the data in training and testing sets
    x_train,x_test,y_train,y_test=train_test_split(features_x,target_y,test_size=0.2,random_state=42)

    ## Scale the data
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    common.log(x_train)

    ## Save the scaler
    common.save_pkl(scaler,scaler_pkl_file_name)

    # build the model
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1],)))
    model.add(Dense(64, input_dim=11, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    # compile the model
    model.compile(optimizer=optimizer, loss=loss,metrics=metrics)

    ## Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)


    # Train the model
    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=100,
        callbacks=[early_stopping, tensorboard]
    )

    common.log(model.summary())

    ## Evaluate model on the test data
    test_loss,test_mae=model.evaluate(x_test,y_test)


    model.save('regression_model.keras')





if __name__ == "__main__":
    train()