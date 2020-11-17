
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt

#database
TrainingDataSet = pd.read_excel("../Database/MainDataset.xlsx").iloc[:200]
# base on database we will set iloc
"""Scaler"""
scaler=MinMaxScaler()
TrainingDataSet=scaler.fit_transform(TrainingDataSet)
print(TrainingDataSet.shape)

x_train,y_train=[],[]
for i in range(60,TrainingDataSet.shape[0]):
    x_train.append(TrainingDataSet[i-60:i])
    y_train.append(TrainingDataSet[i,6])

x_train,y_train=np.array(x_train),np.array(y_train)
x_test,y_test=list(),list()
TestingDataSet = pd.read_excel("../Database/MainDataset.xlsx").iloc[:200]

TestingDataSet=scaler.fit_transform(TestingDataSet)

for j in range(60,TestingDataSet.shape[0]):
    x_test.append(TestingDataSet[j-60:j])
    y_test.append(TestingDataSet[j,6])

x_test,y_test=np.array(x_test),np.array(y_test)


regressor= Sequential()

regressor.add(LSTM(units= 100,activation="relu", return_sequences=True, input_shape=(x_train.shape[1],7)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units= 120,activation="relu", return_sequences=True))
regressor.add(Dropout(0.30))

regressor.add(LSTM(units= 160,activation="relu", return_sequences=True))
regressor.add(Dropout(0.40))
regressor.add(LSTM(units= 200,activation="relu", return_sequences=False)) #this false is vvi
regressor.add(Dropout(0.50))
regressor.add(Dense(units=1))

regressor.summary()

es = EarlyStopping(monitor='val_loss', patience = 3, verbose=1, mode='min')
regressor.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
history=regressor.fit(x_train,y_train,epochs=10,batch_size=32, validation_split = 0.1,callbacks=[es])








""" Evaluate Model on Test data"""

regressor.evaluate(x_test,y_test,batch_size=32, verbose=1)
plt.plot(history.history['loss'],label='Trainig Loss')
plt.plot(history.history['val_loss'],label="validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

"""Accuracy"""
regressor.evaluate(x_test,y_test,batch_size=32, verbose=1)
plt.plot(history.history['accuracy'],label='Trainig Accuracy')
plt.plot(history.history['val_accuracy'],label="validation Accuracy")
plt.title("Training and validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
