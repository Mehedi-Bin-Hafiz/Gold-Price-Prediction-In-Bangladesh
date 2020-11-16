
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#database
MainDatabase = pd.read_excel("../Database/MainDataset.xlsx")
# base on database we will set iloc
X = MainDatabase.iloc[:, 0:5].values  #independent variables
X=X.astype(int)
print(X)
Y = MainDatabase.iloc[ : , -1].values #dependent variables
print(Y)




# # define base model
# def baseline_model():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(32, input_dim=5, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(1, kernel_initializer='normal'))
# 	# Compile model
# 	model.compile(loss='mean_squared_error', optimizer='adam')
# 	return model
# # evaluate model
# estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
# kfold = KFold(n_splits=10)
# results = cross_val_score(estimator, X, Y, cv=kfold)
# print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
#



###Sequestial model

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation


x_train,y_train,x_test,y_test=X[:2100],Y[:2100],X[2100:],Y[2100:]
model = Sequential(
    [
        Dense(256,),
        Activation('relu'),
        Dropout(0.5),
        Dense(128, ),
        Activation('relu'),
        Dropout(0.5),
        Dense(64, ),
        Activation('relu'),
        Dropout(0.5),
        Activation('softmax')
    ]
)





model.summary()

model.compile(optimizer='adam',loss= 'categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train,y_train,epochs=100,batch_size=32,validation_split = 0.1, )



""" Evaluate Model on Test data"""

model.evaluate(x_test,y_test,batch_size=32, verbose=1)
plt.plot(history.history['loss'],label='Trainig Loss')
plt.plot(history.history['val_loss'],label="validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

"""Accuracy"""
model.evaluate(x_test,y_test,batch_size=32, verbose=1)
plt.plot(history.history['accuracy'],label='Trainig Accuracy')
plt.plot(history.history['val_accuracy'],label="validation Accuracy")
plt.title("Training and validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
