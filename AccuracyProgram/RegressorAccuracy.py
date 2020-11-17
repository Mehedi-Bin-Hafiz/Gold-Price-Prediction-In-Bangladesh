from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV,BayesianRidge
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
#database
MainDatabase = pd.read_excel("../Database/MainDataset.xlsx")
# base on database we will set iloc
x = MainDatabase.iloc[:, 0:5].values  #independent variables
x=x.astype(int)
print(x)
y = MainDatabase.iloc[ : , -1].values #dependent variables
print(y)




thirtypercent=0.30  # training size 70%
fourtypercent=0.40   # training size 60%
fiftypercent=0.50    # training size 50%
sixtypercent=0.60    # training size 40%
seventypercent=0.70   # training size 30%



#naive bayes
print("\n########## Gradient Boosting ###########")
gnb = BayesianRidge()

X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=thirtypercent, random_state=0)
clf_gb = GradientBoostingRegressor(n_estimators=200)
clf_gb.fit(X_train,y_train)
pred = clf_gb.predict(X_test)
A_score=clf_gb.score(X_test,pred)
mse = mean_squared_error(y_test,pred)
print("mse=", mse)
rmse = np.sqrt(mse)
print('rmse=',rmse)
print("test size=30, accuracy = {0:.2f}".format(100*A_score),"%")


X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=fourtypercent, random_state=0)
clf_gb = GradientBoostingRegressor(n_estimators=200)
clf_gb.fit(X_train,y_train)
pred = clf_gb.predict(X_test)
A_score=clf_gb.score(X_test,pred)
mse = mean_squared_error(y_test,pred)
print("mse=", mse)
rmse = np.sqrt(mse)
print('rmse=',rmse)
print("test size=40, accuracy = {0:.2f}".format(100*A_score),"%")

X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=fiftypercent, random_state=0)
clf_gb = GradientBoostingRegressor(n_estimators=200)
clf_gb.fit(X_train,y_train)
pred = clf_gb.predict(X_test)
A_score=clf_gb.score(X_test,pred)
mse = mean_squared_error(y_test,pred)
print("mse=", mse)
rmse = np.sqrt(mse)
print('rmse=',rmse)
print("test size=50, accuracy = {0:.2f}".format(100*A_score),"%")

X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=sixtypercent, random_state=0)
clf_gb = GradientBoostingRegressor(n_estimators=200)
clf_gb.fit(X_train,y_train)
pred = clf_gb.predict(X_test)
A_score=clf_gb.score(X_test,pred)
mse = mean_squared_error(y_test,pred)
print("mse=", mse)
rmse = np.sqrt(mse)
print('rmse=',rmse)
print("test size=60, accuracy = {0:.2f}".format(100*A_score),"%")

X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=seventypercent, random_state=0)
clf_gb = GradientBoostingRegressor(n_estimators=200)
clf_gb.fit(X_train,y_train)
pred = clf_gb.predict(X_test)
A_score=clf_gb.score(X_test,pred)
mse = mean_squared_error(y_test,pred)
print("mse=", mse)
rmse = np.sqrt(mse)
print('rmse=',rmse)
print("test size=70, accuracy = {0:.2f}".format(100*A_score),"%")


print('################(Neural Network)MLP regressor ################')
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=thirtypercent, random_state=0)

regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
pred = regr.predict(X_test)
A_score=regr.score(X_test,pred)
mse = mean_squared_error(y_test,pred)
print("mse=", mse)
rmse = np.sqrt(mse)
print('rmse=',rmse)
print("test size=30, accuracy = {0:.2f}".format(100*A_score),"%")



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=fourtypercent, random_state=0)

regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
pred = regr.predict(X_test)
A_score=regr.score(X_test,pred)
mse = mean_squared_error(y_test,pred)
print("mse=", mse)
rmse = np.sqrt(mse)
print('rmse=',rmse)
print("test size=40, accuracy = {0:.2f}".format(100*A_score),"%")


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=fiftypercent, random_state=0)

regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
pred = regr.predict(X_test)
A_score=regr.score(X_test,pred)
mse = mean_squared_error(y_test,pred)
print("mse=", mse)
rmse = np.sqrt(mse)
print('rmse=',rmse)
print("test size=50, accuracy = {0:.2f}".format(100*A_score),"%")


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=sixtypercent, random_state=0)

regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
pred = regr.predict(X_test)
A_score=regr.score(X_test,pred)
mse = mean_squared_error(y_test,pred)
print("mse=", mse)
rmse = np.sqrt(mse)
print('rmse=',rmse)
print("test size=60, accuracy = {0:.2f}".format(100*A_score),"%")


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(x, y,test_size=seventypercent, random_state=0)

regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
pred = regr.predict(X_test)
A_score=regr.score(X_test,pred)
mse = mean_squared_error(y_test,pred)
print("mse=", mse)
rmse = np.sqrt(mse)
print('rmse=',rmse)
print("test size=70, accuracy = {0:.2f}".format(100*A_score),"%")
