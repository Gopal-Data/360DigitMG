import pandas as pd
import numpy as np
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
#loading the data 
forest = pd.read_csv("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 17\\forestfires.csv")
forest.describe()
#selecting the selected variables 
forest1=forest.iloc[:,[30,6,7,8,9,10]]
#spilting the data 
train,test = train_test_split(forest1, test_size = 0.30)
train_X = train.iloc[:, 1:]
train_y = train.iloc[:, 0]
test_X  = test.iloc[:, 1:]
test_y  = test.iloc[:, 0]
#SVM modeling 
# kernel = linear
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X, train_y)
pred_test_linear = model_linear.predict(test_X)
np.mean(pred_test_linear == test_y) #.99

pred_train_linear = model_linear.predict(train_X)
np.mean(pred_train_linear == train_y) #.99

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X, train_y)
pred_test_rbf = model_rbf.predict(test_X)
np.mean(pred_test_rbf==test_y)  #.89
 
pred_train_rbf = model_rbf.predict(train_X)
np.mean(pred_train_rbf==train_y)  #.90

#kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X, train_y)
pred_test_poly = model_poly.predict(test_X)
np.mean(pred_test_poly==test_y)  #.93
pred_train_poly = model_poly.predict(train_X)
np.mean(pred_train_poly==train_y)  #.81

#kernel = sigmoid
model_sigmoid = SVC(kernel = "sigmoid")
model_sigmoid.fit(train_X, train_y)
pred_test_sigmoid = model_sigmoid.predict(test_X)
np.mean(pred_test_sigmoid==test_y)  #.87 

pred_train_sigmoid = model_sigmoid.predict(train_X)
np.mean(pred_train_sigmoid==train_y)  #.86 