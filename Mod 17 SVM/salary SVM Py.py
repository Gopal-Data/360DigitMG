import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

salary= pd.read_csv("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 17\\SalaryData_Test.csv")
list(salary.columns)
salary1 = salary.iloc[0:500:,]

#typecasting the categorical variables 

label_encoder = preprocessing.LabelEncoder() 
salary1['workclass']= label_encoder.fit_transform(salary1['workclass']) 
salary1['education']= label_encoder.fit_transform(salary1['education']) 
salary1['maritalstatus']= label_encoder.fit_transform(salary1['maritalstatus']) 
salary1['occupation']= label_encoder.fit_transform(salary1['occupation']) 
salary1['relationship']= label_encoder.fit_transform(salary1['relationship']) 
salary1['race']= label_encoder.fit_transform(salary1['race']) 
salary1['sex']= label_encoder.fit_transform(salary1['sex']) 
salary1['native']= label_encoder.fit_transform(salary1['native']) 

#spilting the data
train,test = train_test_split(salary1, test_size = 0.20) 
train_X = salary1.iloc[:,0:13]
train_y = salary1.iloc[:,13]
test_X  = salary1.iloc[:,0:13]
test_y  = salary1.iloc[:,13]
 
# kernel = linear
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X, train_y)
pred_test_linear = model_linear.predict(test_X)
np.mean(pred_test_linear == test_y) #.802

pred_train_linear = model_linear.predict(train_X)
np.mean(pred_train_linear == train_y) #.802

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X, train_y)
pred_test_rbf = model_rbf.predict(test_X)
np.mean(pred_test_rbf==test_y)  #.804

pred_train_rbf = model_rbf.predict(train_X)
np.mean(pred_train_rbf==train_y)  #.804

#kernel = poly 
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X, train_y)
pred_test_poly = model_poly.predict(test_X)
np.mean(pred_test_poly==test_y)  #0.804 

pred_train_poly = model_poly.predict(train_X)
np.mean(pred_train_poly==train_y)  #0.804

#kernel = sigmoid
model_sigmoid = SVC(kernel = "sigmoid")
model_sigmoid.fit(train_X, train_y)
pred_test_sigmoid = model_sigmoid.predict(test_X)
np.mean(pred_test_sigmoid==test_y)  # .776

pred_train_sigmoid = model_sigmoid.predict(train_X)
np.mean(pred_train_sigmoid==train_y)  #.776 
