import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

diabetes= pd.read_csv("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 14\\Diabetes.csv")
diabetes.info()
 
# Dummy variables
diabetes.head()
list( diabetes)

#rename columns in Dataframe
diabetes.columns=['pregnant','glucose','pressure','thickness','insulin','index','pedigree','Age','variable']
list(diabetes)

#creating Dummy Varibales
diabetes = pd.get_dummies(diabetes, columns = ["variable"], drop_first = True)
#Data Spliting
X = np.array(diabetes.iloc[:,0:8]) # Predictors 
Y = np.array(diabetes['variable_YES']) # Target 
 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=4, n_estimators=55, criterion="gini" )
rf.fit(X_train, Y_train)  
pred = rf.predict(X_test)

from sklearn.metrics import accuracy_score
pd.crosstab(Y_test, pred, rownames=['Actual'], colnames= ['Predictions'])
print(accuracy_score(Y_test, pred))
 
# test accuracy
test_acc2 = np.mean(rf.predict(X_test)==Y_test)
test_acc2
#0.707

# train accuracy 
train_acc2 = np.mean(rf.predict(X_train)==Y_train)
train_acc2
#0.865