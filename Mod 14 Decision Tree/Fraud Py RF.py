import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
 
fraud = pd.read_csv("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 14\\Fraud_check.csv")
fraud.info()
 
# Dummy variables
fraud.head()
list(fraud)
#rename columns in Dataframe
fraud = fraud.rename(columns={'Taxable.Income':'Taxable','City.Population':'Population' , 'Work.Experience':'Work', 'Marital.Status' : 'Marital'})

#typcasting sales variables
lb= LabelEncoder()
fraud["Undergrad"] = lb.fit_transform(fraud["Undergrad"]) 
fraud["Marital"] = lb.fit_transform(fraud["Marital"]) 
fraud["Urban"] = lb.fit_transform(fraud["Urban"])
str(fraud.Urban)

fraud.loc[fraud.Taxable <= 30000, "Taxable"]= "1"
fraud.loc[fraud.Taxable !='1', "Taxable"]= "0"
fraud=fraud.iloc[:,[0,1,3,4,5,2]]   

X = np.array(fraud.iloc[:,0:6]) # Predictors 
Y = np.array(fraud['Taxable']) # Target 
 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3, n_estimators=125, criterion="gini")
rf.fit(X_train, Y_train) # Fitting RandomForestClassifier model from sklearn.ensemble  
pred = rf.predict(X_test)

from sklearn.metrics import accuracy_score
pd.crosstab(Y_test, pred, rownames=['Actual'], colnames= ['Predictions'])
print(accuracy_score(Y_test, pred))

# test accuracy
test_acc2 = np.mean(rf.predict(X_test)==Y_test)
test_acc2  

# train accuracy 
train_acc3 = np.mean(rf.predict(X_train)==Y_train)
train_acc3  
