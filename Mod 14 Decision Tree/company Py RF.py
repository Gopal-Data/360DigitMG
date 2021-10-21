import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

company = pd.read_csv("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 14\\company.csv")

#typcasting sales variables
company.loc[company.Sales <= 10, "Sales"]= "1"
company.loc[company.Sales != '1', "Sales"]= "0"
 
# n-1 dummy variables will be created for n categories
lb= LabelEncoder()
company["ShelveLoc"] = lb.fit_transform(company["ShelveLoc"]) 
company["Urban"] = lb.fit_transform(company["Urban"])
company["US"] = lb.fit_transform(company["US"])

#spilting the data
colname = list(company.columns)
X = np.array(company.iloc[:,1:11]) # Predictors 
Y = np.array(company['Sales']) # Target 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3, n_estimators=75, criterion="entropy",max_depth=3)
rf.fit(X_train, Y_train) 
pred = rf.predict(X_test)

from sklearn.metrics import accuracy_score
pd.crosstab(Y_test, pred, rownames=['Actual'], colnames= ['Predictions'])
print(accuracy_score(Y_test, pred))

# test accuracy
test_acc2 = np.mean(rf.predict(X_test)==Y_test)
test_acc2  #0.825 #0.791

# train accuracy 
train_acc3 = np.mean(rf.predict(X_train)==Y_train)
train_acc3 # 1.0 #0.864
