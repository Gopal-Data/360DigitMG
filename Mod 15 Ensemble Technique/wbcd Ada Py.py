import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

wbcd = pd.read_csv("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 15\\wbcd.csv")
# checking for NaN Value
wbcd.head()
wbcd.info()
#checking for NaN Value
wbcd.isnull()
wbcd.drop(wbcd.columns[[0]], axis=1, inplace=True)
 
#typecasting the categorical to numerical 
wbcd = pd.get_dummies(wbcd, columns = ["diagnosis"], drop_first = True)
wbcd.head()
 
# Input and Output Split
predictors = wbcd.loc[:, wbcd.columns!="diagnosis_M"]
target = wbcd["diagnosis_M"]

# Train Test partition of the data
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.30, random_state=0)
ada_clf = AdaBoostClassifier(learning_rate = .07, n_estimators = 500)
ada_clf.fit(x_train, y_train) 
 
# Evaluation on Testing Data
confusion_matrix(y_test, ada_clf.predict(x_test))
accuracy_score(y_test, ada_clf.predict(x_test)) #0.9707
# Evaluation on Training Data
accuracy_score(y_train, ada_clf.predict(x_train)) #1.0
