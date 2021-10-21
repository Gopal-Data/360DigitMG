import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

diabetes = pd.read_csv("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 15\\Diabetes_RF.csv")
# checking for NaN Value
diabetes.head()
diabetes.info()
#checking for NaN Value
diabetes.isnull()
 #Renaming the variables
diabetes.columns=['pregnant','glucose','pressure','thickness','insulin','index','pedigree','Age','variable']
 
#typecasting the categorical to numerical 
diabetes = pd.get_dummies(diabetes, columns = ["variable"], drop_first = True)
diabetes.head()
 
# Input and Output Split
predictors = diabetes.loc[:, diabetes.columns!="variable_YES"]
target = diabetes["variable_YES"]

# Train Test partition of the data
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.25, random_state=0)
ada_clf = AdaBoostClassifier(learning_rate = .1, n_estimators = 700)
ada_clf.fit(x_train, y_train) 

# increasing the learning rate and estimators is overfitting the model
# lower learning rate 0.01 keeps the model source under .80
# Higher learning rate .7 to 1 overfitting the model

# Evaluation on Testing Data
confusion_matrix(y_test, ada_clf.predict(x_test))
accuracy_score(y_test, ada_clf.predict(x_test)) #0.8020
# Evaluation on Training Data
accuracy_score(y_train, ada_clf.predict(x_train)) #0.8350