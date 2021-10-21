import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
diabetes = pd.read_csv("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 15\\Diabetes_RF.csv")
#Renaming the varaibles
diabetes.columns=['pregnant','glucose','pressure','thickness','insulin','index','pedigree','Age','variable']

#dummy createded for n categories
diabetes = pd.get_dummies(diabetes, columns = ["variable"], drop_first = True)

# Input and Output Split
predictors = diabetes.loc[:, diabetes.columns!="variable_YES"]
target = diabetes["variable_YES"]

# Train Test partition of the data
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.3, random_state=0)

xgb_clf = xgb.XGBClassifier(max_depths =2, n_estimators = 15000, learning_rate = .2, n_jobs = -5)
xgb_clf.fit(x_train, y_train)

# Evaluation on Testing Data
confusion_matrix(y_test, xgb_clf.predict(x_test))
accuracy_score(y_test, xgb_clf.predict(x_test)) #.757

xgb.plot_importance(xgb_clf)
xgb_clf = xgb.XGBClassifier(n_estimators = 500, learning_rate = 0.1, random_state = 42)

param_test1 = {'max_depth': range(3,15,1), 'gamma': [0.1,0.2,0.3],
               'subsample': [0.8,0.9], 'colsample_bytree': [0.8, 0.9],
               'rag_alpha': [1e-2, 0.1, 1]}
# Grid Search
grid_search = GridSearchCV(xgb_clf, param_test1, n_jobs = -1, cv = 5, scoring = 'accuracy')
grid_search.fit(x_train, y_train)

cv_xg_clf = grid_search.best_estimator_
# Evaluation on Testing Data with model with hyperparameter
accuracy_score(y_test, cv_xg_clf.predict(x_test))  #0.783
grid_search.best_params_