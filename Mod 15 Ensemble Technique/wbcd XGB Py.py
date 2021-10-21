import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

wbcd = pd.read_csv("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 15\\wbcd.csv")

wbcd.drop(wbcd.columns[[0]], axis=1, inplace=True)
# n-1 dummy variables will be created for n categories
wbcd = pd.get_dummies(wbcd, columns = ["diagnosis"], drop_first = True)

# Input and Output Split
predictors = wbcd.loc[:, wbcd.columns!="diagnosis_M"]
target = wbcd["diagnosis_M"]

# Train Test partition of the data
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)

import xgboost as xgb

xgb_clf = xgb.XGBClassifier(max_depths = 5, n_estimators = 10000, learning_rate = 0.3, n_jobs = -1)
xgb_clf.fit(x_train, y_train)
# Evaluation on Testing Data
confusion_matrix(y_test, xgb_clf.predict(x_test))
accuracy_score(y_test, xgb_clf.predict(x_test)) #0.973
xgb.plot_importance(xgb_clf)

xgb_clf = xgb.XGBClassifier(n_estimators = 500, learning_rate = 0.1, random_state = 42)
param_test1 = {'max_depth': range(3,15,1), 'gamma': [0.1, 0.2, 0.3],
               'subsample': [0.7,0.8, 0.9], 'colsample_bytree': [0.8, 0,9],
               'rag_alpha': [1e-2, 0.1, 1]}

# Grid Search
grid_search = GridSearchCV(xgb_clf, param_test1, n_jobs = -1, cv = 5, scoring = 'accuracy')
grid_search.fit(x_train, y_train)
cv_xg_clf = grid_search.best_estimator_

# Evaluation on Testing Data with model with hyperparameter
accuracy_score(y_test, cv_xg_clf.predict(x_test)) #0.974
grid_search.best_params_