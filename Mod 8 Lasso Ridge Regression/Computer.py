import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
comp = pd.read_csv("C:/Users/gopal/Documents/360DigiTMG/mod 8/Computer_Data.csv")
 
#creating Dummies Variable for CD,Premium, Multi
comp = pd.get_dummies(comp, columns=['cd'])
comp = pd.get_dummies(comp, columns=['premium'])
comp = pd.get_dummies(comp, columns=['multi'])
comp = comp.drop(['cd_no'], axis = 1) 
#removing the unwanted column
comp = comp.drop(['multi_no'], axis = 1) 
comp = comp.drop(['premium_no'], axis = 1) 
comp = comp.drop(['Unnamed: 0'], axis = 1)
#Renaming the columns
comp.rename(columns={'cd_yes':'CD','multi_yes':'multi','premium_yes':'premium'}, inplace=True)

corr= comp.corr()
corr

#EDA
datainsight = comp.describe()

#sctterplot and histogram between variables
sns.pairplot(comp)  

comp.head() 

# preparing the model on train data 
model_train = smf.ols("price ~ speed+ hd+ ram+ screen+ ads+ trend+ CD +premium +multi", data = comp).fit()
model_train.summary()

# prediction
pred = model_train.predict(comp)

# Error
resid  = pred - comp.price

# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse
#R Sq 0.776  RMSE 275.129

#LASSO MODEL
from sklearn.linear_model import Lasso
lasso = Lasso(alpha = .1,  normalize = True)
lasso.fit(comp.iloc[:, 1:], comp.price)

# coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(comp.columns[1:]))
 
pred_lasso = lasso.predict(comp.iloc[:, 1:])

# Adjusted r-square#
lasso.score(comp.iloc[:, 1:], comp.price)
 
#RMSE
np.sqrt(np.mean((pred_lasso - comp.price)**2))
#Alpha = 004 , R Sq 0.7755 RMSE 275.132
#Alpha = .1 , R Sq 0.7732 RMSE 276.569

#RIDGE REGRESSION 
from sklearn.linear_model import Ridge
rm = Ridge(alpha = .1, normalize = True)

rm.fit(comp.iloc[:, 1:], comp.price)

#coefficients values for all the independent vairbales#
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(comp.columns[1:]))

pred_rm = rm.predict(comp.iloc[:, 1:])

# adjusted r-square#
rm.score(comp.iloc[:, 1:], comp.price)
 
#RMSE
np.sqrt(np.mean((pred_rm - comp.price)**2))
#Alpha = 004 ,R Sq 0.7755 RMSE 275.148
#Alpha = .1 , R Sq 0.7625 RMSE 276.569