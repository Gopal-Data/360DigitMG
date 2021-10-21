import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
startup = pd.read_csv("C:/Users/gopal/Documents/360DigiTMG/mod 8/50_Startups.csv")

# Rearrange the order of the variables
startup = startup.iloc[:, [4, 0, 1, 2, 3]]
#Renaming the columns
startup.rename(columns={'R&D Spend':'RD','Marketing Spend':'market'}, inplace=True)
#creating Dummies Variable for State
startup = pd.get_dummies(startup, columns=['State'])
startup= startup.drop(['State_New York'], axis = 1) 


# Correlation matrix 
corr= startup.corr()
corr

#EDA
datainsight = startup.describe()

#sctterplot and histogram between variables
sns.pairplot(startup)  

# preparing the model on train data 
model_train = smf.ols("Profit ~ RD + Administration + market + State_California+State_Florida", data = startup).fit()
model_train.summary()

# prediction
pred = model_train.predict(startup)

# Error
resid  = pred - startup.Profit

# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse

#Regession 0.951 RMSE 8854.7610

#LASSO MODEL

from sklearn.linear_model import Lasso

lasso = Lasso(alpha = .05,  normalize = True)

lasso.fit(startup.iloc[:, 1:], startup.Profit)

# coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(startup.columns[1:]))
 
pred_lasso = lasso.predict(startup.iloc[:, 1:])

# Adjusted r-square#
lasso.score(startup.iloc[:, 1:], startup.Profit)
#Alpha .005 = 0.95075 
#Alpha .05 = 0.95075
#RMSE
np.sqrt(np.mean((pred_lasso - startup.Profit)**2))
#Alpha .005 = 8854.7610 
#Alpha .05 =  8854.7610

### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
rm = Ridge(alpha = .05, normalize = True)

rm.fit(startup.iloc[:, 1:], startup.Profit)

#coefficients values for all the independent vairbales#
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(startup.columns[1:]))

pred_rm = rm.predict(startup.iloc[:, 1:])

# adjusted r-square#
rm.score(startup.iloc[:, 1:], startup.Profit)
#Alpha .005 = 0.95070 
#Alpha .05 = 0.9474

#RMSE
np.sqrt(np.mean((pred_rm - startup.Profit)**2))
#Alpha .005 = 8858.7914 
#Alpha .05 = 9146.9749