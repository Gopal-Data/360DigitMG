import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
toyota = pd.read_csv("C:/Users/gopal/Documents/360DigiTMG/mod 8/ToyotaCorolla.csv", encoding= 'unicode_escape')
toyota= toyota.iloc[:,[2,3,6,8,12,13,15,16,17]]
toyota.head() 

#Renaming the columns
toyota.rename(columns={'Age_08_04':'age','Quarterly_Tax':'Tax','Price':'price'}, inplace=True)
 
corr= toyota.corr()
corr

#EDA
datainsight = toyota.describe()

#sctterplot and histogram between variables
sns.pairplot(toyota)  

toyota.head() 

#Preparing the model on train data 
model_train = smf.ols("price ~ age+ KM+ HP+ cc+ Doors+ Gears+ Tax +Weight", data = toyota).fit()
model_train.summary()

#Prediction
pred = model_train.predict(toyota)

#Error
resid  = pred - toyota.price

#RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse

#Rsq  0.864  RMSE 338.2584

#LASSO MODEL
from sklearn.linear_model import Lasso
lasso = Lasso(alpha = .05 ,  normalize = True)
lasso.fit(toyota.iloc[:, 1:], toyota.price)

#coefficient values for all independent variables
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(toyota.columns[1:]))
 
pred_lasso = lasso.predict(toyota.iloc[:, 1:])

# Adjusted r-square#
lasso.score(toyota.iloc[:, 1:], toyota.price)
 
#RMSE
np.sqrt(np.mean((pred_lasso - toyota.price)**2))
#Alpha = 5  Rsq 0.855  RMSE  1378.9197
#Alpha = .05  Rsq 0.863 RMSE  1338.2679

#RIDGE REGRESSION 
from sklearn.linear_model import Ridge
rm = Ridge(alpha = 5, normalize = True)

rm.fit(toyota.iloc[:, 1:], toyota.price)

#coefficients values for all the independent vairbales#
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(toyota.columns[1:]))

pred_rm = rm.predict(toyota.iloc[:, 1:])

# adjusted r-square#
rm.score(toyota.iloc[:, 1:], toyota.price)
 
#RMSE
np.sqrt(np.mean((pred_rm - toyota.price)**2))

#Alpha = 5 Rsq 0.4032  RMSE 2800.8239
#Alpha = .05 Rsq 0.8628  RMSE 1342.9302
 