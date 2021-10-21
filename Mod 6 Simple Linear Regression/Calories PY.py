# Importing libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values
import matplotlib.pyplot as plt # mostly used for visualization purposes 

#loading Dataset
Cal = pd.read_csv("C:\\Users\\gopal\\Documents\\Answer\\mod 6\\calories_consumed.csv")
Cal.describe()
Cal.columns = ['WG', 'CC'] #changing the feature name 
Cal.head() #To know the features names and Top 5 entries 

#Graphical Representation
plt.hist(Cal.WG) #histogram
plt.boxplot(Cal.WG) #boxplot

plt.hist(Cal.CC) #histogram
plt.boxplot(Cal.CC) #boxplot

# Scatter plot
plt.scatter(x = Cal['CC'], y = Cal['WG'], color = 'green') 

# correlation
np.corrcoef(Cal.CC, Cal.WG) 

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('WG ~ CC', data = Cal).fit()
model.summary()
# R SQ value is 0.897

#Confident Interval
model.conf_int(0.05)

pred1 = model.predict(pd.DataFrame(Cal['CC']))

# Regression Line
plt.scatter(Cal.CC, Cal.WG)
plt.plot(Cal.WG, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = Cal.WG - pred1
res_sqr1 = res1*res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

#RMES = 103.30 

#Final Model
from sklearn.model_selection import train_test_split

train, test = train_test_split(Cal, test_size = 0.2)

finalmodel = smf.ols('WG ~ CC', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_WG = np.exp(test_pred)
pred_test_WG

# Model Evaluation on Test data
test_res = test.WG - pred_test_WG
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_WG = np.exp(train_pred)
pred_train_WG

# Model Evaluation on train data
train_res = train.WG - pred_train_WG
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse


#Simple Linear is the best model with R Sq value of 0.897. R Sq above .85 is considered as best model