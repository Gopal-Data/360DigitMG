import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
startup = pd.read_csv("C:/Users/gopal/Documents/Answer/mod 7/50_Startups.csv")

#Renaming the Features 
startup.rename(columns = {'R&D Spend':'RD','Marketing Spend':'Marketing'}, inplace = True) 

#Box plot
plt.boxplot(startup.Profit)
plt.boxplot(startup.RD)
plt.boxplot(startup.Administration)
plt.boxplot(startup.Marketing)
#Histogram
plt.hist(startup.Profit)
plt.hist(startup.RD)
plt.hist(startup.Marketing)
plt.hist(startup.Administration)

#creating Dummy variables and deopping the states column
Startup= pd.get_dummies(startup['State'])
startup= pd.concat([startup,Startup],axis=1)
startup= startup.drop(["State"],axis=1)

# Scatter plot
plt.scatter(x = startup['Profit'], y = startup['RD'], color = 'green') 
plt.scatter(x = startup['Profit'], y = startup['Marketing'], color = 'green') 
plt.scatter(x = startup['Profit'], y = startup['Administration'], color = 'green') 
#Jointplot 
import seaborn as sn
sn.pairplot(startup.iloc[:, :])
cor_values= startup.corr()

# Correlation matrix 
startup.corr()

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
# Model with all variables     
model1 = smf.ols('Profit ~ RD + Marketing + Administration', data = startup).fit()
model1.summary()

# Model with only marketing 
model2 = smf.ols("Profit~Marketing", data= startup).fit()
model2.summary()
# Model with only Administration 
model3= smf.ols("Profit~Administration", data= startup).fit()
model3.summary()
# Model with only Marketing and Administration  
model4= smf.ols("Profit~Administration+Marketing", data= startup).fit()
model4.summary()
#Both variables are significant 

# Checking whether data has any influential values 
# influence index plots
import statsmodels.api as sm
sm.graphics.influence_plot(model1)
#47,49,50 influence Entry

startups = startup.drop(startup.index[[46,48,49]])

# Preparing model                  
model5 = smf.ols('Profit ~ RD + Marketing + Administration', data = startups).fit()    
model5.summary()

model6 = smf.ols('Profit ~ RD + Marketing', data = startup).fit() 
model6.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables

rsq_Profit = smf.ols('Profit ~ RD + Marketing + Administration', data = startup).fit().rsquared  
vif_Profit = 1/(1 - rsq_Profit) 

rsq_RD = smf.ols('RD ~ Profit + Marketing + Administration', data = startup).fit().rsquared  
vif_RD = 1/(1 - rsq_RD)
 
rsq_Marketing = smf.ols('Marketing ~ RD + Profit + Administration', data = startup).fit().rsquared  
vif_Marketing = 1/(1 - rsq_Marketing ) 

rsq_Administration = smf.ols('Administration ~ Marketing + RD + Profit', data = startup).fit().rsquared  
vif_Administration = 1/(1 - rsq_Administration) 


# Storing vif values in a data frame
d1 = {'Variables':['Profit', 'RD', 'Marketing', 'Administration'], 'VIF':[vif_Profit, vif_RD, vif_Marketing, vif_Administration]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

#Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
startup_train, startup_test = train_test_split(startups, test_size = 0.2) # 20% test data

#preparing the model on train data 
model_train = smf.ols("Profit ~ RD + Marketing + Administration", data = startup_train).fit()

#pediction on test data set 
test_pred = model_train.predict(startup_test)

#test residual values 
test_resid  = test_pred - startup_test.Profit
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

# train_data prediction
train_pred = model_train.predict(startup_train)
# train residual values 
train_resid  = train_pred - startup_train.Profit

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse