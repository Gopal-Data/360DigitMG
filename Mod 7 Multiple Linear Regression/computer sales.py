import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

computer = pd.read_csv("C:/Users/gopal/Documents/Answer/mod 7/Computer_Data.csv")

#removed unwanted sequance number
computer= computer.drop(["Unnamed: 0"],axis=1)

plt.boxplot(computer["Price"])
plt.boxplot(computer["speed"])
plt.boxplot(computer["ram"])
plt.boxplot(computer["hd"])
plt.boxplot(computer["ads"])
plt.boxplot(computer["trend"])

#QQ PLot
import statsmodels.api as sm
sm.graphics.qqplot(computer["Price"])   
sm.graphics.qqplot(computer["speed"]) 
sm.graphics.qqplot(computer["ram"])
sm.graphics.qqplot(computer["hd"])  
sm.graphics.qqplot(computer["ads"])  
sm.graphics.qqplot(computer["trend"])  
 
#histogram
plt.hist(computer["Price"])  
plt.hist(computer["speed"])  
plt.hist(computer["ram"]) 
plt.hist(computer["hd"])  
plt.hist(computer["ads"])  
plt.hist(computer["trend"])  
 
import seaborn as sn
sn.pairplot(computer)
corr_values = computer.corr() 

# Creating dummy varibles on computer data set cd, multi, premium are categorical data. 

comps = pd.get_dummies(computer, ['cd','premium','multi'])
comps.drop(["cd_no", "multi_no","premium_no"], axis = 1,  inplace = True) 
computer = comps
 
#model Building 
import statsmodels.formula.api as smf
model1= smf.ols("price~ speed+ hd+ ram +screen+ads+trend+cd_yes+premium_yes+multi_yes", data= computer).fit()
model1.summary() 
## R Sq = 0.776 

import statsmodels.api as sm
sm.graphics.influence_plot(model1)

comp_new = computer.drop(computer.index[[1441,1701, 3784, 4478]])

model2= smf.ols("price~ speed+ hd+ ram +screen+ads+trend+cd_yes+premium_yes+multi_yes", data= comp_new).fit()
model2.summary()
# R Square 0.776
# even after removing the influence entry. no changes in RSquare 
 
#Applying sqrt in all variable
model3 = smf.ols("price ~ np.sqrt(speed)+np.sqrt(screen)+ np.sqrt(ads)+np.sqrt(trend)+np.sqrt(cd_yes)+np.sqrt(premium_yes)+np.sqrt(multi_yes)", data= computer).fit()
model3.summary()
##R Square 0.408  

model4 = smf.ols("price ~ np.log(speed)+ np.log(screen)+ np.log(ads)+ np.log(trend)+ np.log(cd_yes)+ np.log(premium_yes)+ np.log(multi_yes)", data= computer).fit()
model4.summary()
 #RSq Value is 0.508

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_price = smf.ols('price ~ speed+ hd + ram +screen +ads +trend +cd_yes +premium_yes +multi_yes', data = computer).fit().rsquared  
vif_price = 1/(1 - rsq_price) 

rsq_speed = smf.ols('speed ~ price+ hd + ram +screen +ads +trend +cd_yes +premium_yes +multi_yes', data= computer).fit().rsquared  
vif_speed = 1/(1 - rsq_speed)

rsq_hd = smf.ols('hd ~ price+ speed + ram +screen +ads +trend +cd_yes +premium_yes +multi_yes', data= computer).fit().rsquared  
vif_hd = 1/(1 - rsq_hd)

rsq_ram = smf.ols('ram ~ price+ speed + hd +screen +ads +trend +cd_yes +premium_yes +multi_yes', data= computer).fit().rsquared  
vif_ram = 1/(1 - rsq_ram)

rsq_screen = smf.ols('screen ~ price+ speed + hd +ram +ads +trend +cd_yes +premium_yes +multi_yes', data= computer).fit().rsquared  
vif_screen = 1/(1 - rsq_screen)

rsq_ads = smf.ols('ads ~ price+ speed + hd +ram +screen +trend +cd_yes +premium_yes +multi_yes', data= computer).fit().rsquared  
vif_ads = 1/(1 - rsq_ads) 

rsq_trend = smf.ols('trend ~ price+ speed + hd +ram +screen +ads +cd_yes +premium_yes +multi_yes', data= computer).fit().rsquared  
vif_trend = 1/(1 - rsq_trend) 

d1 = {'Variables':['price', 'speed', 'hd', 'ram','screen','ads','trend'], 'VIF':[vif_price, vif_speed, vif_hd, vif_ram,vif_screen,vif_ads, vif_trend]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

# variables are less than 10
 
#Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
computer_train, computer_test = train_test_split(computer, test_size = 0.2) # 20% test data

#preparing the model on train data 
model_train = smf.ols("price ~ np.sqrt(speed)+np.sqrt(screen)+ np.sqrt(ads)+np.sqrt(trend)+np.sqrt(cd_yes)+np.sqrt(premium_yes)+np.sqrt(multi_yes)", data= computer_train).fit()
model_train.summary()

#pediction on test data set 
test_pred = model_train.predict(computer_test)

#test residual values 
test_resid  = test_pred - computer_test.price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

# train_data prediction
train_pred = model_train.predict(computer_train)
# train residual values 
train_resid  = train_pred - computer_train.price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse