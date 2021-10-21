import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

toyota = pd.read_csv("C:/Users/gopal/Documents/Answer/mod 7/ToyotaCorolla.csv",encoding= 'unicode_escape')

toyota1= toyota.iloc[:,[2,3,6,8,12,13,15,16,17]]
toyota1.rename(columns={"Age_08_04":"Age"},inplace=True)
toyota1.rename(columns={"Quarterly_Tax":"tax"},inplace=True)
toyota1.describe()

#boxplot
plt.boxplot(toyota1["Price"])
plt.boxplot(toyota1["Age"])
plt.boxplot(toyota1["HP"])
plt.boxplot(toyota1["cc"])
plt.boxplot(toyota1["tax"])
plt.boxplot(toyota1["Weight"])

#QQ plot
import statsmodels.api as sm
sm.graphics.qqplot(toyota1["Price"])   
sm.graphics.qqplot(toyota1["Age"]) 
sm.graphics.qqplot(toyota1["HP"])
sm.graphics.qqplot(toyota1["tax"])  
sm.graphics.qqplot(toyota1["Weight"])  
sm.graphics.qqplot(toyota1["Gears"])  
sm.graphics.qqplot(toyota1["Doors"])  
sm.graphics.qqplot(toyota1["cc"])  
 
#histogram
plt.hist(toyota1["Price"])  
plt.hist(toyota1["Age"])  
plt.hist(toyota1["HP"]) 
plt.hist(toyota1["tax"])  
plt.hist(toyota1["Weight"]) 
 
import seaborn as sn
sn.pairplot(toyota1)
correlation_values= toyota1.corr()
toyota1.corr()
toyota1.head(2)

##model1
import statsmodels.formula.api as smf

model1= smf.ols("Price~ Age +KM +HP +cc +Doors +Gears +tax +Weight",data= toyota1).fit()
model1.summary() 
#R Sq Value is 0.864
#CC and Doors are insignificant, P value is .05

#Model based on CC 
model2 = smf.ols("Price~ cc",data= toyota1).fit()
model2.summary()
#R Sq value is .016
#CC is significant

#Model based on Doors 
model3 = smf.ols("Price~ Doors", data= toyota1).fit()
model3.summary()
#R Sq value is .034
#Doors is significant

model4 = smf.ols("Price~ cc+ Doors",data= toyota1).fit()
model4.summary()
#R Sq value is  0.047
#Both the data are signifiant

##plotting the influence plot
import statsmodels.api as sm
sm.graphics.influence_plot(model1)

#Removing influence entry from the dat.  
toyota2= toyota1.drop(toyota.index[[80]],axis=0)
model5= smf.ols("Price~ Age+ KM +HP +cc +Doors +Gears +tax +Weight",data= toyota2).fit()
model5.summary()
#RSq 0.869
#Doors is insignificant

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
 
rsq_price = smf.ols('Price ~ Age+ KM +cc +Doors +Gears +tax +Weight+ HP', data = toyota1).fit().rsquared  
vif_price = 1/(1 - rsq_price) 

rsq_age = smf.ols('Age ~ Price+ KM +cc +Doors +Gears +tax +Weight + HP', data = toyota1).fit().rsquared  
vif_age = 1/(1 - rsq_age) 

rsq_KM = smf.ols('KM ~ Age+ Price +cc +Doors +Gears +tax +Weight + HP', data = toyota1).fit().rsquared  
vif_KM = 1/(1 - rsq_KM) 

rsq_cc = smf.ols('cc ~ Age+ KM +Price +Doors +Gears +tax +Weight+ HP ', data = toyota1).fit().rsquared  
vif_cc = 1/(1 - rsq_cc) 

rsq_Doors = smf.ols('Doors ~ Age+ KM +cc +Price +Gears +tax +Weight+ HP ', data = toyota1).fit().rsquared  
vif_Doors = 1/(1 - rsq_Doors) 

rsq_gears = smf.ols('Gears ~ Age+ KM +cc +Doors +Price +tax +Weight + HP', data = toyota1).fit().rsquared  
vif_gears = 1/(1 - rsq_gears) 

rsq_tax = smf.ols('tax ~ Age+ KM +cc +Doors +Gears +Price +Weight + HP', data = toyota1).fit().rsquared  
vif_tax = 1/(1 - rsq_tax) 

rsq_weight = smf.ols('Weight ~ Age+ KM +cc +Doors +Gears +Price +tax+ HP ', data = toyota1).fit().rsquared  
vif_weight = 1/(1 - rsq_weight) 

d1 = {'Variables':['Price', 'Age', 'KM', 'cc','Doors','gears','tax', 'weight'], 'VIF':[vif_price, vif_age, vif_KM, vif_cc, vif_Doors, vif_gears, vif_tax, vif_weight]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame     

# All the VIF is less than 10
 
#Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
toyota1_train, toyota1_test = train_test_split(toyota1, test_size = 0.2) # 20% test data

#preparing the model on train data 
model_train = smf.ols("Price~ Age+ KM +HP +cc +Doors +Gears +tax +Weight", data= toyota1_train).fit()
model_train.summary()

#pediction on test data set 
test_pred = model_train.predict(toyota1_test)

#test residual values 
test_resid  = test_pred - toyota1_test.Price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

# train_data prediction
train_pred = model_train.predict(toyota1_train)
# train residual values 
train_resid  = train_pred - toyota1_train.Price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
