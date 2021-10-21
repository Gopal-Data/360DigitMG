# Importing libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values
import matplotlib.pyplot as plt # mostly used for visualization purposes 

#loading Dataset
emp = pd.read_csv("C:\\Users\\gopal\\Documents\\Answer\\mod 6\\emp_data.csv")
emp.describe()
emp.columns = ['SH', 'CR'] #changing the feature name 
emp.head() #To know the features names and Top 5 entries 

#Graphical Representation
plt.hist(emp.SH) #histogram
plt.boxplot(emp.SH) #boxplot

plt.hist(emp.CR) #histogram
plt.boxplot(emp.CR) #boxplot

# Scatter plot
plt.scatter(x = emp['SH'], y = emp['CR'], color = 'green') 

# correlation
np.corrcoef(emp.SH, emp.CR) 

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('CR ~ SH', data = emp).fit()
model.summary()

#Confident Interval
model.conf_int(0.05)

pred1 = model.predict(pd.DataFrame(emp['SH']))

# Regression Line
plt.scatter(emp.SH, emp.CR)
plt.plot(emp.SH, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = emp.CR - pred1
res_sqr1 = res1*res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(CR) ~ SH + I(SH*SH)', data = emp).fit()
model4.summary()

model4.conf_int(0.05) # 95% confidence interval

pred4 = model4.predict(pd.DataFrame(emp))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X = emp.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = emp.iloc[:, 1].values


plt.scatter(emp.SH, np.log(emp.CR))
plt.plot(X, pred4, color='red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = emp.CR - pred4_at
res_sqr4 = res4*res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

#Linear Model R Sq Value is 0.8312 and RMSE is 3.99 
#Polynomial model R Sq Value is 0.9836 and RMSE is 1.32

#Polynomial model with 2 Degree is the best model as its R Sq value is above .85