# Importing libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values
import matplotlib.pyplot as plt # mostly used for visualization purposes 

#loading Dataset
de = pd.read_csv("C:\\Users\\gopal\\Documents\\Answer\\mod 6\\delivery_time.csv")
de.describe()
de.head()
de.columns = ['DT', 'ST'] #changing the feature name 
de.head() #To know the features names and Top 5 entries 

#Graphical Representation
plt.hist(de.DT) #histogram
plt.boxplot(de.DT) #boxplot

plt.hist(de.ST) #histogram
plt.boxplot(de.ST) #boxplot

# Scatter plot
plt.scatter(x = de['ST'], y = de['DT'], color = 'green') 

# correlation
np.corrcoef(de.ST, de.DT) 
# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('DT ~ ST', data = de).fit()
model.summary()

#Confident Interval
model.conf_int(0.05)

pred1 = model.predict(pd.DataFrame(de['ST']))

# Regression Line
plt.scatter(de.ST, de.DT)
plt.plot(de.ST, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = de.DT - pred1
res_sqr1 = res1*res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(DT) ~ ST + I(ST*ST)', data = de).fit()
model4.summary()

model4.conf_int(0.05) # 95% confidence interval

pred4 = model4.predict(pd.DataFrame(de))
pred4_dt = np.exp(pred4)
pred4_dt

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X = de.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = wcat.iloc[:, 1].values


plt.scatter(de.ST, np.log(de.DT))
plt.plot(X, pred4, color='red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = de.DT - pred4_dt
res_sqr4 = res4*res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

#Linear Model R Sq is 0.6823 and RMSE is 2.79 
#Polynomial model R Sq Value is 0.7649 and RMSE is 2.799042


#Polynominal Model with 2 degree is the better model while comparing 