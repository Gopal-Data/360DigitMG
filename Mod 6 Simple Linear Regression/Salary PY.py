# Importing libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values
import matplotlib.pyplot as plt # mostly used for visualization purposes 

#loading Dataset
sala = pd.read_csv("C:\\Users\\gopal\\Documents\\Answer\\mod 6\\Salary_Data.csv")
sala.describe()
sala.columns = ['ye', 'sa'] #changing the feature name 
sala.head() #To know the features names and Top 5 entries 

#Graphical Representation
plt.hist(sala.ye) #histogram
plt.boxplot(sala.ye) #boxplot

plt.hist(sala.sa) #histogram
plt.boxplot(sala.sa) #boxplot

# Scatter plot
plt.scatter(y = sala['ye'], x = sala['sa'], color = 'green') 

# correlation
np.corrcoef(sala.ye, sala.sa) 

# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('sa ~ ye', data = sala).fit()
model.summary()

#Confident Interval
model.conf_int(0.05)

pred1 = model.predict(pd.DataFrame(sala['ye']))

# Regression Line
plt.scatter(sala.ye, sala.sa)
plt.plot(sala.ye, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
    res1 = sala.sa - pred1
    res_sqr1 = res1*res1
    mse1 = np.mean(res_sqr1)
    rmse1 = np.sqrt(mse1)
    rmse1
    
#Linear Model R Sq Value is 0.957 and RMSE is 5592.044 is the best model  