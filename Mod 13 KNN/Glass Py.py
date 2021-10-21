import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Loading the dataset
glass = pd.read_csv("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 13\\glass.csv")

 
#checking the Data entries 
data = glass.describe()
data

#Histogram
plt.hist(glass.RI) #All the data lies within 1.51 to 1.53
plt.hist(glass.Ca) #Most of the data lies in 8 to 9
plt.hist(glass.Al) #Majority of the data in 1.0 to 2.0
plt.hist(glass.Na) #Majority of the data in 12 to 14
plt.hist(glass.Si) #Majority of the data in 72.5 to 73.5
plt.hist(glass.K) #Majority of the data lies in 0 and 1
plt.hist(glass.Ba) #Majority of the data lies in 0
plt.hist(glass.Fe) # Majority of the data lies in 0
 
#Difference in the scale of the values, we have to normalise the data.
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
glass1 = norm_func(glass.iloc[:,0:9]) 

#Data Spilting X and Y
X = np.array(glass.iloc[:,0:9]) # Predictors 
Y = np.array(glass['Type']) # Target 

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
from sklearn.neighbors import KNeighborsClassifier

#KNN Value is 1
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, Y_train)
pred = knn.predict(X_test)
pred

# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, pred)) #.581
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 

# error on train data
pred_train = knn.predict(X_train)
print(accuracy_score(Y_train, pred_train))
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions']) 

# creating empty list variable 
acc = []

#KNN algorithm for 1 to 15 nearest neighbours and storing the accuracy values

for i in range(10,30,3):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])

import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(10,30,3),[i[0] for i in acc],"ro-")
# test accuracy plot
plt.plot(np.arange(10,30,3),[i[1] for i in acc],"bo-")

#K = 16 is the best K value