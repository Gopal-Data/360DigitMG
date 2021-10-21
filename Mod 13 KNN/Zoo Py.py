import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

zoo = pd.read_csv("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 13\\zoo.csv")
#checking the Data entries 
data = zoo.describe()
data

#histogram
plt.hist(zoo.type) #Type 1 has occupied the majority of the data.
plt.hist(zoo.legs) #Only 1 5 legged animal found in the dataset Starfish.
plt.hist(zoo.breathes) #20 % of the dataset is water living animal
plt.hist(zoo.eggs) #60 % of the dataset is water living and reptile.
plt.hist(zoo.feathers) #Only 20 % dataset is related to Birds
plt.hist(zoo.milk) #  40% of the dataset is mammal

zoo = zoo.iloc[:, 1:]

#Data Spilting X and Y
X = np.array(zoo.iloc[:,0:16]) # Predictors 
Y = np.array(zoo['type']) # Target 

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)
pred

# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, pred)) #1
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 

# error on train data
pred_train = knn.predict(X_train)
print(accuracy_score(Y_train, pred_train))
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions']) 

# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 15 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(1,15,1):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])

import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(1,15,1),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(1,15,1),[i[1] for i in acc],"bo-")  

#K = 7 is the best K value 