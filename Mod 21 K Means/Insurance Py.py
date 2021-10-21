import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import	KMeans

# Kmeans on University Data set 
insurance = pd.read_csv("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 21\\Insurance Dataset.csv")

# Normalization function 
def norm_func(i):
    x = (i - i.min())/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(insurance.iloc[:, 0:]) #applying min max to the new data

###### scree plot or elbow curve ############
TWSS = []   #Assigning TWSS as NULL object
k = list(range(2,11)) # checking the curve for n number of cluster

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS #print the values of TWSS

# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS") 

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 5)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
insurance['clust'] = mb # creating a  new column and assigning it to new column 

insurance.head()
df_norm.head()

insurance = insurance.iloc[:,[5,0,1,2,3,4]] #rearranging the variables

insurance.iloc[:, 1:].groupby(insurance.clust).mean()  #taking the mean of the cluster

insurance.to_csv("Kmeans_university.csv", encoding = "utf-8")  #saving the data to excel

import os
os.getcwd()  