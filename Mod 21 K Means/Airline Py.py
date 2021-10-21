import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import	KMeans
airline = pd.read_excel("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 21\\EastWestAirlines.xlsx", sheet_name="data")
ari = airline.drop(["ID#"], axis = 1) #removing unwanted variable varible and creating a new data for futhure analyics 

# Normalization function 
def norm_func(i):
    x = (i - i.min())/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(ari.iloc[:, 0:]) #applying min max to the new data

###### scree plot or elbow curve ############
TWSS = []   #Assigning TWSS as NULL object
k = list(range(2,8)) # checking the curve for n number of cluster
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS #print the values of TWSS

# Scree plot 
plt.plot(k, TWSS, 'r*-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS") 

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
ari['clust'] = mb # creating a  new column and assigning it to new column 

ari.head()
df_norm.head()

ari = ari.iloc[:,[11,0,1,2,3,4,5,6,7,8,9,10]] #rearranging the variables

ari.iloc[:, 1:].groupby(ari.clust).mean() #taking the mean of the cluster

ari.to_csv("Kmeans_university.csv", encoding = "utf-8") #saving the data to excel

import os
os.getcwd()  