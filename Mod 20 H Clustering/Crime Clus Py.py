import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch 
from scipy.cluster.hierarchy import linkage

#loading the Dataset
crime = pd.read_csv("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 20\\crime_data.csv") 
 
# Normalization function 
def norm_func(i):
    x = (i-i.min())/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
crime1 = norm_func(crime.iloc[:, 1:])
 
# for creating dendrogram 
z = linkage(crime1, method = "complete", metric = "cityblock")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 3 as clusters from the above dendrogram
h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "cityblock").fit(crime1) 
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
crime['cluster'] = cluster_labels # creating a new column and assigning it to new column 
 
# Aggregate mean of each cluster 
crime.iloc[:, 1:13].groupby(crime.cluster).mean()
    
# creating a csv file 
crime.to_csv("crime.csv", encoding = "utf-8")

import os
os.getcwd() 