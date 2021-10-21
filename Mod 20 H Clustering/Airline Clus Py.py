import pandas as pd
import matplotlib.pylab as plt
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 
from sklearn.cluster import AgglomerativeClustering

#loadind the dataset
airline = pd.read_excel("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 20\\EastWestAirlines.xlsx", sheet_name="data") 
airline.describe()
 
# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
airline1 = norm_func(airline.iloc[:, 1:])
airline1.describe()

# for creating dendrogram 
z = linkage(airline1, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
h_complete = AgglomerativeClustering(n_clusters = 5, linkage = 'complete', affinity = "euclidean").fit(airline1) 
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
airline['clust'] = cluster_labels # creating a new column and assigning it to new column 
 
# Aggregate mean of each cluster
airline.iloc[:, 1:13].groupby(airline.clust).mean() #mean of all the variables of the cluster 

# creating a csv file 
airline.to_csv("airline.csv", encoding = "utf-8")

import os
os.getcwd()