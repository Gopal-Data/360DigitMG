import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from sklearn.cluster import	KMeans
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 
from sklearn.cluster import AgglomerativeClustering

wine = pd.read_csv("C:\\Users\\gopal\\Documents\\360DigiTMG\\mod 22\\wine.csv")
wine.describe()
# Considering only numerical data and removing the type data 
wine_data = wine.iloc[:, 1:]
# Normalizing the numerical data 
winnorm = scale(wine_data)
winnorm
pca = PCA(n_components = 6)
pca_values = pca.fit_transform(winnorm)
# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var
pca.components_
pca.components_[0]
# Cumulative variance 
var1 = np.cumsum(np.round(var, decimals = 4) * 100)
var1
# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")
# PCA scores
pca_values
pca_data = pd.DataFrame(pca_values)
pca_data.columns = "comp0", "comp1", "comp2", "comp3", "comp4", "comp5"
finalPCA = pd.concat([wine.Type, pca_data.iloc[:, 0:3]], axis = 1)
# Scatter diagram
import matplotlib.pylab as plt
plt.scatter(x = finalPCA.comp0, y = finalPCA.comp1)

#K Mean For PCA Data
def norm_func(i):
    x = (i - i.min())/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
Winenorm = norm_func(finalPCA.iloc[:, 1:]) #applying min max to the new data

###### scree plot or elbow curve ############
TWSS = []   #Assigning TWSS as NULL object
k = list(range(2,8)) # checking the curve for n number of cluster
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(Winenorm)
    TWSS.append(kmeans.inertia_)    
TWSS #print the values of TWSS
# Scree plot 
plt.plot(k, TWSS, 'r*-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS") 
# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters =3)
model.fit(Winenorm)
model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
Winenorm['clust'] = mb # creating a  new column and assigning it to new column 
WineKmeanPCA = Winenorm.iloc[:,[3,0,1,2]] #rearranging the variables
WineKmeanPCA.iloc[:, 1:].groupby(WineKmeanPCA.clust).mean() #taking the mean of the cluster

#Hierarchical Clustering
#Creating dendrogram 
z = linkage(Winenorm, method = "complete", metric = "euclidean") 

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 4 as clusters from the above dendrogram
h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(Winenorm) 
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
Winenorm['clust'] = cluster_labels # creating a new column and assigning it to new column 
# Aggregate mean of each cluster
Winenorm.iloc[:, 0:].groupby(Winenorm.clust).mean() #mean of all the variables of the cluster  

#Original Data
#K Mean 
def norm_func(i):
    x = (i - i.min())/ (i.max() - i.min())
    return (x)
# Normalized data frame (considering the numerical part of data)
Winenorm1 = norm_func(wine.iloc[:, 1:]) #applying min max to the new data

#scree plot or elbow curve
TWSS = []   #Assigning TWSS as NULL object
k = list(range(2,8)) # checking the curve for n number of cluster
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(Winenorm1)
    TWSS.append(kmeans.inertia_)    
TWSS #print the values of TWSS
# Scree plot 
plt.plot(k, TWSS, 'r*-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS") 
# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters =3)
model.fit(Winenorm1)
model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
Winenorm1['clust'] = mb # creating a  new column and assigning it to new column 
WineKmeanOD = Winenorm1.iloc[:,[13,0,1,2,3,4,5,6,7,8,9,10,11,12]] #rearranging the variables
wine.iloc[:, 1:].groupby(WineKmeanOD.clust).mean() #taking the mean of the cluster


#Hierarchical Clustering
#Creating dendrogram 
z = linkage(Winenorm1, method = "complete", metric = "euclidean") 
# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()
# Now applying AgglomerativeClustering choosing 4 as clusters from the above dendrogram
h_complete = AgglomerativeClustering(n_clusters = 5, linkage = 'complete', affinity = "euclidean").fit(Winenorm1) 
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
Winenorm1['clust'] = cluster_labels # creating a new column and assigning it to new column 
# Aggregate mean of each cluster
Winenorm1.iloc[:, 0:].groupby(Winenorm1.clust).mean() #mean of all the variables of the cluster