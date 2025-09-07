#1 Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

#2 Data Collection & Analysis
#loading the data from csv file to Pandas Dataframe
customer_data = pd.read_csv('Mall_Customers.csv')

#first 5 rows of the dataframe
# print(customer_data.head())

#finding the number of columns and rows
print(customer_data.shape)

#get some informations about the dataset
print(customer_data.info())

#checking missing values
print(customer_data.isnull().sum())

#3 Choosing the Annual Income & Spending Score Columns
X = customer_data.iloc[:,[3,4]].values

print(X)

#4 Choosing the Number of Clusters
# WCSS -> Within Cluster Sums of Squares

# finding WCSS value for different number of clusters
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init = 'k-means++', random_state =42)
    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

#plot an elbow graph
#choose the elbow point where after it there is no significant drop in the value
sns.set()
plt.plot(range(1,11),wcss)
plt.title("The Elbow Point Graph")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS Value")
#plt.show()

#5 Optimum Number of clusters = 5

#6 Training the k-means Clustering Model
kmeans = KMeans(n_clusters = 5, init ='k-means++', random_state = 0)

#return a label for each datapoint base on their cluster
y = kmeans.fit_predict(X)

#7 Visulizing all the clusters
# plotting all the clusters and their centroids 

plt.figure(figsize=(8,8))
plt.scatter(X[y==0,0], X[y==0,1], s=50 , c ='green', label = 'Cluster 1')
plt.scatter(X[y==1,0], X[y==1,1], s=50 , c ='red', label = 'Cluster 2')
plt.scatter(X[y==2,0], X[y==2,1], s=50 , c ='yellow', label = 'Cluster 3')
plt.scatter(X[y==3,0], X[y==3,1], s=50 , c ='violet', label = 'Cluster 4')
plt.scatter(X[y==4,0], X[y==4,1], s=50 , c ='blue', label = 'Cluster 5')
 
#plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=100, c='cyan',label = 'centroids')

plt.title('Customer Group')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')

plt.show()
