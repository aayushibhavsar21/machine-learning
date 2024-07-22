# In supervised learning, we have target variables. We can train models based on the salaries of employees based on their department and position.
# But in supervised learning, we do not have any target variables. We have to find characteristics based on the given data. 
# 
# 
# K means to find clusters in a graph.
# For clusters, we have to identify the centres of clusters. For the centre, we will take any random point and find the distance between the data points.
#     center.  Which datapoint belongs to which cluster is defined by which cluster it is closer to.
# After deciding on clusters, we will relocate the cluster centers, recalculate the distances from each data point, and redefine the cluster.
# We will repeat this same process until the data point does not change its cluster, even after changing the position of the centre points. 
# 
# 
# In this method, we have to declare k (the number of clusters). The best number of K is difficult to identify for large amounts of data.
# So, for that, we will use the albow method.
# albow method: in this method, we will find the sum of error (SSE) = summation of distance (data point - center)^2 from 0 to the number of data points for each cluster.
#                                          final sum of error (SSE) = sum of (sum of error of each cluster)
# We will find SSE for all possible clusters of graphs (e.g., 1 cluster for the entire graph, 2 clusters for the entire graph,..., n clusters for the entire graph).
# Generally, SSE is max for 1 cluster for the entire graph and 0 for n clusters.
# Error keeps reducing, and we will refer to the elbow point in the graph, and based on it, we will define the number of k.



import pandas as pd
df = pd.read_csv("A:/machine learning/Unsupervised learning/income.csv")

import matplotlib.pyplot as plt
plt.scatter(df.Age,df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.show()

# based on scatter plot we can identify that there might be 3 clusters .

# Elbow method to find best value of k

from sklearn.cluster import KMeans

SSE = []
k_rng = range(1,10)

for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age','Income($)']])
    SSE.append(km.inertia_)   # -> it will give us SSE

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,SSE)
plt.show()


km = KMeans(n_clusters=2)

y_predicted = km.fit_predict(df[['Age','Income($)']])
print(y_predicted)        # -> which data point belongs to which cluster 

df['cluster'] = y_predicted

print(km.cluster_centers_)

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]

plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')

plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.legend()
plt.show()

#Preprocessing using min max scaler

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])

scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])

print(df.head())

km = KMeans(n_clusters=2)
y_predicted = km.fit_predict(df[['Age','Income($)']])
y_predicted

df['cluster']=y_predicted

print(km.cluster_centers_)

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]

plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')

plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.legend()
plt.show()



#___________ cluster for iris flower ___________

from sklearn.datasets import load_iris
iris = load_iris()

import pandas as pd
df = pd.DataFrame(iris.data , columns=iris.feature_names)

df = df.drop(['sepal length (cm)','sepal width (cm)'], axis='columns')
print(df.head())

import matplotlib.pyplot as plt 
plt.scatter(df['petal length (cm)'],df['petal width (cm)'])
plt.xlabel('length')
plt.xlabel('width')
plt.show()

from sklearn.cluster import KMeans
SSE = []

for k in range(1,10):
    km = KMeans(n_clusters=k)
    km.fit(df[['petal length (cm)','petal width (cm)']])
    SSE.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(range(1,10),SSE)
plt.show()

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['petal length (cm)','petal width (cm)']])

df['cluster'] = y_predicted

df1=df[df['cluster']==0]
df2=df[df['cluster']==1]
df3=df[df['cluster']==2]

plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color ='red' )
plt.scatter(df2['petal length (cm)'],df2['petal width (cm)'],color ='green' )
plt.scatter(df3['petal length (cm)'],df3['petal width (cm)'],color ='blue' )
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.show()
