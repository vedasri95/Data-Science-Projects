
# coding: utf-8

# In[52]:


#importing required packages
import pandas as pd
import math
from matplotlib import pyplot as plt
import random
from copy import deepcopy
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import seaborn as sns


# In[53]:


#Reading 'TwoDimHard' dataset into a pandas dataframe
data = pd.read_csv('TwoDimHard.csv',delimiter='\,',engine='python')


# In[54]:


#Utility function to calculate Euclidean distance between two data points
def euclideanDistance(instance1, instance2):
    distance = 0
    for i in range(len(instance1)):
        distance += pow((instance1[i] - instance2[i]), 2)
    return math.sqrt(distance)


# In[55]:


##A utility function that takes an instance and centroids as input and assignes the instance to the nearest centroid and returns the centroid
def assignCluster(instance, C):
    minimum = euclideanDistance(instance,C[0])
    cluster = 0;
    for j in range(len(C)):
        dist = euclideanDistance(instance, C[j])
        if dist < minimum:
            minimum = dist
            cluster = j               
    return cluster


# In[56]:


#A utility function that creates a scatter plot for the data and the predicted cluster centres 
def plotPoints(data, C):
    colors = ['y', 'g', 'b', 'r']
    for centroid in C:
        plt.scatter(centroid[0], centroid[1], s = 130, marker = "x", c='black')
    for i in range(len(C)):
        for j in range(len(data)):
            if data.iloc[j]['assigned_cluster'] == i+1:
                plt.scatter(data.iloc[j]['X.1'], data.iloc[j]['X.2'], s=7, c = colors[i])
    plt.show()


# In[57]:


#A utility function to calculate SSE for each cluster and the overall SSE
def calculate_error(df, C, features):
    SSE = 0
    for i in range(len(C)):
        count = 0
        SSE_cluster = 0
        for j in range(len(df)):
            cluster = df.iloc[j]['assigned_cluster'];
            if int(cluster) == i+1:
                count += 1
                point = np.asarray(df.iloc[j][features])
                dist = euclideanDistance(C[i], point)
                SSE_cluster = SSE_cluster + (dist*dist)
        print("SSE Cluster",i+1,": ", SSE_cluster)
        SSE = SSE + SSE_cluster
    print("Over all SSE : ", SSE)


# In[58]:


#Utility function to calculate the SSB. 
def getSSB(df, C, sizes, features):
    SSB = 0
    values = []
    for i in features:
        values.append(df[i].mean()) 
    for i in range(len(C)):
        euclideanDistance(C[i], np.asarray(values))
        SSB += sizes[i] * pow(euclideanDistance(C[i], np.asarray(values)),2)
    print("SSB: ", SSB)


# In[59]:

#function for k means clustering which takes k and the data as input.
def kmeans(data, k):
    features = ['X.1', 'X.2']
    #First k points in the dataset are chosen as initial centroids
    x = data.iloc[1:k+1]['X.1']
    y = data.iloc[1:k+1]['X.2']
    
    output_df = pd.DataFrame()
    new_clusters = np.array(list(zip(x,y)), dtype=np.float64)
    
    #creating an array to store old clusters
    old_clusters = np.zeros(new_clusters.shape)
    clusters = np.zeros(len(data))
    count=0
    
    #Keep iterating until the old and new centroids converge
    while not np.array_equal(old_clusters, new_clusters):
        count += 1
        #print(count)
        #Iterating through the data set to assign new clusters to each data point
        for i in range(len(data)):
            point = np.asarray(data.iloc[i][features])
            #storing the assigned cluster for each data point in 'clusters' array
            clusters[i] = assignCluster(point, new_clusters)
        #Storing the current centroids
        old_clusters = deepcopy(new_clusters)
        sizes = []
        
        #Loop to calculate new centroids
        for i in range(k):
            points = []
            size = 0
            #Iterating through the data set to collect all points belonging to a particular cluster
            for j in range(len(data)):
                    if clusters[j] == i:
                        size += 1
                        points.append(np.array(data.iloc[j][features]))
            #New centroid of the cluster is the average of data points belonging to that cluster
            new_clusters[i] = np.mean(np.asarray(points), axis=0)
            sizes.append(size)
    #While loop ends here. Stored the final clusters in a special attribute 'assigned_cluster'
    data['assigned_cluster'] = clusters.astype(int)+1
    
    #preparing output to write into csv file
    output_df['ID'] = data['ID']
    output_df['assigned_cluster'] = data['assigned_cluster']
    #writing the output to csv file
    output_df.to_csv('outputFile.csv', sep=',', columns = output_df.columns)
    
    #creating the scatter plot for the dataset with the 'assigned_cluster' attribute
    plotPoints(data, new_clusters)
    getSSB(data, new_clusters, np.asarray(sizes), features)
    calculate_error(data, new_clusters, features)
    return clusters


# In[60]:

#Reading value for k from the user.
k = int(input("Please enter the value of k for the TwoDimHard dataset:"))


# In[61]:

#calling the function to perform clustering
kmeans(data, k)

