import sys
import numpy as np
from matplotlib import pyplot as plt
from io import StringIO
import re
import random
import math
from random import choice,uniform
import pandas as pd


# finding a euclidian distance between data point 
#and centroid using square root((x2 - x1)^2  + (y2 - y1)^2)
def EuclideanDistance(data,centroid_list):
	a = 0
	sum_squar_distace ={}
	
	for i in range(len(data)):
		euclidean = data[i]-centroid_list[i]
		a += math.pow(euclidean,2)
		a = math.sqrt(a)
		
	return a
#def fix_k(k,distance):

#clasifying using dstance 
def clustering(centroid_list,data):
	minimum_distance = sys.maxsize 
	position = -1
	
	for i in range(len(centroid_list)):
		distance = EuclideanDistance(data, centroid_list[i])
		# classify the data point in the cluster 
		#where it is having minimum distance from centroid 
		if (distance < minimum_distance):
			minimum_distance = distance
			position = i
	#return the position which determine which cluster it belongs to  
	return position

# again calculating the centroid and updating.
def recalculate_centroid(cluster,centroid,data):
	
	for i in range(len(centroid)):
		c = centroid[i]
		
		c = (c *(cluster-1)+data[i])/float(cluster)
		centroid[i] = round(c,3)
      
	return centroid

def final_cluster(centroid,data):
	final_cluster = [[] for i in range(len(centroid))]

	for data_point in data:
		# finding in which cluster data point belong to and 
		#then allocating it to the particular cluster.
		position = clustering(centroid,data_point)
		final_cluster[position].append(data_point)
	return final_cluster

def clustering_handler(data,k):
	n = len(data[0])
	iterations = 30
	# to intitialze centroid find out feature minimunm and maximum value.
	minimum_val = [sys.maxsize for i in range(n)] 
	maximum_val= [-sys.maxsize -1 for i in range(n)] 
	for j in data:
		for i in range(len(j)):
			
			if (j[i] < minimum_val[i]):
				minimum_val[i] = j[i]
			if (j[i] > maximum_val[i]):
				maximum_val[i] = j[i]
	

	# allocate array to intialize centroid
	centroid = [[0 for i in range(n)] for j in range(k)] 		
	# from the minimum and maximum value found 
	#for the feature initalize centroid 
	for centroid_list in centroid:
		for i in range(len(centroid_list)):
			centroid_list[i] = uniform(minimum_val[i]+1, maximum_val[i]-1)
	# allocating array to get data into cluster
	size_of_cluster= [0 for i in range(len(centroid))]
	data_point_cluster = [0 for i in range(len(data))] 

	# findout centroid and classify datapoint into cluster
	for j in range(iterations):
		cluster_change = True
		for m in range(len(data)):
			temp = data[m];
			#returns in which cluster data point lie to and 
			# then update the centroid
			position = clustering(centroid,temp)
			size_of_cluster[position] += 1
			centroid[position] = recalculate_centroid(size_of_cluster[position],centroid[position],temp)

			# after recalculating the mean check if the data point lie into different cluster
			if ( position !=data_point_cluster[m]):
				cluster_change = False
			# if true then allocate to that cluster
			data_point_cluster[m] = position
		# cluster did not got change then nothing
		if (cluster_change == True):
			break
	# actual centroid after all calculation		
	final_centroid = centroid
	# according to the final centroid classify the data points 
	#into related cluster
	final_output = final_cluster(centroid,data)
	
	return final_centroid,final_output

					

def PlotClusters(clusters,center):
    n = len(clusters) 
    
    # ploting two dimensions only
    tmp = [[] for i in range(n)] 

    for i in range(n):
        cluster = clusters[i] 
        for j in cluster:
            tmp[i].append(j) 

    cluster_colors = ['r','b','g','c','m','y'] 

    for i in tmp:
        #Choose color randomly from list, then remove it
        #(to avoid duplicates)
        c = choice(cluster_colors) 
        cluster_colors.remove(c) 

        x1 = [] 
        x2 = [] 

        for j in i:
            x1.append(j[0]) 
            x2.append(j[1]) 
        plt.title('Clustered Datapoints')
        plt.plot(x1,x2,'o',color=c) 

    for i in range(len(center)):
    	plt.scatter(center[i][0], center[i][1], marker="x", color='r')
    plt.show()
 
    #plt.show() 
        

with open(sys.argv[1],'rt')as my_file:
	#read data into list
	data = []
	for line in my_file:
		data.append((re.findall(r"[-+]?\d*\.\d+|\d+", line)))
	data.remove(data[150])
	
	final_data = []
	
	for i in range(0, len(data)):
		process_data = data[i]
		temp_data = []
		for j in range(len(process_data)):
				changto = float(process_data[j])
				temp_data.append(changto)
		final_data.append(temp_data)
	# shffleing data to get more confident result 
	random.shuffle(final_data) 

	cluster = 3
	# plotting the cluster before any change has been made 
	for j in final_data:
			ax = plt.scatter(x=j[0],y=j[1], color = 'r')
			plt.title('Data visualization')
	
	plt.show()

	# k-menas algorithm - find centroid and make cluster
	center,y=clustering_handler(final_data,cluster)
	# visulaizing cluster
	PlotClusters(y,center)

	# how to choose the value of k
	from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(final_data)
    wcss.append(kmeans.inertia_) # find within cluster distance 
 
    
#plot a graph with number of cluster and WCSS - within cluster distance 
plt.plot(range(1, 11), wcss)
plt.title('Elbow plotting')
plt.xlabel('K - Number of Cluters')
plt.ylabel('WCSS- Within cluster distance') #within cluster sum of squares
plt.show()


	