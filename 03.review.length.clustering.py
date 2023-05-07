# https://www.kaggle.com/code/lylinhnguyen/preprocessing-data

import pandas as pd
import json
import math

#data paths
REVIEW_PATH = '/kaggle/input/yelp-dataset/yelp_academic_dataset_review.json'   

#load the data
def load_rows(file_path, nrows=None, only_return_count=False, verbose=True):
    with open(file_path) as json_file:
        count = 0
        objs = []
        line = json_file.readline()
        while (nrows is None or count<nrows) and line:
            count += 1
            if not only_return_count:
                obj = json.loads(line)
                objs.append(obj)
            line = json_file.readline()
        if only_return_count:
            return count
        
        return pd.DataFrame(objs)

#load first 10 records  
review_data = load_rows(REVIEW_PATH,10)
print(review_data['text'])

#get length
review_data['review length'] = review_data['text'].apply(len)
print (review_data['review length'])


#push all cluster to list
clusters = []
for data in review_data['review length']:
    clusters.append([data])
print(clusters)


#calculate the min distance between clusters
def min_distance_between_clusters(list1, list2):
    mindistance = 1000
    for i in list1:
        for j in list2:
            mindistance = min(abs(i - j), mindistance)
    return mindistance


#merger clusters
while(len(clusters)>3):
    mindistance = 1000
    firstclusterindex = 0
    secondclusterindex = 0
    
    #check all cluster and find the min distance between two clusters
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            distancecluster = min_distance_between_clusters(clusters[i], clusters[j])
            if distancecluster < mindistance:
                firstclusterindex = i
                secondclusterindex = j
                mindistance = distancecluster
    
    #merger clusters
    for i in range(len(clusters[secondclusterindex])):
        clusters[firstclusterindex].append(clusters[secondclusterindex][i])
    
    #delete second cluster after merging in first cluster
    del clusters[secondclusterindex]

#check the final clusters
print(len(clusters))
print(clusters)