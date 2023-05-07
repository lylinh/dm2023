# Link: https://www.kaggle.com/code/lylinhnguyen/preprocessing-data
import pandas as pd
import json
import math
import random

#data path
REVIEW_PATH = '/kaggle/input/yelp-dataset/yelp_academic_dataset_review.json' 

#load data method
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


#get length of the data
review_data['review length'] = review_data['text'].apply(len)
print (review_data['review length'])

#push the data to list
listdata = []
for data in review_data['review length']:
    listdata.append(data)
print(listdata)

def k_means_clustering(data, k):
    #get random centroids
    centroids = random.sample(data, k)
    
    while True:
        #define clusters list 
        clusters = []
        
        #initiate cluster list
        for i in range(k):
            clusters.append([])

        for datapoint in data:
            # define distance list
            distances = []
            
            #calculate the distance between the datapoint to each centroid
            for centroid in centroids:
                distances.append(abs(datapoint - centroid))
            
            #find the index of the closest centroid to the datapoint  
            closest_centroid_idx = distances.index(min(distances))
            
            #push the datapoint to the closest cluster
            clusters[closest_centroid_idx].append(datapoint)
        
        #define new centroids
        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroids.append(sum(cluster) / len(cluster))
            else:
                new_centroids.append(random.choice(data))
        
        if new_centroids == centroids:
            break
        
        centroids = new_centroids
        
    return clusters

clusters = k_means_clustering(listdata, 3)

print(len(clusters))
print(clusters)