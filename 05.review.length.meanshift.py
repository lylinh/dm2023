#kaggle link: https://www.kaggle.com/code/lylinhnguyen/preprocessing-data
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

#define mode
mode = [[] for _ in range(len(listdata))]
print(mode)


#define bandwidth and threshold
h = 420
t = 100

#calculate kernel
def kf (x):
    if (x <= h): return 1
    return 0

def shiftMode(listdata, xm):
    weighted_sum = 0
    total_weights = 0
    for j in range (len(listdata)):
        weighted_sum = weighted_sum + listdata[j]*kf(abs(listdata[j]-xm))
        total_weights = total_weights + kf(abs(listdata[j]-xm))
    return weighted_sum/total_weights


C=[]
# define clusters based on the length
for i in range(len(listdata)):
    m = 0
    mode[i].append(listdata[i])
    while (abs(mode[i][m] - mode[i][m - 1]) >= t):
        mode[i].append(shiftMode(listdata,mode[i][m]))
        m = m + 1
    if [mode[i][m]] not in C:
        C.append([mode[i][m]])

# push review data to clusters
Clusters = []
for i in range (len(C)):
    distances = []
    for j in range (len(listdata)):
        distances.append(abs(listdata[j] - C[i]))
    index = distances.index(min(distances))
    Clusters[i].append([review_data[index]])
    
print(Clusters)