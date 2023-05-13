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
m = len(listdata)
n = len(listdata)
mode = [0] * m
for j in range (m):
    mode[j] = [0] * n
print(mode)


#define bandwidth and threshold
h = 420
t = 100

#calculate kernel
def kf (x):
    if (x <= h): return 1
    return 0

def shiftMode(listdata, xm):
    sum1 = 0
    sum2 = 0
    for j in range (len(listdata)):
        sum1 = sum1 +  listdata[j]*kf(abs(listdata[j]-xm))
        sum2 = sum2 + kf(abs(listdata[j]-xm))
    return sum1/sum2

for i in range(len(listdata)):
    m = 0
    mode[i][m] = listdata[i]
    while (abs(mode[i][m] - mode[i][m - 1]) >= t):
        mode[i][m + 1] = shiftMode(listdata,mode[i][m])
        m = m + 1
    mode[i][0] = mode[i][m]

C = []
for i in range(len(listdata)):
    if ([mode[i][0]] not in C):
        C.append([mode[i][0]])
        
print(C)