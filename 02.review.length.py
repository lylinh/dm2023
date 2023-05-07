import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import math

#Data Paths
REVIEW_PATH = '/kaggle/input/yelp-dataset/yelp_academic_dataset_review.json'    

# read file
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
    
review_data = load_rows(REVIEW_PATH,5000)
print(review_data['text'])

#get length of review
review_data['review length'] = review_data['text'].apply(len)
print (review_data['review length'])


# calculate mean and standard deviation
def mean(data):
  n = len(data)
  mean = sum(data) / n
  return mean
 
def variance(data):
  n = len(data)
  mean = sum(data) / n
  deviations = [(x - mean) ** 2 for x in data]
  variance = sum(deviations) / n
  return variance
 
def stdev(data):
  var = variance(data)
  std_dev = math.sqrt(var)
  return std_dev

plt.hist(review_data['review length'],bins=100, density=True, alpha=0.6, color='green')

# mean and standard deviation
mu = mean(review_data['review length'])
std =  stdev(review_data['review length'])

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin-1000, xmax+200, 100)
p = norm.pdf(x, mu, std)
  
plt.plot(x, p, 'k', linewidth=2)
title = "Fit Values: {:.2f} and {:.2f}".format(mu, std)
plt.title(title)

plt.savefig('02.review.length.pdf')
plt.show()