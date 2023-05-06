# Link: https://www.kaggle.com/code/lylinhnguyen/preprocessing-data

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import re
from nltk.corpus import stopwords


#Data Paths
REVIEW_PATH = '/kaggle/input/yelp-dataset/yelp_academic_dataset_review.json'      #Contains full review text data including the user_id that wrote the review and the business_id the review is written for.

def load_rows(file_path, nrows=None, verbose=True):
    
    with open(file_path) as json_file:
        count = 0
        objs = []
        line = json_file.readline()
        while (nrows is None or count<nrows) and line:
            count += 1
            objLog = {}
            obj = json.loads(line)
            objLog = obj['text']
            objs.append(objLog)
            line = json_file.readline()
        
        return objs
    

array_data = load_rows(REVIEW_PATH,50)
print(array_data)


#Remove punctuations, lower case, trim. . .
#Break by space
#Remove stop words: a, in, of, the, at. . 

all_words = []
all_words_each_record = []

commonly_words = ["like","liked","likewise","get"]

for sentence in array_data:
    
    sentence = re.sub(r"\W+", " ", sentence)
    sentence = re.sub("[^a-zA-Z]+", " ", sentence)
    sentence = sentence.lower()
    sentence = sentence.strip()
    newSentence = ''
    words = sentence.split()
    list_convert = []
    for w in words:
        if w not in stopwords.words("english"):
            list_convert.append(w)
            if not all_words.__contains__(w) and not commonly_words.__contains__(w):
                all_words.append(w)
    
    all_words_each_record.append(list_convert)

all_words=sorted(all_words)
print(all_words_each_record)

print('-----------------------')
print(all_words)



#Find DF(w)
DFw = {}
total_record = len(all_words_each_record)
for w in all_words:
    counter = 0
    for record in all_words_each_record:
        if record.__contains__(w):
            counter += 1
    DFw[w]=counter/total_record

print(DFw)



#Find IDF(w)
import math

IDFw = {}

total_record = len(all_words_each_record)
for w in all_words:
    n = total_record/DFw[w]
    IDFw[w] = math.log(n)
    
print(IDFw)



#find TF(w,d):
TFwd = []
TFIDF = []

for r in all_words_each_record:
    TFwdr = {}
    TFIDFr = {}
    for w in r:
        if not TFwdr.__contains__(w) and IDFw.__contains__(w):
            counter = 0
            for w2 in r:
                if w == w2:
                    counter += 1
                    
            TFwdr[w] = counter
            TFIDFr[w] = TFwdr[w] * IDFw[w]
    TFwd.append(TFwdr)
    TFIDF.append(TFIDFr)
    
print(TFwd)
print('--------------')
print(TFIDF)
