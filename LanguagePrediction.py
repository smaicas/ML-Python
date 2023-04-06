import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
# from nltk.tokenize import word_tokenize
from nltk import FreqDist, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from string import punctuation
from collections import Counter
from itertools import chain
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re 
from functools import reduce
import string

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import ijson
DATASET_original_jsonl= 'C:/Workspace/ML/Data/Codeforces/dump-original.jsonl/dump-original.jsonl'

with open(DATASET_original_jsonl) as json_file:      
    data = json_file.readlines()
    data = [json.loads(line) for line in data if line.strip()] # filter empty lines
df = pd.DataFrame(data)

# FILTERING
DATASET_FILTER_Accepted_Languages = ['C#', 'Python', 'Java']
df = df[df['language'].apply(lambda x: any(lang in x for lang in DATASET_FILTER_Accepted_Languages))]

# SAMPLE
print("Count:")
# print(len(df))
print("Sample:")
print(df.iloc[0])
print(df.iloc[0]['language'])
print(df.iloc[0]['source'])

# PREPROCESSING
# requires nltk.download('punkt')
# create a translation table with punctuation characters mapped to None
translator = str.maketrans('', '', string.punctuation)

# apply the translation table and convert to lowercase
df['source'] = df['source'].apply(lambda x: re.sub(r'[^\x00-\x7f]', r'', x)) # remove non-ascii characters
df['source'] = df['source'].apply(lambda x: x.lower().translate(translator)) # remove punctuation and convert to lowercase
df['tokens'] = df['source'].apply(word_tokenize) # tokenize

# EXTRACT FEATURES
vectorizer = CountVectorizer(analyzer=lambda x: x) # use tokens as analyzer
tfidf = TfidfTransformer() # use tf-idf transformer
X = vectorizer.fit_transform(df['tokens']) # fit and transform the vectorizer
X = tfidf.fit_transform(X) # transform the X using tf-idf transformer
y = df['language'] # set the target variable

# SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TRAIN & EVALUATE
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier() # create the model
clf.fit(X_train, y_train) # fit the model on the training data
score = clf.score(X_test, y_test) # evaluate the model on the testing data
print(f"Accuracy: {score}")

# PREDICT
new_code_snippet = "print('Hello, world!')"
new_tokens = word_tokenize(new_code_snippet) # tokenize
new_tokens = [word.lower() for word in new_tokens if word not in punctuation] # remove punctuation and convert to lowercase
new_X = vectorizer.transform([new_tokens]) # transform the new code snippet using the vectorizer
new_X = tfidf.transform(new_X) # transform the new code snippet using the tf-idf transformer
prediction = clf.predict(new_X) # predict the programming language of the new code snippet
print(f"Prediction: {prediction}")
