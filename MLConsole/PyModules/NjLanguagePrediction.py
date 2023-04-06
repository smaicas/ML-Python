import numpy as np
import pandas as pd
import json
from nltk import FreqDist, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from string import punctuation
from collections import Counter
from itertools import chain
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
import re
from functools import reduce
import string
from sklearn.tree import DecisionTreeClassifier


def load_dataset(path):
    with open(path) as json_file:
        data = json_file.readlines()
        data = [json.loads(line) for line in data if line.strip()]  # filter empty lines
    df = pd.DataFrame(data)
    return df


def preprocess(df):
    # create a translation table with punctuation characters mapped to None
    translator = str.maketrans('', '', string.punctuation)
    # apply the translation table and convert to lowercase
    df['source'] = df['source'].apply(lambda x: re.sub(r'[^\x00-\x7f]', r'', x))  # remove non-ascii characters
    df['source'] = df['source'].apply(
        lambda x: x.lower().translate(translator))  # remove punctuation and convert to lowercase
    df['tokens'] = df['source'].apply(word_tokenize)  # tokenize
    return df


def extract_features(df):
    vectorizer = CountVectorizer(analyzer=lambda x: x)  # use tokens as analyzer
    tfidf = TfidfTransformer()  # use tf-idf transformer
    X = vectorizer.fit_transform(df['tokens'])  # fit and transform the vectorizer
    X = tfidf.fit_transform(X)  # transform the X using tf-idf transformer
    y = df['language']  # set the target variable
    return X, y, vectorizer, tfidf


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier()  # create the model
    clf.fit(X_train, y_train)  # fit the model on the training data
    score = clf.score(X_test, y_test)  # evaluate the model on the testing data
    return clf, score


def predict_language(code_snippet, vectorizer, tfidf, clf):
    # tokenize
    new_tokens = word_tokenize(code_snippet)
    # remove punctuation and convert to lowercase
    new_tokens = [word.lower() for word in new_tokens if word not in punctuation]
    # transform the new code snippet using the vectorizer
    new_X = vectorizer.transform([new_tokens])
    # transform the new code snippet using the tf-idf transformer
    new_X = tfidf.transform(new_X)
    # predict the programming language of the new code snippet
    prediction = clf.predict(new_X)
    return prediction


def main():
    DATASET_PATH = 'C:/Workspace/ML/Data/Codeforces/dump-original.jsonl/dump-original.jsonl'
    DATASET_FILTER_LANGUAGES = ['C#', 'Python', 'Java']
    df = load_dataset(DATASET_PATH)
    df = df[df['language'].apply(lambda x: any(lang in x for lang in DATASET_FILTER_LANGUAGES))]
    df = preprocess(df)
    X, y, vectorizer, tfidf = extract_features(df)
    clf, score = train_and_evaluate(X, y)
    print(f"Accuracy: {score}")
    code_snippet = "print('Hello, world!')"
    prediction = predict_language(code_snippet, vectorizer, tfidf, clf)
    print(f"Prediction: {prediction}")


if __name__ == '__main__':
    main()
