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
from colorama import init, Fore, Style
from NjLog import log, logwarn, logerror, logcall, logok


def load_dataset(path, nitems=None, filter_params=None):
    logcall()
    with open(path) as json_file:
        data = json_file.readlines()
        data = [json.loads(line) for line in data if line.strip()]  # filter empty lines

        # filter data based on filter_params
        if filter_params:
            for key, values in filter_params.items():
                filtered_data = []
                for row in data:
                    if key in row and any(value in row[key] for value in values):
                        filtered_data.append(row)
                data = filtered_data

        # get first nitems if specified
        if nitems:
            data = data[:nitems]

        # create pandas DataFrame from the data
        df = pd.DataFrame(data)

    logok()
    return df


def save_dataset(df, path):
    logcall()
    data = df.to_dict('records')
    with open(path, 'w') as json_file:
        for row in data:
            json.dump(row, json_file)
            json_file.write('\n')
    logok()


def filter_dataset(df):
    logcall()
    dataset_filter_languages = ['C#', 'Python', 'Java']
    df = df[df['language'].apply(lambda x: any(lang in x for lang in dataset_filter_languages))]
    logok()
    return df


def preprocess(df):
    logcall()
    # create a translation table with punctuation characters mapped to None
    translator = str.maketrans('', '', string.punctuation)
    # apply the translation table and convert to lowercase
    df['source'] = df['source'].apply(lambda x: re.sub(r'[^\x00-\x7f]', r'', x))  # remove non-ascii characters
    df['source'] = df['source'].apply(
        lambda x: x.lower().translate(translator))  # remove punctuation and convert to lowercase
    df['tokens'] = df['source'].apply(word_tokenize)  # tokenize
    logok()
    return df


def extract_features(df):
    logcall()
    vectorizer = CountVectorizer(analyzer=lambda x: x)  # use tokens as analyzer
    tfidf = TfidfTransformer()  # use tf-idf transformer
    x = vectorizer.fit_transform(df['tokens'])  # fit and transform the vectorizer
    x = tfidf.fit_transform(x)  # transform the X using tf-idf transformer
    y = df['language']  # set the target variable
    logok()
    return x, y, vectorizer, tfidf


def train_and_evaluate(x, y):
    logcall()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier()  # create the model
    clf.fit(X_train, y_train)  # fit the model on the training data
    score = clf.score(X_test, y_test)  # evaluate the model on the testing data
    logok()
    return clf, score


def predict_language(code_snippet, vectorizer, tfidf, clf):
    logcall()
    translator = str.maketrans('', '', string.punctuation)
    # apply the translation table and convert to lowercase
    code_snippet = code_snippet.apply(lambda x: re.sub(r'[^\x00-\x7f]', r'', x))  # remove non-ascii characters
    code_snippet = code_snippet.apply(lambda x: x.lower().translate(translator))  # remove punctuation and lowercase
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
    logok()
    return prediction


def create_partial_dataframe(dataset_path, items=500, filter_params=None):
    logcall()
    df = load_dataset(dataset_path)

    # Filter the DataFrame based on the filter_params dictionary
    if filter_params:
        for key, values in filter_params.items():
            if not isinstance(values, list):
                values = [values]
            df = df[df[key].isin(values)]

    # Select the first `items` rows of the DataFrame
    new_df = df.head(items)

    # Save the new DataFrame in the same format as the original JSON file but with a different filename
    new_filename = dataset_path.replace('.jsonl', '.test.jsonl')
    new_df.to_json(new_filename, orient='records', lines=True)
    logok()


def main():
    dataset_path = 'C:/Workspace/ML/Data/Codeforces/dump-original.jsonl/dump-original.jsonl'
    df = load_dataset(dataset_path)
    log(f"Original data count:{len(df)}")
    df = filter_dataset(df)
    log(f"Filtered data count:{len(df)}")
    df = preprocess(df)
    X, y, vectorizer, tfidf = extract_features(df)
    clf, score = train_and_evaluate(X, y)
    print(f"Accuracy: {score}")
    code_snippet = "print('Hello, world!')"
    prediction = predict_language(code_snippet, vectorizer, tfidf, clf)
    print(f"Prediction: {prediction}")


if __name__ == '__main__':
    main()
