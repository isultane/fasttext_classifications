# Author: Sultan S. Alqahtani
# Date: 06/16/2021 

#import imp
from genericpath import isfile
from ntpath import join
from pydoc import doc
import string
from tracemalloc import stop
#from numpy import vectorize
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

from nltk import word_tokenize
from collections import defaultdict
from nltk import FreqDist
from nltk.corpus import stopwords
import os
import random
import string
import pickle

#ML algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier  



stop_words = set(stopwords.words('english'))
stop_words.add('said')
stop_words.add('mr')

# BASE_DIR = "path to files"
# LABELS = ['labels list']
BSE_DIR = './data/balanced_data/'


# def creat_data_set():
#     with open('pathtofile', 'w', encoding='utf9') as outfile:
#         for label in LABELS:
#             dir = '%s%s' % (BASE_DIR, label)
#             for filename in os.listdir(dir):
#                 fullfilenme = '%s%s' % (dir, filename)
#                 print(fullfilenme)
#                 with open(fullfilenme, 'rb') as file:
#                     text = file.read().decode(errors='replace').replace('\n', '')
#                     outfile.write('%s\t%s\t%s\n' %  (label, filename, text))

def setup_docs(porject_data):
    docs = [] #(label, text)
    with open (porject_data, 'r', encoding='utf8') as datafile:
        for row in datafile:
            parts = row.split(' ', 1)
            doc = (parts[0], parts[1].strip())

            docs.append(doc)
    return docs

def clean_text(text):
    # remove punctution 
    text = text.translate(str.maketrans('', '', string.punctuation))

    # convert to lower case
    text = text.lower()
    return text

def get_tokens(text):
    # get individual words
    tokens = word_tokenize(text)
    # remove common words that are useless
    tokens = [t for t in tokens if not t in stop_words]
    return tokens

def print_frequency_dis(docs):
    tokens = defaultdict(list)

    # lets make a gaint list of all the words for ech category
    for doc in docs:
        doc_label = doc[0]
        doc_text = clean_text(doc[1])

        doc_tokens = get_tokens(doc_text)

        tokens[doc_label].extend(doc_tokens)

        for category_label, category_tokens in tokens.items():
            print(category_label)
            fd = FreqDist(category_tokens)
            print(fd.most_common(20))

def get_splits(docs):
    #scramble docs
    random.shuffle(docs)

    X_train = []    #traingin documents
    y_train = []    #corresponding training labels

    X_test = []     #test documents
    y_test = []     #corresponding testing labels

    pivot = int(.80 * len(docs))

    for i in range(0, pivot):
        X_train.append(docs[i][1])
        y_train.append(docs[i][0])

    for i in range(pivot, len(docs)):
        X_test.append(docs[i][1])
        y_test.append(docs[i][0])

    return X_train, X_test, y_train, y_test 

def evaluate_clssifier(title, classifier, vectorizer, X_test, y_test):
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_tfidf)

    precision = metrics.precision_score(y_test, y_pred, average="binary", pos_label="__label__sec")
    recall = metrics.recall_score(y_test, y_pred, average="binary", pos_label="__label__sec")
    f1 = metrics.f1_score(y_test,y_pred, average="binary", pos_label="__label__sec")

    print("%s\t%f\t%f\t%f\n" % (title, precision, recall, f1))

def train_classifier(classifier_title,classifier_algorithm,docs):
    X_train, X_test, y_train, y_test = get_splits(docs)

    # the object that turns text into vectors 
    vectorizer = CountVectorizer(stop_words='english',ngram_range=(1,3),min_df=3, analyzer='word')

    # crete doc-term matrix
    dtm = vectorizer.fit_transform(X_train)

    # train the classfier 
    classifier = classifier_algorithm.fit(dtm, y_train)

    evaluate_clssifier(classifier_title, classifier, vectorizer, X_train, y_train)
    evaluate_clssifier(classifier_title, classifier, vectorizer, X_test, y_test)

#   store the classifier
#   clf_filename = classifier_title+'.pkl'
#   pickle.dump(classifier, open(clf_filename, 'wb'))

#   also store the vectorizer so we can transform new data
#   vec_filename = 'count_vectorizer.pkl'
#   pickle.dump(vectorizer, open(vec_filename, 'wb'))

def classify(text):
    #load classifier
    clf_filename = ''
    nb_clf = pickle.load(open(clf_filename, 'rb'))

    #vectorize the new text
    vec_filname = ''
    vectorizer = pickle.load(open(vec_filname, 'rb'))

    pred = nb_clf.predict(vectorizer.transform([text]))

    print(pred[0])


if __name__ == '__main__':
    #create_data_set()
    projects_files = [f for f in os.listdir(BSE_DIR) if isfile(join(BSE_DIR, f))]
   
    tested_projects = []
    training_list = []
    
    for porject_data in projects_files:
        if porject_data not in tested_projects:
            tested_projects.append(porject_data)
        training_list = [i for i in projects_files if i not in tested_projects]
        for train in training_list:
            docs = setup_docs(BSE_DIR+train)
            #print_frequency_dis(docs)
            train_classifier('LogisticRegression', LogisticRegression(),docs)
            train_classifier('RandomForestClassifier', RandomForestClassifier(),docs)
            # train_classifier('GaussianNB', GaussianNB(),docs)
            train_classifier('KNeighborsClassifier', KNeighborsClassifier(),docs)
            train_classifier('MLPClassifier', MLPClassifier(),docs)
            #new_doc = setup_docs(BSE_DIR+porject_data)
            #classify(new_doc)
        training_list.clear()
        tested_projects.clear()
    print("Done!")

    # useful link https://stackoverflow.com/questions/10592605/save-classifier-to-disk-in-scikit-learn 