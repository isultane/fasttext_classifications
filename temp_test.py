# Author: Sultan S. Alqahtani
# Date: 06/16/2021 

#import imp
import csv
from genericpath import isfile
from ntpath import join
from pydoc import doc
import string
import numpy
from tracemalloc import stop
#from numpy import vectorize
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import KFold

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


def train_classifier(classifier_title,classifier_algorithm,docs, project_name):
    #split document into 80% training and 20% testing
    X_train, X_test, y_train, y_test = get_splits(docs)
   
    y_train = numpy.array(y_train)
    X_train = numpy.array(X_train)

    kf = KFold(n_splits=10)
    # kf.get_n_splits(X_train)
    metrics = []
    for train_index, test_index in kf.split(X_train):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train_, X_test_ = X_train[train_index], X_train[test_index]
        y_train_, y_test_ = y_train[train_index], y_train[test_index]
        vect = CountVectorizer(stop_words='english',ngram_range=(1,3),min_df=3, analyzer='word')
        
        if classifier_title is 'GaussianNB':
            X_train_dtm = vect.fit_transform(X_train_).toarray()
            X_test_dtm = vect.transform(X_test_).toarray()
        else:
            X_train_dtm = vect.fit_transform(X_train_)
            X_test_dtm = vect.transform(X_test_)

        nb = classifier_algorithm
        nb.fit(X_train_dtm, y_train_)
        y_pred_class = nb.predict(X_test_dtm)
        
        metrics.append(accuracy_score(y_test_, y_pred_class))

        report = classification_report(y_test_, y_pred_class, output_dict=True )
        precision =  report['macro avg']['precision'] 
        recall = report['macro avg']['recall']    
        f1_score = report['macro avg']['f1-score']
        pf = calculte_pf(y_test_, y_pred_class)
        g_score = (2*recall*(1-pf))/(recall + (1-pf))

        write_kfold_results(precision, recall, f1_score, pf, g_score, project_name, classifier_title)

    # metrics = numpy.array(metrics)
    # print('Mean accuracy: ', numpy.mean(metrics, axis=0))
    # print('Std for accuracy: ', numpy.std(metrics, axis=0))

# write kfold results to CSV file
def write_kfold_results(p_score, r_socre, f1_score, pf, g_score,project, model_name):
    with open('./data/bug_reports/results/V2_balanced_data_'+str(model_name)+'_results.csv', 'a') as results:
            write = csv.writer(results)
            data = [p_score, r_socre, f1_score, pf, g_score,project, model_name]
            write.writerow(data)
    
def calculte_pf(y_test, predictions):
    CM = confusion_matrix(y_test, predictions)
    TN = CM[0][0]
    #  FN = CM[1][0]
    #  TP = CM[1][1]
    FP = CM[0][1]
    if FP == 0 and TN == 0:
        pf = 1
        return pf
    else:
        pf = FP / (FP + TN)
        return pf

if __name__ == '__main__':
    #create_data_set()
    projects_files = [f for f in os.listdir(BSE_DIR) if isfile(join(BSE_DIR, f))]

    tested_projects = []
    training_list = []
    
    for target_project in projects_files:
        #preparing the training project data
        print('Processing project:' + target_project)
        docs = setup_docs(BSE_DIR+target_project)
        #print_frequency_dis(docs)

        train_classifier('LogisticRegression', LogisticRegression(),docs, target_project)
        train_classifier('RandomForestClassifier', RandomForestClassifier(),docs,target_project)
        train_classifier('GaussianNB', GaussianNB(),docs,target_project)
        train_classifier('KNeighborsClassifier', KNeighborsClassifier(),docs,target_project)
        train_classifier('MLPClassifier', MLPClassifier(),docs,target_project)        
    print("Done!")

   
    # finl version