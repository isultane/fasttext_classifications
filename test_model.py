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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score

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
    #split document into 80% training and 20% testing
    X_train, X_test, y_train, y_test = get_splits(docs)

    # the object that turns text into vectors 
    vectorizer = CountVectorizer(stop_words='english',ngram_range=(1,3),min_df=3, analyzer='word')

    # crete doc-term matrix
    if classifier_title is 'GaussianNB':
        dtm = vectorizer.fit_transform(X_train).toarray()
    else:
        dtm = vectorizer.fit_transform(X_train)

    # train the classfier 
    classifier = classifier_algorithm.fit(dtm, y_train)

    # evaluate_clssifier(classifier_title, classifier, vectorizer, X_train, y_train)
    # evaluate_clssifier(classifier_title, classifier, vectorizer, X_test, y_test)

    # X_test_tfidf = vectorizer.transform(X_test)
    # y_pred = classifier.predict(X_test_tfidf)   
    
    # print(confusion_matrix(y_test,y_pred))
    # print(classification_report(y_test,y_pred))
    # print(accuracy_score(y_test, y_pred))

    # store the classifier
    clf_filename = classifier_title+'.pkl'
    pickle.dump(classifier, open(clf_filename, 'wb'))

    # also store the vectorizer so we can transform new data
    vec_filename = classifier_title+'_count_vectorizer.pkl'
    pickle.dump(vectorizer, open(vec_filename, 'wb'))

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
    
def validate(classifier_title,target_project, target_title, training_title):
    X_target_train, X_target_test,y_target_train, y_target_test = get_splits(target_project)

    #load classifier
    clf_filename = classifier_title+'.pkl'
    nb_clf = pickle.load(open(clf_filename, 'rb'))

    #vectorize the new text
    vec_filname = classifier_title+'_count_vectorizer.pkl'
    vectorizer = pickle.load(open(vec_filname, 'rb'))

    if classifier_title is 'GaussianNB':
        y_pred = nb_clf.predict(vectorizer.transform(X_target_test).toarray())
    else:
        y_pred = nb_clf.predict(vectorizer.transform(X_target_test))

   
    report = classification_report(y_target_test, y_pred, output_dict=True )
    precision =  report['macro avg']['precision'] 
    recall = report['macro avg']['recall']    
    f1_score = report['macro avg']['f1-score']
    pf = calculte_pf(y_target_test, y_pred)
    g_score = (2*recall*(1-pf))/(recall + (1-pf))

    print("Writing the results of %s classifier after validating  %s project data.\n" % (classifier_title,target_title))
    with open('updated_'+target_title, 'a') as file:
        file.write('\n Results of ' + classifier_title + '_vs_' + training_title)
        file.write('\n macro_precision : {}'.format(precision))
        file.write('\n macro_recall : {}'.format(recall))
        file.write('\n macro_f1 : {}'.format(f1_score))
        file.write('\n pf : {}'.format(pf))
        file.write('\n g_score : {}'.format(g_score))
    
    # print(confusion_matrix(y_target_test,y_pred))
    # print(classification_report(y_target_test,y_pred))
    # print(accuracy_score(y_target_test, y_pred))

    print("Deleting tmp files: model and vectorizer files")
    if os.path.exists(clf_filename):
        os.remove(clf_filename)
    if os.path.exists(vec_filname):
        os.remove(vec_filname)


if __name__ == '__main__':
    #create_data_set()
    projects_files = [f for f in os.listdir(BSE_DIR) if isfile(join(BSE_DIR, f))]

    tested_projects = []
    training_list = []
    
    for target_project in projects_files:
        if target_project not in tested_projects:
            tested_projects.append(target_project)
        training_list = [i for i in projects_files if i not in tested_projects]
        for train in training_list:
            #preparing the training project data
            docs = setup_docs(BSE_DIR+train)
            #print_frequency_dis(docs)
            #preparing the target project data 
            new_docs = setup_docs(BSE_DIR+target_project)

            train_classifier('LogisticRegression', LogisticRegression(),docs)
            validate('LogisticRegression',new_docs,target_project, train)

            train_classifier('RandomForestClassifier', RandomForestClassifier(),docs)
            validate('RandomForestClassifier',new_docs,target_project, train)

            train_classifier('GaussianNB', GaussianNB(),docs)
            validate('GaussianNB',new_docs,target_project, train)

            train_classifier('KNeighborsClassifier', KNeighborsClassifier(),docs)
            validate('KNeighborsClassifier',new_docs,target_project, train)

            train_classifier('MLPClassifier', MLPClassifier(),docs)
            validate('MLPClassifier',new_docs,target_project, train)

            # validating target project
            
        training_list.clear()
        tested_projects.clear()
    print("Done!")

    # useful link https://stackoverflow.com/questions/10592605/save-classifier-to-disk-in-scikit-learn 
    # useful link https://stackabuse.com/text-classification-with-python-and-scikit-learn/