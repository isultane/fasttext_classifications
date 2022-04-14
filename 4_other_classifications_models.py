import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier  
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold

import time

def get_score(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

def kfold_validation(matrix, targets, model_scores):
    folds = KFold(n_splits=10)
    for train_index, test_index in folds.split(matrix):
        # split data to train test
        x_train, x_test, y_train, y_test = matrix[train_index], matrix[test_index],\
                                    targets[train_index], targets[test_index]

        model_scores.append(get_score(LogisticRegression(), x_train, x_test, y_train, y_test))
    
    return model_scores


def train_LR(matrix, targets, scores_l):
    print("time starts here...")
    # kfold cross validation
    start_time = time.time()
    folds = KFold(n_splits=10)
    for train_index, test_index in folds.split(matrix):
        # split data to train test
        x_train, x_test, y_train, y_test = matrix[train_index], matrix[test_index],\
                                    targets[train_index], targets[test_index]

        scores_l.append(get_score(LogisticRegression(), x_train, x_test, y_train, y_test))
        
    train_time = time.time()
    total_traning_testing_time = 'Train & tesing time: {:.2f}s'.format(train_time - start_time)
    print("Resuts after 10 fold cross validations ...")
    print("LogisticRegression(): ", scores_l)
    print("Total training and validation time: " + total_traning_testing_time)

def train_SVM(matrix, targets, scores_svm):
    print("time starts here ...")
    # kfold cross validation
    start_time = time.time()
    folds = KFold(n_splits=10)
    for train_index, test_index in folds.split(matrix):
        # split data to train test
        x_train, x_test, y_train, y_test = matrix[train_index], matrix[test_index],\
                                    targets[train_index], targets[test_index]

        scores_svm.append(get_score(SVC(), x_train, x_test, y_train, y_test))
    train_time = time.time()
    total_traning_testing_time = 'Train & tesing time: {:.2f}s'.format(train_time - start_time)
    print("Resuts after 10 fold cross validations ...")
    print("SVC(): ", scores_svm)
    print("Total training and validation time: " + total_traning_testing_time)

def train_RFC(matrix, targets, scores_rf):
    print("time starts here ...")
    start_time = time.time()
    # kfold cross validation
    folds = KFold(n_splits=10)
    for train_index, test_index in folds.split(matrix):
        # split data to train test
        x_train, x_test, y_train, y_test = matrix[train_index], matrix[test_index],\
                                    targets[train_index], targets[test_index]

        scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), x_train, x_test, y_train, y_test))
    train_time = time.time()
    total_traning_testing_time = 'Train & tesing time: {:.2f}s'.format(train_time - start_time)
    print("Resuts after 10 fold cross validations ...")
    print("RandomForestClassifier(): ", scores_svm)
    print("Total training and validation time: " + total_traning_testing_time)


if __name__ == '__main__':
    scores_l = []
    scores_svm = []
    scores_rf = []

    # 'cooking.train' is just for testing the code. It is going to be updated with te real datasets after pre-processing. 
    data = open('Chromium.train').readlines()

    count_vect = CountVectorizer()

    print('Loading data ...')
    labels, texts = ([], [])
    for line in data:
        label, text = line.split(' ', 1)
        labels.append(label)
        texts.append(text)

    trainDF = pd.DataFrame()
    trainDF['label'] = labels
    trainDF['text'] = texts

    # to fit the text in the dataframe
    # You have to do some encoding before using fit. As it known fit() does not accept Strings.
    count_vect = CountVectorizer()
    matrix = count_vect.fit_transform(trainDF['text'])
    encoder = LabelEncoder()
    targets = encoder.fit_transform(trainDF['label'])

    print("starts LR algorithm ...")
    train_LR(matrix, targets, scores_l)

    print("starts SVM algorithm ...")
    train_SVM(matrix, targets, scores_svm)

    print("starts RFC algorithm ...")
    train_RFC(matrix, targets, scores_rf)
    

# in this .py we need to run the same dataset on SVC, RFC, LR, and other to check the results and compare them with fasttext results.
# Also, we need to comare the performance for each algoeithm along with the same processed data to fasttext performance.