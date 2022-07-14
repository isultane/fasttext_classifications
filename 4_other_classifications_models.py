import csv
from os import listdir
from os.path import isfile, join

import pandas as pd
from sklearn import metrics

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier  
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.naive_bayes import GaussianNB

# ML models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import time

data_path = './data/balanced_data/'

def get_score(model, x_train, x_test, y_train, y_test, project, model_name):
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    #print(classification_report(y_test, predictions))  
    precision,recall,fscore,support=score(y_test,predictions,average='macro')     

    try:
        CM = confusion_matrix(y_test, predictions)
        TN = CM[0][0]
    #  FN = CM[1][0]
    #  TP = CM[1][1]
        FP = CM[0][1]
        if FP == 0 and TN == 0:
            pf = 1
        else:
            pf = FP / (FP + TN)
        g_score = (2*recall*(1-pf))/(recall + (1-pf))
        write_kfold_results(precision, recall, fscore, pf, g_score, project, model_name)
    except:
        pass
    

    return model.score(x_test, y_test)

# write kfold results to CSV file
def write_kfold_results(p_score, r_socre, f1_score, pf, g_score,project, model_name):
    with open('./data/bug_reports/results/balanced_data_'+str(model_name)+'_results.csv', 'a') as results:
            write = csv.writer(results)
            data = [p_score, r_socre, f1_score, pf, g_score,project, model_name]
            write.writerow(data)

def train_evaluate_model(model,matrix, targets, project, model_name):
    # kfold cross validation
    start_time = time.time()
    folds = KFold(n_splits=10)
    counter = 0
    for train_index, test_index in folds.split(matrix):
        counter += 1
        print(counter)
        # split data to train test
        x_train, x_test, y_train, y_test = matrix[train_index], matrix[test_index],\
                                    targets[train_index], targets[test_index]

        get_score(model, x_train, x_test, y_train, y_test, project, model_name)
        
    train_time = time.time()
    total_traning_testing_time = 'Train & tesing time: {:.2f}s'.format(train_time - start_time)
    print("Total training and validation time: " + total_traning_testing_time)


if __name__ == '__main__':
    
    projects_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    for project in projects_files:
        print("Procssing "+project)
        data = open(data_path+project).readlines()
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

        print("starts LogisticRegression algorithm ...")
        train_evaluate_model(LogisticRegression(),matrix, targets,project, 'LR')

        print("starts RandomForestClassifier algorithm ...")
        train_evaluate_model(RandomForestClassifier(),matrix, targets,project, 'RFC')

        print("starts GaussianNB algorithm ...")
        train_evaluate_model(GaussianNB(),matrix.todense(), targets,project, 'GNB')

        print("starts KNeighborsClassifier algorithm ...")
        train_evaluate_model(KNeighborsClassifier(),matrix, targets,project, 'KNN')

        print("Starts MLPClassifier algorithm ...")
        train_evaluate_model(MLPClassifier(),matrix, targets,project, 'MLP')

# in this .py we need to run the same dataset on SVC, RFC, LR, and other to check the results and compare them with fasttext results.
# Also, we need to comare the performance for each algoeithm along with the same processed data to fasttext performance.