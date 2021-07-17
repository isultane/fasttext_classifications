import time
from operator import index, mod
from os import sep
import fasttext
import numpy as np
from numpy.lib.function_base import average
import pandas as pd
import re

from scipy.sparse.sputils import matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder


class CVEsTagger(object):
 
    def split_train_test(datasetFile, testing_size=0.2):
        # split dataset into train and test, then save them
        with open(datasetFile, "r") as reader:
            data = reader.readlines()

        x_train, x_test = train_test_split(data, test_size=testing_size)

        # save training data
        with open("cve_cwe_summaries.train", 'w') as trainFile:
            trainFile.writelines(x_train)
        
        # save testing data
        with open("cve_cwe_summaries.valid", 'w') as testFile:
            testFile.writelines(x_test)


# function to extract Precision, Recall and F1
def extract_P_R_F1(N, p, r):
    precision_score = p
    recall_socre = r
    f1_score = (2*precision_score * recall_socre) / (precision_score + recall_socre)

    print("[fold N records:", N, " P: ", precision_score, "R: " ,recall_socre, "F1: ",f1_score, "]")
    return N, precision_score, recall_socre, f1_score

# this code for testing the approach of k-fold validation
def fasttext_kfold_model(df, k, lrs, epochs, dims, loss_fns, ngrams):
    start_time = time.time()
    # record results
    models = []
    train_scores = []
    val_scores = []
    # split the dataset into k folds
    kf = KFold(n_splits=k, shuffle=True)
    fold_counter = 0

    best_results = {
        "model": None,
        "aggr_metrics": {"overall": {"f1":0.0}}
    }
    # for each fold
    for train_index, test_index in kf.split(df['label'], df['text']):
        fold_counter +=1
        print("Processing fold: ", fold_counter)
        train_fold = './data/fasttext_train_fold_'+str(fold_counter)+'.txt'
        test_fold = './data/fasttext_test_fold_'+str(fold_counter)+'.txt'
        # save training subset for each fold
        df[['label', 'text']].iloc[train_index].to_csv(train_fold, index=None, header=None, sep='\t')
        
        # save tesing subset for each trained fold
        df[['label', 'text']].iloc[test_index].to_csv(test_fold, index=None, header=None, sep='\t')
        
        # tuning the configurations [lr, epoch, ngram, dim, loss]
        for lr in lrs:
            for epoch in epochs:
                for ngram in ngrams:
                    for dim in dims:
                        for loss_fn in loss_fns:

                            conf = [lr, epoch, ngram, dim, loss_fn]
                            print(conf)
                            try:
                                # fit model for this set of parameter values
                                model = fasttext.train_supervised(train_fold,
                                                                lr = lr, 
                                                                epoch = epoch, 
                                                                dim=dim,
                                                                wordNgrams = ngram,
                                                                loss = loss_fn)
                                models.append(model)
                                
                                '''
                                train_pred = [model.predict(x)[0][0].split('__')[-1] for x in df.iloc[train_index]['label']]
                                train_score = f1_score(df['label'].values[train_index].astype(str), train_pred, average='macro')
                                print('Train score:', train_score)
                                train_scores.append(train_score)

                                val_pred = [model.predict(x)[0][0].split('__')[-1] for x in df.iloc[test_index]['label']]
                                val_score = f1_score(df['label'].values[test_index].astype(str), val_pred, average='macro')
                                print('Val score:', val_score)
                                val_scores.append(val_score)
                                '''
                                metrics = extract_P_R_F1(*model.test(test_fold))
                                '''
                                if metrics["f1_score"] > best_results["aggr_metrics"]["overall"]["f1"]:
                                    model_results = {
                                        "model": model,
                                        "aggr_metrics": metrics['f_score']
                                    }
                                    best_results = model_results
                                '''
                            except Exception as e:
                                print(f"Error for fold={fold_counter} and conf {conf}: {e}")

        '''
        print('mean train scores: ', np.mean(train_scores))
        print('mean val scores: ', np.mean(val_scores))
        '''
        print("************************************ FOLD DONE ************************************")
        time.spleep(5)
  

    train_time = time.time()
    print('Train time: {:.2f}s'.format(train_time - start_time))

    return models

# read fasttext fromat into dataframe.
def read_data():
    data = open('./data/testin_fasttext_code_data.train').readlines()
    count_vect = CountVectorizer()

    labels, texts = ([], [])
    for line in data:
        label, text = line.split(' ', 1)
        text = text.strip('\n')
        labels.append(label)
        texts.append(text)

    trainDF = pd.DataFrame()
    trainDF['label'] = labels
    trainDF['text'] = texts

    matrix = count_vect.fit_transform(trainDF['text'])
    encoder = LabelEncoder()
    targets = encoder.fit_transform(trainDF['label'])

    return trainDF


df = read_data()

models = fasttext_kfold_model(df, 
                            k = 3,
                            lrs = [0.1,0.3,0.7,0.9],
                            epochs = [10],
                            dims = [50,100],
                            loss_fns = ["ns", "hs", "softmax", "ova"],
                            ngrams = [1,2,3])

#if __name__ == '__main__':
   # cve_tagger = CVEsTagger()
   # cve_tagger.fasttext_model_kfold("cve_cwe_summaries.train",10)