# Author: Sultan S. Alqahtani
# Date: 06/16/2021
import time
from operator import index, mod
import os
import fasttext
import numpy as np
from numpy.lib.function_base import average
import pandas as pd
import re
import csv
import glob

from utilities import get_keywords
from utilities import read_training_data


from scipy.sparse.sputils import matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

'''
We have two sources of the datasets to be fed in fasttext model:
 1- bug reports datasets
 2- vulnerabilities reports datasets
'''
source_dataset = glob.glob("./data/bug_reports/*.txt") # source datasets path (bug reports after extracted in fasttext format)
#training_dataset = glob.glob("./data/bug_reports/*.train")
#testing_dataset = "./data/cve_cwe_summaries.valid"


class fasttextModel(object):
    def split_train_test(self, datasetFile, training_path, testing_path, testing_size=0.2):
        # split dataset into train and test, then save them
        with open(datasetFile, "r") as reader:
            data = reader.readlines()
            x_train, x_test = train_test_split(data, test_size=testing_size)
            # save training data
            with open("./data/bug_reports/"+training_path, 'w') as trainFile:
                trainFile.writelines(x_train)
            # save testing data
            with open("./data/bug_reports/"+testing_path, 'w') as testFile:
                testFile.writelines(x_test)
    # function to extract Precision, Recall and F1

    def extract_P_R_F1(self, N, p, r):
        precision_score = p
        recall_socre = r
        f1_score = (2*precision_score * recall_socre) / \
            (precision_score + recall_socre)

        print("[ N records:", N, " P: ", precision_score,
              "R: ", recall_socre, "F1: ", f1_score, "]")
        return N, precision_score, recall_socre, f1_score

    # this code for testing the approach of k-fold validation
    def fasttext_kfold_model(self, df, k, lrs, epochs, dims, loss_fns, ngrams, project_name):
        start_time = time.time()
        # record results
        models = []
        #train_scores = []
        #val_scores = []
        # split the dataset into k folds
        kf = KFold(n_splits=k, shuffle=True)
        fold_counter = 0
        best_results = {
            "conf": None,
            "model": None,
            "f_score": 0.0,
            "p_score": 0.0,
            "r_score": 0.0,
            "kfold_counter": 0
        }
        # for each fold
        for train_index, test_index in kf.split(df['label'], df['text']):
            fold_counter += 1
            print("Processing fold: ", fold_counter)
            train_fold = './data/kfold_train_test_data/fasttext_fold_' + \
                str(fold_counter)+str(project_name)+'.train'
            test_fold = './data/kfold_train_test_data/fasttext_fold_' + \
                str(fold_counter)+str(project_name)+'.valid'
            # save training subset for each fold
            df[['label', 'text']].iloc[train_index].to_csv(
                train_fold, index=None, header=None, sep=' ')

            # save tesing subset for each trained fold
            df[['label', 'text']].iloc[test_index].to_csv(
                test_fold, index=None, header=None, sep=' ')

            # tuning the configurations [lr, epoch, ngram, dim, loss]
            for lr in lrs:
                for epoch in epochs:
                    for dim in dims:
                        for loos_fn in loss_fns:
                            for ngram in ngrams:

                                conf = [lr, epoch, dim, loos_fn, ngram]
                                print(conf)
                                try:
                                    # fit model for this set of parameter values
                                    model = fasttext.train_supervised(train_fold,
                                                                  lr=lr,
                                                                  epoch=epoch,
                                                                  dim=dim,
                                                                  loss=loos_fn,
                                                                  wordNgrams=ngram
                                                                  )
                                    # models.append(model) # <-- this is a bug!

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
                                    N, precision_score, recall_socre, f1_score = self.extract_P_R_F1(
                                    *model.test(test_fold))

                                    if f1_score > best_results["f_score"]:
                                        model_results = {
                                            "conf": conf,
                                            "model": model,
                                            "f_score": f1_score,
                                            "p_score": precision_score,
                                            "r_score": recall_socre,
                                            "kfold_counter": fold_counter
                                        }

                                        best_results = model_results

                                except Exception as e:
                                    print(f"Error for fold={fold_counter} and conf {conf}: {e}")

            '''
            print('mean train scores: ', np.mean(train_scores))
            print('mean val scores: ', np.mean(val_scores))
            '''
            self.write_kfold_best_results(best_results, project_name)

            # to get the best k-fold model results and save it to be used later
            best_model = best_results["model"]
            best_model.save_model("./data/kfold_train_test_data/best_kfold" +
                              str(best_results["kfold_counter"])+str(project_name)+"_model.bin")
            #print("best results: ", best_results["conf"])
            #print("best values: ", best_results["f_score"], best_results["p_score"], best_results["r_score"])
            print("************************************ FOLD DONE ************************************")

        train_time = time.time()
        print('Train time: {:.2f}s'.format(train_time - start_time))

        return models

    # write kfold results to CSV file
    def write_kfold_best_results(self, kfold_results, pname):
        with open('./data/kfold_train_test_data/kfold_best_'+str(pname)+'_results.csv', 'a') as results:
            write = csv.writer(results)
            write.writerow([kfold_results["conf"], kfold_results["f_score"],
                       kfold_results["p_score"], kfold_results["r_score"], kfold_results["kfold_counter"], pname])

if __name__ == '__main__':
    fasttext_model_object = fasttextModel()
    '''
    reading bug reprts for projects Ambari, Camel, Derby, Wicket and Chromium after they pre-processed and splited into 
    .train and .valid format of fasttext.
    '''
    for project_data in source_dataset:
        project_name = os.path.basename(project_data)
        print("processing project: " + project_name)

        training_file = project_name.split('.')[0]
        training_file = training_file + '.train'
        print(training_file)

        testing_file = project_name.split('.')[0]
        testing_file = testing_file + '.valid'
        print(testing_file)

        # split the data set into training and validating sets
        fasttext_model_object.split_train_test(project_data, training_file, testing_file)
        # read the pre-processed dataset into dataframe - TBC
        
        df = read_training_data("./data/bug_reports/"+training_file)

        models = fasttext_model_object.fasttext_kfold_model(df,
                              k=10,
                              lrs=[0.1, 0.3, 0.7, 0.9],
                              epochs=[10],
                              dims=[50, 100],
                              loss_fns=["ns", "hs", "softmax", "ova"],
                              ngrams=[1, 2, 3],
                              project_name=project_name.split('.')[0])

