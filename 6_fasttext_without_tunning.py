# Author: Sultan S. Alqahtani
# Date: 05/04/2022

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

from utilities import read_training_data
from utilities import parse_labels
from utilities import predict_labels
from utilities import calc_precision_recall_f1


from scipy.sparse.sputils import matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

bugreports_source_dataset = glob.glob(
    "./data/temp/*.txt")  # source datasets path (bug reports after extracted in fasttext format)

class fasttextModelWithoutTunning(object):
    def split_train_test(self, datasetFile, training_path, testing_path, testing_size=0.5):
        # split dataset into train and test, then save them (training 50% and testing 50%), same as it is followed by [REF]
        with open(datasetFile, "r") as reader:
            data = reader.readlines()
            x_train, x_test = train_test_split(data, test_size=testing_size)
            # save training data
            with open("./data/temp/"+training_path, 'w') as trainFile:
                trainFile.writelines(x_train)
            # save testing data
            with open("./data/temp/"+testing_path, 'w') as testFile:
                testFile.writelines(x_test)
    # this code for testing the approach of k-fold validation
    def fasttext_kfold_model(self, df, k, project_name):
        start_time = time.time()
        # record results
        models = []
        
        # split the dataset into k folds
        kf = KFold(n_splits=k, shuffle=True)
        fold_counter = 0
        # for each fold
        for train_index, test_index in kf.split(df['label'], df['text']):
            best_results = {
                "f_score": 0.0,
                "p_score": 0.0,
                "r_score": 0.0,
                "g_score":0.0,
                "pf_score":0.0,
                "kfold_counter": 0
            }
            fold_counter += 1
            print("Processing fold: ", fold_counter)
            train_fold = './data/temp/ft_k' + \
                str(fold_counter)+"_"+str(project_name)+'.train'
            test_fold = './data/temp/ft_k' + \
                str(fold_counter)+"_"+str(project_name)+'.valid'
            # save training subset for each fold
            df[['label', 'text']].iloc[train_index].to_csv(
                train_fold, index=None, header=None, sep=' ')

            # save tesing subset for each trained fold
            df[['label', 'text']].iloc[test_index].to_csv(
                test_fold, index=None, header=None, sep=' ')

            try:
                # fit model for this set of parameter values
                model = fasttext.train_supervised(input=train_fold)
                test_labels = parse_labels(test_fold)
                pred_labels = predict_labels(test_fold, model)
                
                precision_score, recall_socre, f1_score, pf, g_score, = calc_precision_recall_f1(test_labels, pred_labels)
                print("Precision: ",precision_score , " Recall: ",recall_socre," F1_score: ",f1_score, "prob. false alarm: ", pf, "g_score", g_score)
                model_results = {
                        "model": model,
                        "f_score": f1_score,
                        "p_score": precision_score,
                        "r_score": recall_socre,
                        "g_score":g_score,
                        "pf_score":pf,
                        "kfold_counter": fold_counter
                }
                best_results = model_results                
            except Exception as e:
                print(f"Error for fold={fold_counter}")
            '''
            print('mean train scores: ', np.mean(train_scores))
            print('mean val scores: ', np.mean(val_scores))
            '''
            self.write_kfold_best_results(best_results, project_name)

            # to get the best k-fold model results and save it to be used later
            best_model = best_results["model"]
            best_model.save_model("./data/temp/best_k" + str(best_results["kfold_counter"])+"_"+str(project_name)+"_model.bin")
        
           #print("best values: ", best_results["f_score"], best_results["p_score"], best_results["r_score"])
            print(
                "************************************ FOLD DONE ************************************")
        train_time = time.time()
        print('Train time: {:.2f}s'.format(train_time - start_time))

        return models

    # write kfold results to CSV file
    def write_kfold_best_results(self, kfold_results, pname):
        with open('./data/temp/kfold_best_'+str(pname)+'_results.csv', 'a') as results:
            write = csv.writer(results)
            write.writerow([kfold_results["f_score"], kfold_results["p_score"],
                            kfold_results["r_score"],kfold_results["g_score"],
                            kfold_results["pf_score"], kfold_results["kfold_counter"], pname])
        
if __name__ == '__main__':
    fasttext_model_object = fasttextModelWithoutTunning()
    print("here...1")
    '''
    1- for bug reeport project: reading bug reprts for projects Ambari, Camel, Derby, Wicket and Chromium after they pre-processed and splited into 
    .train and .valid format of fasttext.
    '''
    for project_data in bugreports_source_dataset:
        project_name = os.path.basename(project_data)
        print("processing project: " + project_name)
        print("here...2")

        # prepare .train and .valid files names
        training_file = project_name.split('.')[0]
        training_file = training_file + '.train'
        print(training_file)

        testing_file = project_name.split('.')[0]
        testing_file = testing_file + '.valid'
        print(testing_file)
        # split the data set into training and validating sets
        fasttext_model_object.split_train_test(
            project_data, training_file, testing_file)

        df = read_training_data("./data/temp/"+training_file)

        models = fasttext_model_object.fasttext_kfold_model(df, k=10, project_name=project_name.split('.')[0])