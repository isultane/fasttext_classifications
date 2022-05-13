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
from utilities import calc_accurecy


from scipy.sparse.sputils import matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
#from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score


bugreports_source_dataset = glob.glob(
    "./data/bug_reports/*.txt")  # source datasets path (bug reports after extracted in fasttext format)

class fasttextModelWithoutTunning(object):
    def split_train_test(self, datasetFile, training_path, testing_path, testing_size=0.5):
        # split dataset into train and test, then save them (training 50% and testing 50%), same as it is followed by [REF]
        with open(datasetFile, "r") as reader:
            data = reader.readlines()
            x_train, x_test = train_test_split(data, test_size=testing_size)
            # save training data
            with open("./data/bug_reports/results/"+training_path, 'w') as trainFile:
                trainFile.writelines(x_train)
            # save testing data
            with open("./data/bug_reports/results/"+testing_path, 'w') as testFile:
                testFile.writelines(x_test)
    # this code for testing the approach of k-fold validation
    def fasttext_kfold_model(self, df, k, project_name):
        start_time = time.time()
        print("Processing project" + project_name)
        # record results
        models = []

        train_fold = ""
        test_fold = ""
        
        # split the dataset into k folds
        kf = KFold(n_splits=k, shuffle=True)
        fold_counter = 0
        # for each fold
        for train_index, test_index in kf.split(df['label'], df['text']):
            fold_counter += 1
            #print("Processing fold: ", fold_counter)
            train_fold = './data/bug_reports/results/ft_k' + \
                str(fold_counter)+"_"+str(project_name)+'.train'
            test_fold = './data/bug_reports/results/ft_k' + \
                str(fold_counter)+"_"+str(project_name)+'.valid'
            # save training subset for each fold
            df[['label', 'text']].iloc[train_index].to_csv(
                train_fold, index=None, header=None, sep=' ')

            # save tesing subset for each trained fold
            df[['label', 'text']].iloc[test_index].to_csv(
                test_fold, index=None, header=None, sep=' ')

            
            # fit model for this set of parameter values
            model = fasttext.train_supervised(train_fold, epoch=40, lr=1.0)
            test_labels = parse_labels(test_fold)
            pred_labels = predict_labels(test_fold, model)
            precision_score, recall_socre, f1_score, pf, g_score = calc_accurecy(test_labels, pred_labels)
            #print("Precision: ",precision_score , " Recall: ",recall_socre," F1_score: ",f1_score, "prob. false alarm: ", pf, "g_score", g_score)

            '''
            print('mean train scores: ', np.mean(train_scores))
            print('mean val scores: ', np.mean(val_scores))
            '''
            self.write_kfold_results(precision_score, recall_socre, f1_score, pf, g_score, fold_counter,project_name)
           
            # to get the best k-fold model results and save it to be used later
            model.save_model("./data/bug_reports/results/k" + str(fold_counter)+"_"+str(project_name)+"_model.bin")
        
           #print("best values: ", best_results["f_score"], best_results["p_score"], best_results["r_score"])
           #print("************************************ FOLD DONE ************************************")
        train_time = time.time()
        #print('Train & tesing time: {:.2f}s'.format(train_time - start_time))
        total_traning_testing_time = 'Train & tesing time: {:.2f}s'.format(train_time - start_time)
        self.write_training_time(total_traning_testing_time, project_name)

        print("Deleting tmp files")
        if os.path.exists(train_fold) and os.path.exists(test_fold):
            os.remove(train_fold)
            os.remove(test_fold)
            
        return models

    # write kfold results to CSV file
    def write_kfold_results(self, p_score, r_socre, f1_score, pf, g_score,fold_counter, pname):
        with open('./data/bug_reports/results/kfold_'+str(pname)+'_results.csv', 'a') as results:
            write = csv.writer(results)
            data = [p_score, r_socre, f1_score, pf, g_score,fold_counter, pname]
            write.writerow(data)
    
    # write kfold results to CSV file
    def write_training_time(self, training_time, pname):
        with open('./data/bug_reports/results/total_training_time_results.csv', 'a') as tresults:
            write = csv.writer(tresults)
            data = [training_time, pname]
            write.writerow(data)
        
if __name__ == '__main__':
    fasttext_model_object = fasttextModelWithoutTunning()
    '''
    1- for bug reeport project: reading bug reprts for projects Ambari, Camel, Derby, Wicket and Chromium after they pre-processed and splited into 
    .train and .valid format of fasttext.
    '''
    for project_data in bugreports_source_dataset:
        project_name = os.path.basename(project_data)
        #print("processing project: " + project_name)

        # prepare .train and .valid files names
        training_file = project_name.split('.')[0]
        training_file = training_file + '.train'
        #print(training_file)

        testing_file = project_name.split('.')[0]
        testing_file = testing_file + '.valid'
        #print(testing_file)
        # split the data set into training and validating sets
        fasttext_model_object.split_train_test(
            project_data, training_file, testing_file)

        df = read_training_data("./data/bug_reports/results/"+training_file)

        models = fasttext_model_object.fasttext_kfold_model(df, k=10, project_name=project_name.split('.')[0])