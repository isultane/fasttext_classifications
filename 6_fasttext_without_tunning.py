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

'''
We have two sources of the datasets to be fed in fasttext model:
 1- bug reports datasets
 2- vulnerabilities reports datasets
'''
bugreports_source_dataset = glob.glob(
    "./data/bug_reports/*.txt")  # source datasets path (bug reports after extracted in fasttext format)

class fasttextModel(object):
    def split_train_test(self, datasetFile, training_path, testing_path, testing_size=0.5):
        # split dataset into train and test, then save them (training 50% and testing 50%), same as it is followed by [REF]
        with open(datasetFile, "r") as reader:
            data = reader.readlines()
            x_train, x_test = train_test_split(data, test_size=testing_size)
            # save training data
            with open("./data/bug_reports/"+training_path, 'w') as trainFile:
                trainFile.writelines(x_train)
            # save testing data
            with open("./data/bug_reports/"+testing_path, 'w') as testFile:
                testFile.writelines(x_test)

    # this code for testing the approach of k-fold validation
    def fasttext_without_tunning_model(self, project_name):
        # fit model for this set of parameter values
        model = fasttext.train_supervised(train_fold)
        
        self.write_kfold_best_results(best_results, project_name)

        # to get the best k-fold model results and save it to be used later
        best_model = best_results["model"]
            best_model.save_model("./data/best_kfold_models/best_k" + str(best_results["kfold_counter"])+"_"+str(project_name)+"_model.bin")
            #print("best results: ", best_results["conf"])
            #print("best values: ", best_results["f_score"], best_results["p_score"], best_results["r_score"])
            print(
                "************************************ FOLD DONE ************************************")

        train_time = time.time()
        print('Train time: {:.2f}s'.format(train_time - start_time))

        return model

    # write kfold results to CSV file
    def write_kfold_best_results(self, kfold_results, pname):
        with open('./data/best_kfold_models/kfold_best_'+str(pname)+'_results.csv', 'a') as results:
            write = csv.writer(results)
            write.writerow([kfold_results["conf"], kfold_results["f_score"],
                            kfold_results["p_score"], kfold_results["r_score"],kfold_results["g_score"],kfold_results["pf_score"], kfold_results["kfold_counter"], pname])
# main function - start implemetnation to fit models (kfold).
if __name__ == '__main__':
    fasttext_model_object = fasttextModel()
    '''
    1- for bug reeport project: reading bug reprts for projects Ambari, Camel, Derby, Wicket and Chromium after they pre-processed and splited into 
    .train and .valid format of fasttext.
    2- for security vuln. project: reading CVEs and thier associated labels from CWEs, preprocessed the dataset, and split into train/test and create the model.
    '''
    for project_data in bugreports_source_dataset:
#    for project_data in vulnerabilities_source_dataset:
        project_name = os.path.basename(project_data)
        print("processing project: " + project_name)

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
        # read the pre-processed dataset into dataframe - TBC

        df = read_training_data("./data/bug_reports/"+training_file)
      # df = read_training_data("./data/vulnerabilities_reports/"+training_file)


        models = fasttext_model_object.fasttext_kfold_model(df,
                                                            k=10,
                                                            lrs=[
                                                                0.1, 0.3, 0.7, 0.9],
                                                            epochs=[10],
                                                            dims=[50, 100],
                                                            loss_fns=[
                                                                "ns", "hs", "softmax", "ova"],
                                                            ngrams=[1, 2, 3],
                                                            project_name=project_name.split('.')[0])