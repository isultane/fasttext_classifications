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
    "./data/bug_reports/*.txt")  # source datasets path (bug reports after extracted in fasttext format)

class fasttextModelWithoutTunning(object):
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

    def fasttext_model(self, project_name):
        
if __name__ == '__main__':
    fasttext_model_object = fasttextModel()
    '''
    1- for bug reeport project: reading bug reprts for projects Ambari, Camel, Derby, Wicket and Chromium after they pre-processed and splited into 
    .train and .valid format of fasttext.
    '''
    for project_data in bugreports_source_dataset:
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

        df = read_training_data("./data/bug_reports/"+training_file)

        models = fasttext_model_object.fasttext_kfold_model(df, project_name=project_name.split('.')[0])