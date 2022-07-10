# Author: Sultan S. Alqahtani
# Date: 05/04/2022

from os import listdir
from os.path import isfile, join
from utilities import parse_labels, predict_labels, calc_accurecy
from numpy import array

import os
import csv
import fasttext

data_path = './data/temp/'

def cross_validate_project(test_project, training_list):
    print("Cross validate: ", test_project)
    print("traning projects: ", training_list)

    for  train in training_list:
        model = fasttext.train_supervised(data_path+train, epoch=40, lr=0.5, loss='ova')
        test_labels = parse_labels(data_path+test_project)
        pred_labels = predict_labels(data_path+test_project, model)
        precision_score, recall_socre, f1_score, pf, g_score = calc_accurecy(test_labels, pred_labels)
        #print("Precision: ",precision_score , " Recall: ",recall_socre," F1_score: ",f1_score, "prob. false alarm: ", pf, "g_score", g_score)
        write_kfold_results(precision_score, recall_socre, f1_score, pf, g_score, train,test_project)
        model.save_model("./data/bug_reports/results/" + str(train)+"_vs_"+str(test_project)+"_model.bin")

# write kfold results to CSV file
def write_kfold_results(p_score, r_socre, f1_score, pf, g_score,fold_counter, pname):
    with open('./data/bug_reports/results/crossvalidated_'+str(pname)+'_results.csv', 'a') as results:
            write = csv.writer(results)
            data = [p_score, r_socre, f1_score, pf, g_score,fold_counter, pname]
            write.writerow(data)


projects_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
tested_projects = []
training_list = []
for project in projects_files:
    if project not in tested_projects:
        tested_projects.append(project)
    training_list = [i for i in projects_files if i not in tested_projects]
    
    cross_validate_project(project, training_list)
    training_list.clear()
    tested_projects.clear()
