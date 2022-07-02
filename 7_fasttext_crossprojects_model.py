# Author: Sultan S. Alqahtani
# Date: 05/04/2022

from os import listdir
from os.path import isfile, join
from utilities import parse_labels, predict_labels, calc_accurecy

import os
import fasttext

tested_projects = []

onlyfiles = [f for f in listdir("./data/bug_reports/") if isfile(join("./data/bug_reports/", f))]

for f in onlyfiles:
    if f not in tested_projects:
        tested_projects.append(f)
        training_f = os.path.abspath("./data/bug_reports/"+f)
        model = fasttext.train_supervised(training_f, epoch=40, lr=1.0)

'''
 test_labels = parse_labels(test_fold)
    pred_labels = predict_labels(test_fold, model)
    precision_score, recall_socre, f1_score, pf, g_score, = calc_accurecy(test_labels, pred_labels)
    print("Precision: ",precision_score , " Recall: ",recall_socre," F1_score: ",f1_score, "prob. false alarm: ", pf, "g_score", g_score)

'''
   


