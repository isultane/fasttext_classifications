# Author: Sultan S. Alqahtani
# Date: 06/16/2021
import argparse
import numpy as np
import pandas as pd

from utilities import calc_precision_recall_f1

from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, recall_score, precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import matplotlib.pyplot as pyplot

import fasttext
import re
import glob
import os

# declaration of data sources paths
validation_dataset = glob.glob("./data/temp/*.valid")
models_files = glob.glob("./data/temp/*.bin")



def predict_probs(testfile, model):
    # Return predictions probabilities
    lines = open(testfile, 'r').readlines()
    pred_probs = []

    for line in lines:
        text = ' '.join(line.split()[2:])
        prob = model.predict([re.sub('\n', ' ', text)])[1][0]
        pred_probs.append(str(prob).replace('[','').replace(']','').replace('"','').replace('\'',''))
    return pred_probs

def predict_labels(testfile, model):
    # Return predictions
    lines = open(testfile, 'r').readlines()
    pred_label = []

    for line in lines:
        text = ' '.join(line.split()[2:])
        label = model.predict([re.sub('\n', ' ', text)])[0][0]
        pred_label.append(str(label).replace('[','').replace(']','').replace('"','').replace('\'',''))
    return pred_label

def parse_labels(testfile):
    #return test labeles
    lines = open(testfile, 'r').readlines()
    test_lables = []
    for line in lines:
        labele = line.split()[0]
        test_lables.append(labele)

    return test_lables

def conv_to_numric(actual_labels):
    numric_labels = []
    for i in range(0, len(actual_labels)):
        if actual_labels[i] == '__label__nonsec':
            numric_labels.append(int(0))
        else:
            numric_labels.append(int(1))
    return numric_labels
        
def roc_auc_calc(test_lbls, pred_lbls):

    thresholds = list(np.array(list(range(0,105,1)))/100)
    data = list(zip(test_lbls, pred_lbls))
    df = pd.DataFrame(data, columns=['actuals', 'predictions'])
    df = df.explode('actuals')

    roc_point = []

    for threshold in thresholds:
        tp=0; fp=0; fn=0; tn=0

        for index, instance in df.iterrows():
            actual = instance['actuals']
            prediction = instance['predictions']

            if prediction >= threshold:
                prediction_class = 1
            else:
                prediction_class = 0

            if actual == 1 and prediction_class == 1:
                tp = tp + 1
            elif actual == 1 and prediction_class == 0:
                fn = fn + 1
            elif actual == 0 and prediction_class == 1:
                fp = fp + 1
            elif actual == 0 and prediction_class == 0:
                tn = tn + 1
            #print(threshold, tp, fn, fp, tn)
       
        fpr = fp / (tn + fp)
        tpr = tp / (tp + fn)
       
        roc_point.append([tpr, fpr])

    return roc_point

def plot_roc(roc_point):

    print("starts here...")
    pivot = pd.DataFrame(roc_point, columns=["x", "y"])
   
    #pyplot.scatter(pivot.y, pivot.x)
    pyplot.plot(pivot.x, pivot.y, marker='.', label='chromium test') # change between x and y

    pyplot.plot([0,1])
    pyplot.xlabel("false postive rate")
    pyplot.ylabel("true positive rate")
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()

    # AUC score
    auc = round(abs(np.trapz(pivot.x, pivot.y)),4)

    print("AUC score: ", auc)

if __name__ == "__main__":
    print("starts here ...")
    # define list to plot ROC for all projects
    roc_point = []
    # read *.valid data to extract test labels
    for (fl, ml) in zip(validation_dataset, models_files):
        file_name = os.path.basename(fl)
        print("Reading file: " + file_name)
        # get list of labeles to be tested 
        test_labels = parse_labels(fl)
        # convert the labels inro nummric - security is ONE and non-security is ZERO
        test_y = conv_to_numric(test_labels)

        # get list of predicted labels - security or non security
        pred_labels = predict_labels(fl, model=fasttext.load_model(ml))
        # get list of predicted labels probabilities
        pred_probs = predict_probs(fl, model=fasttext.load_model(ml))
    
        print(calc_precision_recall_f1(test_labels, pred_labels))
        lr_probs = np.array(pred_probs, dtype=float)

        roc_point.append(roc_auc_calc(test_y, lr_probs))
    
    for pl in roc_point:
        # hardcoded for calcualting ROC and AUC
        roc_point = roc_auc_calc(test_y, lr_probs)
        # plot ROC curve
        plot_roc(pl)

    '''
    # plot precision and recall curve
    precisio_recall_plot(test_y, pred_probs)
    
    # keep probabilities for the positive outcome only 
    #lr_probs = lr_probs[:, 1]
    
    # calculate scores
    ns_auc = roc_auc_score(test_y, ns_probs)
    lr_auc = roc_auc_score(test_y, lr_probs)

    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('fasttext: ROC AUC=%.3f' % (lr_auc))

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(test_y, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(test_y, lr_probs)
    # plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='fasttext')
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()
    
    
    
    
    

    # solution inspired from answer on SO: https://stackoverflow.com/questions/45713695/fasttext-precision-and-recall-trade-off/65757511?noredirect=1#comment121322970_65757511
    # the plot still not working
    auc = roc_auc_score(test_y, np.array(pred_probs, dtype=float))
    print('ROC AUC=%.3f' % (auc))
    # using existing python library to calculat P and R
    cm = confusion_matrix(test_labels, pred_labels)
    print(cm)
    print("Accuracy Score: " , accuracy_score(test_labels, pred_labels))
    print("Recall Score: ", recall_score(test_labels, pred_labels, average='micro'))
    print("Precision Score: ", precision_score(test_labels, pred_labels, average='micro'))

    # calculate roc curve
    fpr, tpr, _ = roc_curve(test_y, np.array(pred_probs, dtype=float))

    # no skill plot
    pyplot.plot([0,1])
    # plot the roc curve for the model
    pyplot.plot(fpr, tpr, marker='.', label='ROC curve')
    # axis labels
    pyplot.xlabel('False Positive Rate (sensitivity)')
    pyplot.ylabel('True Positive Rate (specificity)')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()   
'''