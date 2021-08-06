import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import matplotlib.pyplot as plt

import fasttext
import re

def predict_probs(testfile, model):
    # Return predictions probabilities
    lines = open(testfile, 'r').readlines()
    pred_probs = []

    for line in lines:
        text = ' '.join(line.split()[2:])
        prob = model.predict([re.sub('\n', ' ', text)])[1][0]
        pred_probs.append(prob)
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

# Function to calculate Precision and Recall.
# Source: https://medium.com/@douglaspsteen/precision-recall-curves-d32e5b290248
def calc_precision_recall(y_true, y_pred):
    
    # Convert predictions to series with index matching y_true
   # y_pred = pd.Series(y_pred, index=y_true.index)
    
    # Instantiate counters
    TP = 0
    FP = 0
    FN = 0

    # Determine whether each prediction is TP, FP, TN, or FN
    for i in range(len(y_true)): 
        print("Predicted label:",y_pred[i], "True label: "+y_true[i])
        if (y_true[i]==y_pred[i]=='__label__sec-report')or (y_true[i]==y_pred[i]=='__label__nonsec-report'):
           TP += 1
        if (y_pred[i]=='__label__nonsec-report') and (y_true[i]!=y_pred[i]):
           FP += 1
        if (y_pred[i]=='__label__sec-report') and (y_true[i]!=y_pred[i]):
           FN += 1
    
    
    # Calculate true positive rate and false positive rate
    # Use try-except statements to avoid problem of dividing by 0
    try:
        precision = TP / (TP + FP)
    except:
        precision = 1
    
    try:
        recall = TP / (TP + FN)
    except:
        recall = 1
    
    f1_score = (2*precision * recall) / \
            (precision + recall)

    return precision, recall, f1_score

if __name__ == "__main__":
    test_labels = parse_labels('cooking.valid')

    pred_labels = predict_labels('cooking.valid', model=fasttext.load_model("model_cooking.bin"))
    #pred_probs = predict_probs('ambari.valid', model=fasttext.load_model("model_ambari.bin"))

    # hard coded P and R
    print(calc_precision_recall(test_labels, pred_labels))
    '''
    # using existing python library to calculat P and R
    cm = confusion_matrix(test_labels, pred_labels)
    print(cm)
    print("Accuracy Score: " , accuracy_score(test_labels, pred_labels))
    print("Recall Score: ", recall_score(test_labels, pred_labels, average=None))
    print("Precision Score: ", precision_score(test_labels, pred_labels, average=None))

    
    # solution exlained here but I'm not confident 100%. https://stackoverflow.com/questions/45713695/fasttext-precision-and-recall-trade-off/65757511?noredirect=1#comment121322970_65757511
    # the plot still not working
    auc = roc_auc_score(test_labels, pred_probs)
    print('ROC AUC=%.3f' % (auc))
    '''