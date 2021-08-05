import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import matplotlib.pyplot as plt

import fasttext
import re

def predict_labels(testfile, model):
    # Return predictions
    lines = open(testfile, 'r').readlines()
    predicted_lables = []

    for line in lines:
        text = ' '.join(line.split()[2:])
        label = model.predict([re.sub('\n', ' ', text)])[0][0]
        predicted_lables.append(label)
    return predicted_lables

def parse_labels(testfile):
    #return test labeles
    lines = open(testfile, 'r').readlines()
    test_lables = []
    for line in lines:
        labele = line.split()[0]
        test_lables.append(labele)

    return test_lables

if __name__ == "__main__":
    test_labels = parse_labels('ambari.valid')

    pred_labels = predict_labels('ambari.valid', model=fasttext.load_model("model_ambari.bin"))
    
    eq = test_labels == pred_labels
    #print("Accuracy: " + str(eq.sum() / len(test_labels)))
    cm = confusion_matrix(test_labels, pred_labels)
    print(cm)
    print("Accuracy Score: " , accuracy_score(test_labels, pred_labels))
    print("Recall Score: ", recall_score(test_labels, pred_labels, average=None))
    print("Precision Score: ", precision_score(test_labels, pred_labels, average=None))


    ## Create the Confusion Matrix Display Object(cmd_obj). Note the 
    ## alphabetical sorting order of the labels.
    cmd_obj = ConfusionMatrixDisplay(cm, display_labels=['non-security', 'security'])

    cmd_obj.plot()

    cmd_obj.ax_.set(
                title='Sklearn Confusion Matrix with labels!!', 
                xlabel='Predicted labels', 
                ylabel='Actual labels')
    ## Finally, call the matplotlib show() function to display the visualization
    ## of the Confusion Matrix.
    plt.show()