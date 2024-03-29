'''
# Author: Sultan S. Alqahtani
# Date: 07/28/2021 
# It is modified version from original code https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
'''
# roc curve and auc
import pandas as pd
import fasttext
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc

from matplotlib import pyplot
# generate 2 class dataset
data = open('./data/temp/Chromium.valid').readlines()

count_vect = CountVectorizer()

print('Loading data ...')
labels, texts = ([], [])
for line in data:
    label, text = line.split(' ', 1)
    labels.append(label)
    texts.append(text)

trainDF = pd.DataFrame()
trainDF['label'] = labels
trainDF['text'] = texts

# to fit the text in the dataframe
# You have to do some encoding before using fit. As it known fit() does not accept Strings.
count_vect = CountVectorizer()
matrix = count_vect.fit_transform(trainDF['text'])
encoder = LabelEncoder()
targets = encoder.fit_transform(trainDF['label'])

# split into train/test sets
trainX, testX, trainy, testy = train_test_split(matrix, targets, test_size=0.2, random_state=2)

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(testy))]
# load model
#model = fasttext.load_model("model_ambari.bin")
model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(trainX, trainy)

# ******************* ROC Curves and AUC in Python ************** #
# predict probabilities
lr_probs = model.predict_proba(testX)
print(lr_probs)
'''
for line in trainDF['text']:
    line = line.replace("\n", " ")
    pred_label = model.predict(line, k=-1,threshold=0.5)[0][0]
    lr_probs.append(pred_label)
'''
# keep probabilities for the positive outcome only 
lr_probs = lr_probs[:, 1]
print(lr_probs)
# calculate scores
ns_auc = roc_auc_score(testy, ns_probs)
lr_auc = roc_auc_score(testy, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic Regression: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic Regression')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

'''
# ******************* Precision-Recall Curves in Python ************** #
# predict probabilities
lr_probs = model.predict_proba(testX)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# predict class values
yhat = model.predict(testX)
lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
# summarize scores
print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(testy[testy==1]) / len(testy)
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
'''