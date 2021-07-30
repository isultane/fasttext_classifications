import fasttext
import re
import pandas as pd

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder



def split_df(data):
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
    trainX, testX, trainy, testy = train_test_split(
        matrix, targets, test_size=0.2)

    return trainX, testX, trainy, testy
def extract_P_R_F1(self, N, p, r):
        precision_score = p
        recall_socre = r
        f1_score = (2*precision_score * recall_socre) / \
            (precision_score + recall_socre)

        print("[ N records:", N, " P: ", precision_score,
              "R: ", recall_socre, "F1: ", f1_score, "]")
 

test_sentences = open('ambari.valid').readlines()
model = fasttext.load_model("model_ambari.bin")

print(model.test_label('ambari.valid'))
'''

trainX, testX, trainy, testy = split_df(test_sentences)


# label the data
labels, probabilities = model.predict([re.sub('\n', ' ', sentence) 
                                                     for sentence in test_sentences])
auc = roc_auc_score(testy, probabilities)
print('ROC AUC=%.3f' % (auc))

# convert fasttext multilabel results to a binary classifier (probability of TRUE)
labels = list(map(lambda x: x == ['__label__nonsec-report'] or x == ['__label__sec-report'], labels))
probabilities = [probability[0] if label else (1-probability[0]) 
                 for label, probability in zip(labels, probabilities)]

auc = roc_auc_score(testy, probabilities)
print('ROC AUC=%.3f' % (auc))
'''