# Author: Sultan S. Alqahtani
# Date: 06/16/2021 
import csv
import nltk
import pandas as pd
import re
import string
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
#nltk.download('stopwords') # it is used at the first time to download stopwords list
#nltk.download('punkt')

def get_keywords(row):     
    # split into tokens by white space
    # tokens = [x.strip() for x in row.split()]
      tokens = word_tokenize(row)
    # convert to lower cases
      tokens = [w.lower() for w in tokens]
    # prepare regex for char filtering
      re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
      tokens = [re_punc.sub(' ', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
      tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
      stop_words = set(stopwords.words('english'))
      tokens = [w for w in tokens if w not in stop_words]
    # stemming of words
    # porter = PorterStemmer()
    # stemmed = [porter.stem(word) for word in tokens]
    # filter out short tokens
    #  tokens = [word for word in tokens if len(word) > 2]
      return ' '.join(tokens).strip()


# read fasttext fromat into dataframe.
def read_training_data(training_dataset):
    data = open(training_dataset).readlines()
    labels, texts = ([], [])
    for line in data:
        label, text = line.split(' ', 1)
        text = text.strip('\n')
        text = get_keywords(text) # this text might need more pre-processing, tokanizing, stemming, keywords, etc. 
        labels.append(label)
        texts.append(text)

    trainDF = pd.DataFrame()
    trainDF['label'] = labels
    trainDF['text'] = texts

    ''' # if you need to treat the dataset in text fromat to be fit in dataframe
    count_vect = CountVectorizer()
    matrix = count_vect.fit_transform(trainDF['text'])
    encoder = LabelEncoder()
    targets = encoder.fit_transform(trainDF['label'])
    '''
    return trainDF
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
def calc_precision_recall_f1(y_true, y_pred):
    
    # Convert predictions to series with index matching y_true
   # y_pred = pd.Series(y_pred, index=y_true.index)
    
    # Instantiate counters
    TP = 0
    FP = 0
    FN = 0

    # Determine whether each prediction is TP, FP, TN, or FN
    for i in range(len(y_true)): 
        #print("Predicted label:",y_pred[i], "True label: "+y_true[i])
        if (y_true[i]==y_pred[i]=='__label__sec')or (y_true[i]==y_pred[i]=='__label__nonsec'):
           TP += 1
        elif (y_pred[i]=='__label__nonsec') and (y_true[i]!=y_pred[i]):
           FP += 1
        elif (y_pred[i]=='__label__sec') and (y_true[i]!=y_pred[i]):
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
