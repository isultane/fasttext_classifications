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

# Returns a list of common english terms (words)
def initialize_words():
    content = None
    with open('/home/sultan/Downloads/wordlist.txt') as f: # A file containing common english words
        content = f.readlines()
    return [word.rstrip('\n') for word in content]

def parse_sentence(sentence, wordlist):
    new_sentence = "" # output  
    tokens = re.findall(r'\w+\b', sentence)
 #   tokens = word_tokenize(sentence)
    # convert to lower cases
    tokens = [w.lower() for w in tokens]
    # prepare regex for char filtering
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub(' ', w) for w in tokens]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    for term in tokens:
        new_sentence += parse_terms(term, wordlist)
        new_sentence += " "

    return " ".join(new_sentence.split())
# source : https://stackoverflow.com/questions/20516100/term-split-by-hashtag-of-multiple-words
def parse_terms(term, wordlist):
    words = []
    word = find_word(term, wordlist)  
    while word != None and len(word) > 2:
        words.append(word)            
        if len(term) == len(word): # Special case for when eating rest of word
            break
        term = term[len(word):]
        word = find_word(term, wordlist)
    return " ".join(words)
# source: https://stackoverflow.com/questions/20516100/term-split-by-hashtag-of-multiple-words
def find_word(token, wordlist):
    i = len(token) + 1
    while i > 1:
        i -= 1
        if token[:i] in wordlist:
            return token[:i]
    return None 

# read fasttext fromat into dataframe.
def read_training_data(training_dataset):
    data = open(training_dataset).readlines()
    labels, texts = ([], [])
    for line in data:
        label, text = line.split(' ', 1)
        text = text.strip('\n')
   #     text = get_keywords(text) # this text might need more pre-processing, tokanizing, stemming, keywords, etc. 
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
    print(pred_label)
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
    TP = 0; FP = 0; FN = 0; TN = 0

    # Determine whether each prediction is TP, FP, TN, or FN
    for i in range(len(y_true)): 
        #print("Predicted label:",y_pred[i], "True label: "+y_true[i])
        if y_true[i]=='__label__sec' and y_pred[i]=='__label__sec':
           TP += 1
        elif y_true[i]=='__label__sec' and y_pred[i]=='__label__nonsec':
           FN +=1
        elif y_true[i]=='__label__nonsec' and y_pred[i]=='__label__sec':
           FP += 1
        elif y_true[i]=='__label__nonsec' and y_pred[i]=='__label__nonsec':
           TN += 1
    
    # Calculate true positive rate and false positive rate
    # Use if-else statements to avoid problem of dividing by 0. For this speial case, we have defined
    # if the TP, FP, and FN are all Zero, the precision, recall and F1 are One. This migh occure in case
    # in which the kfold contains doc without __label__sec labels and the model correctly returns __label__nonsec (TN).
    # Reference: https://stats.stackexchange.com/questions/8025/what-are-correct-values-for-precision-and-recall-when-the-denominators-equal-0
    if TP == 0 and FP == 0:
      precision = 1
    else:
      precision = TP / (TP + FP) 

    if TP == 0 and FN == 0:
      recall = 1
    else:
      recall = TP / (TP + FN)
    
    # We report probability of false alarm (pf) as follow:
    if FP == 0 and TN == 0:
        pf = 1
    else:
        pf = FP / (FP + TN)
    
    try:
      f1_score = (2*precision * recall) / (precision + recall)
      g_score = (2*recall*(1-pf))/(recall + (1-pf))
    except:
      if precision == 0 or recall == 0:
        f1_score = 0
        g_score = 0
      else:
        f1_score = 1
        g_score = 1
    
    
    return precision, recall, f1_score, pf, g_score
