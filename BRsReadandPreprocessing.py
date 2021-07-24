# The surce code originally modified from https://realpython.com/
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
import os
import glob
filenames = glob.glob("/home/sultan/Downloads/sbrbench-master/Clean_sbr_datasets/*.csv")
chromiumFilenames = glob.glob("/home/sultan/Downloads/sbrbench-master/Clean_sbr_datasets/Chromium_dataset/*.csv")

save_path = '/home/sultan/BRClassifications/'

#outfile = open("Chromium_2_output_preprocessed.txt", "w")

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



def readCSVfiles(file_):
  # preparing the output file path + name
  outfileBasename = os.path.basename(file_)
  outfile = os.path.splitext(outfileBasename)[0]
  outfile = os.path.join(save_path, outfile+"_Nostem_output_.txt")   

  # open the output file for writing
  file_1 = open(outfile, "w")

  with open(file_, 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
        #    outfile.write(f'labal,doc\n')
            line_count += 1
        else:
            if row[26] == '1':
                row[26] = '__label__sec-report'
            else:
                row[26] = '__label__nonsec-report'
            # pre-processing the input file (tokanizing, stop words, etc.)
            row_preprocessed = get_keywords(row[13] + row[14])

            # wrting to the output file
            file_1.write(f'{row[26]} {row_preprocessed} \n') 

            line_count += 1
  # closing the output file 
  file_1.close()
  print(f'Processed {line_count} lines.')

def readChromiumFile(file_):
  # preparing the output file path + name
  outfileBasename = os.path.basename(file_)
  outfile = os.path.splitext(outfileBasename)[0]
  outfile = os.path.join(save_path, outfile+"_dataset_.txt")   

  # open the output file for writing
  file_1 = open(outfile, "w")

  with open(file_, 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
        #    outfile.write(f'labal,doc\n')
            line_count += 1
        else:
            if row[3] == '1':
                row[3] = '__label__sec-report'
            else:
                row[3] = '__label__nonsec-report'
            # pre-processing the input file (tokanizing, stop words, etc.)
            row_preprocessed = get_keywords(row[2])

            # wrting to the output file
            file_1.write(f'{row[3]} {row_preprocessed} \n') 

            line_count += 1
  # closing the output file 
  file_1.close()
  print(f'Processed {line_count} lines.')

# preprocess projects Ambari, Camel, Derby, and Wicket
for f in filenames:
  print("Reading file: " + os.path.splitext(f)[0])
  readCSVfiles(f)


'''
# preprocess project Chromium

for f in chromiumFilenames:
  print("Reading file: " + os.path.splitext(f)[0])
  readChromiumFile(f)
'''