# The surce code originally modified from https://realpython.com/
# Author: Sultan S. Alqahtani
# Date: 06/16/2021 
import csv
import os
import glob

from utilities import parse_sentence
from utilities import initialize_words


# declaration of data sources paths
filenames = glob.glob("/home/sultan/Downloads/sbrbench-master/Clean_sbr_datasets/*.csv")

# saving the pre-processed bug reports
save_path = './data/bug_reports/'

# reading bug reports from csv files.
# the fuction is preprocessing projects Ambari, Camel, Derby, and Wicket into fasttext format
def readCSVfiles(file_):
  # preparing the output file path + name
  outfileBasename = os.path.basename(file_)
  outfile = os.path.splitext(outfileBasename)[0]
  outfile = os.path.join(save_path, outfile+".txt")   

  # open the output file for writing
  file_1 = open(outfile, "w")
  
  wordlist = initialize_words()

  with open(file_, 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
        #    outfile.write(f'labal,doc\n')
            line_count += 1
        else:
            if row[26] == '1':
                row[26] = '__label__sec'
            else:
                row[26] = '__label__nonsec'
            # pre-processing the input file (tokanizing, stop words, etc.)
            row_preprocessed = parse_sentence(row[13] + row[14], wordlist)

            # wrting to the output file
            file_1.write(f'{row[26]} {row_preprocessed} \n') 

            line_count += 1
  # closing the output file 
  file_1.close()
  print(f'Processed {line_count} lines.')

# reading chrom bug reports and prepare the data into fasttext format
def readChromiumFile(file_):
  # preparing the output file path + name
  outfileBasename = os.path.basename(file_)
  outfile = os.path.splitext(outfileBasename)[0]
  outfile = os.path.join(save_path, outfile+".txt")   

  # open the output file for writing
  file_1 = open(outfile, "w")

  wordlist = initialize_words()

  with open(file_, 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
        #    outfile.write(f'labal,doc\n')
            line_count += 1
        else:
            if row[3] == '1':
                row[3] = '__label__sec'
            else:
                row[3] = '__label__nonsec'
            # pre-processing the input file (tokanizing, stop words, etc.)
            row_preprocessed = parse_sentence(row[2], wordlist)

            # wrting to the output file
            file_1.write(f'{row[3]} {row_preprocessed} \n') 

            line_count += 1
  # closing the output file 
  file_1.close()
  print(f'Processed {line_count} lines.')

# start read bug reports data and convert it into fasttext format
for f in filenames:
  file_name = os.path.basename(f)
  print("Reading file: " + file_name)
  if file_name.startswith("Chromium"):
    readChromiumFile(f)
  else:
    readCSVfiles(f)
