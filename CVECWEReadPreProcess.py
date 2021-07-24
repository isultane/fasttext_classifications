import fasttext
import re
import jieba
import pandas as pd
import random
import os
import csv
from collections import defaultdict
from types import MethodType, FunctionType

class TransformData(object):
    # convert CVE CWE-tags from csv file to fasttext format
    def to_fasttextFormat(self, inPath, outPath, index=False):
        results = open(outPath, "w+")
        with open  (inPath) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter= ',')
            line_count = 0
            for row in csv_reader:
                if not row[1]:
                    row[1] = '__label__NA'
                    text = ''.join(row[2:])
                    text = text.replace(',', ' ')
                    results.write(f'{row[1]} {text}\n')
                    line_count +=1
                else:
                    row[1] = '__label__'+ row[1]
                    text = ''.join(row[2:])
                    text = text.replace(',', ' ')
                    results.write(f'{row[1]} {text}\n')
                    line_count +=1

            results.close
            print(f'Processed {line_count} lines.')

    #clean the final dataset to make sure it is on fasttext format and the unlabeled doc to be labeled
    def check_fix_dataset_format(self, inPath, outPath, index=False):
        print("Strats processing fasttext dataset to check incorrect labeld docs")
        # stats varibles for counting the statistics (corrects and incorrect lines 'docs')
        correctLabeledDoc_count = 0
        incorrectLabeledDoc_count = 0
        line_count = 0
        unknown_docs = 0
        # open unprocessed cve_cwe_summaries for fixing
        cve_cwe_summary_unccheked = open(inPath, "r")
        # write reviwed cve_cwe_summaries after fixing
        cve_cwe_summary_checked = open(outPath, "w")

        for line in cve_cwe_summary_unccheked:
            if line != "\n":
                line_count += 1
            check_label = line.split()[0]
            if check_label.find("__label__") != -1:
                correctLabeledDoc_count += 1
                cve_cwe_summary_checked.write(f'{line}')
            elif((not line.split()[1]) or (line.split()[1].find("Unspecified") != -1) or (line.split()[1].find("Unrestricted") != -1)):
                incorrectLabeledDoc_count += 1
                correct_lable = '__label__NA'
                text =  ' '.join(line.split()[2:])
                text = text.replace(',', ' ')
                cve_cwe_summary_checked.write(f'{correct_lable} {text}\n')
               #print(f'{correct_lable} {text}\n')
               # print(line.split()[1])
            elif line.split()[1].find("CWE-") != -1:
              #  print("Here"+ line.split()[1])
                incorrectLabeledDoc_count += 1
                correct_lable = '__label__'+line.split()[1]
                text =  ' '.join(line.split()[2:])
                text = text.replace(',', ' ')
                cve_cwe_summary_checked.write(f'{correct_lable} {text}\n')
            else:
                unknown_docs += 1
                continue
        
        print("Done!")
        print("Results statistics as follow:")
        dataset_size = line_count
        totalCorrected_docs = correctLabeledDoc_count +  incorrectLabeledDoc_count
        correct_docs_percentage = "{:.0%}".format(correctLabeledDoc_count/dataset_size)
        incorrect_docs_percentage = "{:.0%}".format(incorrectLabeledDoc_count/dataset_size)
        unknown_docs_precentage = "{:.0%}".format(unknown_docs/dataset_size)
        print(f'size of dataset: {dataset_size} docs')
        print(f'size of correct docs in the dataset: {totalCorrected_docs} docs')
        print(f'% of correct docs in fasttext format: {correct_docs_percentage}')
        print(f'% incorrect and fixed docs in fasttest format: {incorrect_docs_percentage}')
        print(f'% of docs unkown or can not be labeled: {unknown_docs_precentage}')


if __name__ == '__main__':
    # define class object
    transData = TransformData()

    # convert csv to fasttext format
    if( not os.path.exists("cve_cwe_summary_noisy.txt")):
        print("Strart extracting CVE-CWE-Summary from NVD dataset...")
        print("and convert csv to txt format")
        transData.to_fasttextFormat("./data/CVE_CWE_Summary.csv", "./data/cve_cwe_summary_noisy.txt")

    # check and update incorrect docs(lables) in the dataste
    transData.check_fix_dataset_format("./data/cve_cwe_summary_noisy.txt", "./data/cve_cwe_summary_clean.txt")