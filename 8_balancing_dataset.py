#!/usr/local/bin/python3
# @author cpuhrsch https://github.com/cpuhrsch
# @author Loreto Parisi loreto@musixmatch.com

import argparse
import glob
import os
import numpy as np
from sklearn.metrics import confusion_matrix

data_source = glob.glob('./data/bug_reports/*.txt')

def parse_labels(path):
    with open(path, 'r') as f:
        return np.array(list(map(lambda x: x[9:], f.read().split())))

def balancing_dataset(path, pname):

    lines = open(path, 'r').readlines()
    
    count = 0
    for line in lines:
        label = line.split()[0]
        if label =='__label__sec':
            count += 1
            with open(pname+'_balanced_dataset.txt', 'a') as file:
                file.write(line)
    
    for line in lines:
        label = line.split()[0]
        if label == '__label__nonsec' and count>0:
            count -= 1
            with open(pname+'_balanced_dataset.txt', 'a') as file:
                file.write(line)

if __name__ == "__main__":
    for project_data in data_source:
        project_file = os.path.basename(project_data)
        print("Balancing project: " + project_file)
        pname = project_file.split('.')[0]
        balancing_dataset('./data/bug_reports/'+project_file, pname)