'''
# Author: Sultan S. Alqahtani
# Date: 13/04/2022 
# It is modified version from original code https://github.com/ChristianBirchler/ticket-tagger-analysis/blob/main/code-pipeline/classifiers/ml_bin_classifier.py 
'''
import json
import fasttext
import os
import os.path
import numpy as np
from numpy import array
from sklearn.model_selection import KFold
import sys


# returns the label and chance of that label
def get_guess(r):
    labels = r[0]
    j = 0
    if labels[0] == '__label__sec-report':
        j = 1
    return r[0][j], r[1][j]


# example command to run classifier on the balanced pandas dataset:
# python ml_bin_classifier.py ../../datasets/data_set-pandas-balanced.txt ./out.txt
if __name__ == '__main__':
   
    print("start here")
    # get sys args
    data_set = "./data/temp/chromium.txt"
    fn_in = os.path.basename(data_set)
    f_out = "./data/temp/out.txt"

    # define paths for temporary files
    b_path_train = "./data/temp/tmp/b_tmp_train.txt"
   # e_path_train = "./data/temp/tmp/e_tmp_train.txt"
   # q_path_train = "./data/temp/tmp/q_tmp_train.txt"
   # b_path = './data/temp/BUG-' + fn_in
   # e_path = './data/temp/ENHANCEMENT-' + fn_in
   # q_path = './data/temp/QUESTION-' + fn_in

    try:
        print("Converting dataset to array")
        f = open(data_set, 'r+', encoding="UTF-8")
        data = array(f.readlines())
        f.close()
        
        # array for details
        fold_outputs = []

        # fold count
        fold = 1

        # ten fold loop
        kfold = KFold(10, shuffle=True, random_state=1)
        for train, test in kfold.split(data):

            # init stats
            TP_b = 0
            TP_FN_b = 0
            TP_FP_b = 0
            
            print("New tenfold iteration:", str(fold), "-----------------------------------------")
            print("Creating bug train file")
            b_tmp_train = open(b_path_train, "w", encoding="UTF-8")
            for line in data[train]:
                b_tmp_train.write("".join(line))
            b_tmp_train.close()

            
            # get test data
            test_data = data[test]

            print("start training...")
            # train the models
            b_model = fasttext.train_supervised(input=b_path_train,epoch=40, lr=1.0)
           

            # testing loop
            print("start testing for tenfold iteration...")
            for i, line in enumerate(test_data):

                # get correct answer and input text from test data
                t = line.partition(' ')
                issue_text = t[2].replace('\n', '').replace('\r', '')
                correct_answer = t[0]

                # predict with the models
                b_res = b_model.predict(issue_text, k=-1)
            
                # parse guesses
                b_guess = get_guess(b_res)

               
                # get most likely label from the results
                res = {
                    b_guess[0]: b_guess[1]
                }
                guess = max(res, key=res.get)

                # save results for recall, precision and f1 calculations
                
                if guess == '__label__nonsec-report':
                    TP_FP_b += 1
                    if guess == correct_answer:
                        TP_b += 1
                
                if correct_answer == '__label__sec-report':
                    TP_FN_b += 1
                
                # log
                print("Issue nr." + str(i) + " has predicted: ")
                print("Bug res: ", str(b_guess))
                print("Final guess: ", str(guess))
                print("Correct answer: ", correct_answer)
                print("TP_FP_b:",TP_FP_b)
                print("TP_b" ,TP_b)
                print("TP_FN_b",TP_FN_b)
                print("-------------------------------------------------")

            # calculate benchmarks for ten fold iteration
            b_recall = TP_b / TP_FN_b
            b_precision = TP_b / TP_FP_b
            b_f1 = 2 * ((b_precision * b_recall) / (b_precision + b_recall))
            
            ges_recall = (b_recall) / 1
            ges_precision = (b_precision) / 1
            ges_f1 = (b_f1 ) / 1

            micro = (TP_b) / len(test_data)

            result = {
                '10-Fold iteration:': fold,
                'Mean f1': ges_f1,
                'Mean recall': ges_recall,
                'Mean precision': ges_precision,
                'Bug recall': b_recall,
                'Bug precision': b_precision,
                'Bug f1': b_f1,
                'Micro': micro
            }
            # log
            print("Fold over, here are results: ")
            print(json.dumps(result, indent=4))
            fold_outputs.append(result)

            fold += 1

        print("Done with 10 fold validation")
        # calculate over-all results
        mean_recall = 0
        mean_precision = 0
        mean_f1 = 0
        mean_micro = 0
        for f in fold_outputs:
            mean_f1 += (f['Mean f1'] / 10)
            mean_recall += (f['Mean recall'] / 10)
            mean_precision += (f['Mean precision'] / 10)
            mean_micro += (f['Micro'] / 10)

        # compile results as json
        output = {
            'Results': {
                'F1': mean_f1,
                'Recall': mean_recall,
                'Precision': mean_precision,
                'Micro': mean_micro
            },
            'Details': fold_outputs
        }
        dump = json.dumps(output, indent=4)
        print(dump)
        # write to output
        print("Writing output to file")
        o = open(f_out, 'w', encoding="UTF-8")
        o.write(dump)
        o.close()
    # catch and print exceptions
    except Exception as e: 
        print(e)
    # in any case delete existing temporary files
    finally:
        print("Deleting tmp files")
        if os.path.exists(b_path_train):
            os.remove(b_path_train)
        print("Exit.")

# formulas used for calculations:

# recall = TP/(TP+FN)
# recall_bug = #final guess == bug && correct answer == bug / #correct answers == bug

# precision = TP / (TP+FP)
# precision_bug = #final guess == bug && correct answer == bug / #final guess == bug

# f1 = 2*((precision*recall)/(precision+recall))