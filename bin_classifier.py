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
    if labels[0] == '__label__other':
        j = 1
    return r[0][j], r[1][j]


# example command to run classifier on the balanced pandas dataset:
# python ml_bin_classifier.py ../../datasets/data_set-pandas-balanced.txt ./out.txt
if __name__ == '__main__':

    print('* execute ' + sys.argv[0])

    # catch missing arguments
    try:
        a1 = sys.argv[1]
        a2 = sys.argv[2]
    except IndexError as error:
        print('\033[91m' + "Could not read arguments. Please use the correct command format. Example command:")
        print("python ml_bin_classifier.py ../../datasets/data_set-pandas-balanced.txt ./out.txt")
        exit()

    # get sys args
    data_set = sys.argv[1]
    fn_in = os.path.basename(data_set)
    f_out = sys.argv[2]

    # define paths for temporary files
    b_path_train = os.path.dirname(__file__) + './tmp/b_tmp_train.txt'
    e_path_train = os.path.dirname(__file__) + './tmp/e_tmp_train.txt'
    q_path_train = os.path.dirname(__file__) + './tmp/q_tmp_train.txt'
    b_path = os.path.dirname(__file__) + './tmp/BUG-' + fn_in
    e_path = os.path.dirname(__file__) + './tmp/ENHANCEMENT-' + fn_in
    q_path = os.path.dirname(__file__) + './tmp/QUESTION-' + fn_in

    try:

        print("Generating binary datasets")
        cmd = "python ../data_acquisition/create_binary_datasets.py " + str(data_set) + " ./tmp/"
        os.system(cmd)

        print("Converting dataset to array")
        f = open(data_set, 'r+', encoding="UTF-8")
        data = array(f.readlines())
        f.close()

        print("Converting bug dataset to array")
        f = open(b_path, 'r+', encoding="UTF-8")
        b_data = array(f.readlines())
        f.close()

        print("Converting enhancement dataset to array")
        f = open(e_path, 'r+', encoding="UTF-8")
        e_data = array(f.readlines())
        f.close()

        print("Converting question dataset to array")
        f = open(q_path, 'r+', encoding="UTF-8")
        q_data = array(f.readlines())
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

            TP_e = 0
            TP_FN_e = 0
            TP_FP_e = 0

            TP_q = 0
            TP_FN_q = 0
            TP_FP_q = 0

            print("New tenfold iteration:", str(fold), "-----------------------------------------")
            print("Creating bug train file")
            b_tmp_train = open(b_path_train, "w", encoding="UTF-8")
            for line in b_data[train]:
                b_tmp_train.write("".join(line))
            b_tmp_train.close()

            print("Creating enhancement train file")
            e_tmp_train = open(e_path_train, "w", encoding="UTF-8")
            for line in e_data[train]:
                e_tmp_train.write("".join(line))
            e_tmp_train.close()

            print("Creating question train file")
            q_tmp_train = open(q_path_train, "w", encoding="UTF-8")
            for line in q_data[train]:
                q_tmp_train.write("".join(line))
            q_tmp_train.close()

            # get test data
            test_data = data[test]

            print("start training...")
            # train 3 models
            b_model = fasttext.train_supervised(input=b_path_train)
            e_model = fasttext.train_supervised(input=e_path_train)
            q_model = fasttext.train_supervised(input=q_path_train)

            # testing loop
            print("start testing for tenfold iteration...")
            for i, line in enumerate(test_data):

                # get correct answer and input text from test data
                t = line.partition(' ')
                issue_text = t[2].replace('\n', '').replace('\r', '')
                correct_answer = t[0]

                # predict with 3 models
                b_res = b_model.predict(issue_text, k=-1)
                e_res = e_model.predict(issue_text, k=-1)
                q_res = q_model.predict(issue_text, k=-1)

                # parse guesses
                b_guess = get_guess(b_res)
                e_guess = get_guess(e_res)
                q_guess = get_guess(q_res)

                # get most likely label from 3 results
                res = {
                    b_guess[0]: b_guess[1],
                    e_guess[0]: e_guess[1],
                    q_guess[0]: q_guess[1]
                }
                guess = max(res, key=res.get)

                # save results for recall, precision and f1 calculations
                if guess == '__label__bug':
                    TP_FP_b += 1
                    if guess == correct_answer:
                        TP_b += 1

                if guess == '__label__enhancement':
                    TP_FP_e += 1
                    if guess == correct_answer:
                        TP_e += 1

                if guess == '__label__question':
                    TP_FP_q += 1
                    if guess == correct_answer:
                        TP_q += 1

                if correct_answer == '__label__bug':
                    TP_FN_b += 1

                if correct_answer == '__label__enhancement':
                    TP_FN_e += 1

                if correct_answer == '__label__question':
                    TP_FN_q += 1

                # log
                print("Issue nr." + str(i) + " has predicted: ")
                print("Bug res: ", str(b_guess))
                print("Enhancement res: ", str(e_guess))
                print("Question res: ", str(q_guess))
                print("Final guess: ", str(guess))
                print("Correct answer: ", correct_answer)
                print("-------------------------------------------------")

            # calculate benchmarks for ten fold iteration
            b_recall = TP_b / TP_FN_b
            b_precision = TP_b / TP_FP_b
            b_f1 = 2 * ((b_precision * b_recall) / (b_precision + b_recall))

            e_recall = TP_e / TP_FN_e
            e_precision = TP_e / TP_FP_e
            e_f1 = 2 * ((e_precision * e_recall) / (e_precision + e_recall))

            q_recall = TP_q / TP_FN_q
            q_precision = TP_q / TP_FP_q
            q_f1 = 2 * ((q_precision * q_recall) / (q_precision + q_recall))

            ges_recall = (b_recall + e_recall + q_recall) / 3
            ges_precision = (b_precision + e_precision + q_precision) / 3
            ges_f1 = (b_f1 + e_f1 + q_f1) / 3

            micro = (TP_b+TP_q+TP_e) / len(test_data)

            result = {
                '10-Fold iteration:': fold,
                'Mean f1': ges_f1,
                'Mean recall': ges_recall,
                'Mean precision': ges_precision,
                'Bug recall': b_recall,
                'Bug precision': b_precision,
                'Bug f1': b_f1,
                'Enhancement recall': e_recall,
                'Enhancement precision': e_precision,
                'Enhancement f1': e_f1,
                'Question recall': q_recall,
                'Question precision': q_precision,
                'Question f1': q_f1,
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
    except:
        print("An Error occurred")
    # in any case delete existing temporary files
    finally:
        print("Deleting tmp files")
        if os.path.exists(b_path_train):
            os.remove(b_path_train)
        if os.path.exists(e_path_train):
            os.remove(e_path_train)
        if os.path.exists(q_path_train):
            os.remove(q_path_train)
        if os.path.exists(b_path):
            os.remove(b_path)
        if os.path.exists(e_path):
            os.remove(e_path)
        if os.path.exists(q_path):
            os.remove(q_path)
        print("Exit.")

# formulas used for calculations:

# recall = TP/(TP+FN)
# recall_bug = #final guess == bug && correct answer == bug / #correct answers == bug

# precision = TP / (TP+FP)
# precision_bug = #final guess == bug && correct answer == bug / #final guess == bug

# f1 = 2*((precision*recall)/(precision+recall))