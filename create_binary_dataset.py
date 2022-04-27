#!/usr/bin/python3

import sys
import re
import os


def only(issue, label):
    """
    If the label is not the same as the label given
    as input then the issue is relabeled with 'other'
    """
    prefix_pattern = re.compile(r'__label__\w+')
    match = prefix_pattern.match(issue)
    prefix_ind = match.span()
    prefix = issue[prefix_ind[0]:prefix_ind[1]]
    postfix = issue[prefix_ind[1]:]

    if prefix == '__label__' + label:
        new_prefix = '__label__' + label
    else:
        new_prefix = '__label__other'

    return new_prefix + postfix


if __name__ == '__main__':
    print('* execute ' + sys.argv[0])

    f_in = "./data/temp/dataset.txt"
    fn_in = os.path.basename(f_in)

    dir_out = ""
    if len(sys.argv) == 3:
        dir_out = sys.argv[2]

    f_out_bug = open(dir_out + 'BUG-' + fn_in, mode='w', encoding="UTF-8")
    f_out_enhancement = open(dir_out + 'ENHANCEMENT-' + fn_in, mode='w', encoding="UTF-8")
    f_out_question = open(dir_out + 'QUESTION-' + fn_in, mode='w', encoding="UTF-8")

    with open(f_in, mode='r', encoding="UTF-8") as f:
        cnt = 0
        for issue in f:
            # print('* check label of issue ' + str(cnt))
            f_out_bug.write(only(issue, 'bug'))
            f_out_enhancement.write(only(issue, 'enhancement'))
            f_out_question.write(only(issue, 'question'))
            cnt += 1

    f_out_bug.close()
    f_out_enhancement.close()
    f_out_question.close()
