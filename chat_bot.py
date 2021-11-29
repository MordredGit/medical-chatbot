import pickle
from re import S
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# mapping strings to numbers
le = preprocessing.LabelEncoder()

severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()


def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum = sum + severityDictionary[item]
    if((sum*days)/(len(exp)+1) > 13):
        print("You should take the consultation from doctor. ")
    else:
        print("It might not be that bad but you should take precautions.")


def getDescription():
    global description_list
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open('symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)


def getInfo():
    name = input("Your Name: \t\t\t\t->")
    print("Hello ", name)


def check_pattern(dis_list, inp):
    import re
    pred_list = []

    regexp = re.compile(inp)
    for item in dis_list:
        if regexp.search(item):
            pred_list.append(item)

    if(len(pred_list) > 0):
        return 1, pred_list
    else:
        return 0, item


def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return disease


def tree_to_code(tree, feature_names, reduced_data):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    chk_dis = ",".join(feature_names).split(",")
    rows = [[i for i in chk_dis[index: index + 5]]
            for index in range(0, len(chk_dis) - 1, 5)]

    from tabulate import tabulate
    print(tabulate(rows))
    symptoms_present = []

    while True:
        disease_input = input("Enter the symptom you are experiencing: ->")
        conf, cnf_dis = check_pattern(chk_dis, disease_input)
        if conf == 1:
            break
        print("Enter valid symptom.")

    print("Searches related to input: ")
    for num, it in enumerate(cnf_dis):
        print(num, ")", it)

    conf_inp = int(
        input(f"Select the one you meant (0 - {num}): ")) if num != 0 else 0
    disease_input = cnf_dis[conf_inp]

    while True:
        try:
            num_days = int(input("Okay. From how many days ? : "))
            break
        except ValueError:
            print("Enter number of days.")

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero(
            )]
            print("Are you experiencing any ")
            symptoms_exp = []
            for syms in list(symptoms_given):
                print(syms, "? : ", end='')
                while True:
                    inp = input("")
                    if(inp == "yes" or inp == "no"):
                        break
                    else:
                        print("provide proper answers i.e. (yes/no) : ", end="")
                if(inp == "yes"):
                    symptoms_exp.append(syms)

            calc_condition(symptoms_exp, num_days)
            print("You may have", present_disease[0])
            print(description_list[present_disease[0]])
            precaution_list = precautionDictionary[present_disease[0]]
            print("Take following measures : ")
            for i, j in enumerate(precaution_list):
                print(i+1, ")", j)

            confidence_level = (1.0 * len(symptoms_present)
                                ) / len(symptoms_given)
            print("confidence level is " + str(confidence_level))

    recurse(0, 1)


def main():

    getSeverityDict()
    getDescription()
    getprecautionDict()
    getInfo()
    with open("trained_model", "rb") as model:
        clf = pickle.load(model)
        training = pd.read_csv('Training.csv')
        global le
        le.fit(training['prognosis'])
        reduced_data = training.groupby(training['prognosis']).max()

        cols = training.columns
        tree_to_code(clf, cols, reduced_data)


if __name__ == "__main__":
    main()
