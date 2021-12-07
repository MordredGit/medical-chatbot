import pickle
import pandas as pd
from sklearn import preprocessing as prep
from sklearn.tree import _tree
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

le = prep.LabelEncoder()

sevDict = dict()
descList = dict()
precDict = dict()


def getInfo():
    name = input("Your Name: \t\t\t\t->")
    print("Hello ", name)


def calcCondition(exp, days):
    sum = 0
    for item in exp:
        sum = sum + sevDict[item]
    if((sum * days) / (len(exp) + 1) > 13):
        print("It is getting worse. You consult a doctor. ")
    else:
        print("It might not be that severe. Precautionary measures will help.")


def loadSevDict():
    global sevDict
    with open('symptom_severity.csv') as csv_file:
        csvReader = csv.reader(csv_file, delimiter=',')
        try:
            for row in csvReader:
                element = {row[0]: int(row[1])}
                sevDict.update(element)
        except:
            pass


def loadDesc():
    global descList
    with open('symptom_Description.csv') as csv_file:
        csvReader = csv.reader(csv_file, delimiter=',')
        for row in csvReader:
            _description = {row[0]: row[1]}
            descList.update(_description)


def loadPrecDict():
    global precDict
    with open('symptom_precaution.csv') as csv_file:

        csvReader = csv.reader(csv_file, delimiter=',')
        for row in csvReader:
            precElement = {row[0]: [row[1], row[2], row[3], row[4]]}
            precDict.update(precElement)


def treeToCode(tree, feature_names, reduced_data):
    tree_ = tree.tree_
    tree_feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    all_diseases = ",".join(feature_names).split(",")
    rows = [[i for i in all_diseases[index: index + 5]]
            for index in range(0, len(all_diseases) - 3, 5)]

    from tabulate import tabulate
    print(tabulate(rows))
    symptoms_present = []

    while True:
        input_disease = input(
            "Enter the symptom that you are afflicted with: ->")
        conf, confirmed_disease = matchSymptom(all_diseases, input_disease)
        if conf == 1:
            break
        print("Enter valid symptom.")

    print("Symptom found related to input: ")
    for num, symptom in enumerate(confirmed_disease):
        print(num, ")", symptom)

    conf_input = int(
        input(f"Select the one you meant (0 - {num}): ") if num != 0 else 0)
    disease_input = confirmed_disease[conf_input]

    while True:
        try:
            num_days = int(
                input("From how many days are you experiencing the symptom ? : "))
            break
        except ValueError:
            print("Enter number of days.")

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = tree_feature_name[node]
            threshold = tree_.threshold[node]

            val = 1 if name == disease_input else 0

            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[
                reduced_data.loc[present_disease].values[0].nonzero()
            ]
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

            calcCondition(symptoms_exp, num_days)
            print("You may have", present_disease[0])
            print(descList[present_disease[0]])
            precaution_list = precDict[present_disease[0]]
            print("Take following measures : ")
            for i, j in enumerate(precaution_list):
                print(i+1, ")", j)

    recurse(0, 1)


def matchSymptom(dis_list, inp):
    import re
    predictionList = []

    reg_ex = re.compile(inp)
    for item in dis_list:
        if reg_ex.search(item):
            predictionList.append(item)

    if(len(predictionList) > 0):
        return 1, predictionList
    else:
        return 0, item


def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return disease


def main():

    loadSevDict()
    loadDesc()
    loadPrecDict()
    getInfo()
    with open("trained_model", "rb") as model:
        clf = pickle.load(model)
        training = pd.read_csv('Training.csv')
        global le
        le.fit(training['prognosis'])
        reduced_data = training.groupby(training['prognosis']).max()

        cols = training.columns
        treeToCode(clf, cols, reduced_data)


if __name__ == "__main__":
    main()
