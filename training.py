from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import pickle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


training = pd.read_csv('Training.csv')
cols = training.columns[:-1]
x, y = training[cols], training['prognosis']

# mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier().fit(x_train, y_train)
scores = cross_val_score(clf, x_test, y_test, cv=5)

print(f'Training Score:  {clf.score(x, y)}')
print(f"Testing Cross Validation Score: {scores.mean()}")

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

print(tree.export_text(clf))
fig = plt.figure()
_ = tree.plot_tree(clf,
                   feature_names=features,
                   filled=True)
plt.show()
print(f'{importances =}')
print(f'{indices =}')
print(f'{features =}')

with open("trained_model", "wb") as output:
    pickle.dump(clf, output)
