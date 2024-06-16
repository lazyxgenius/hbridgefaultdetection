import warnings
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from matplotlib import pyplot

import joblib

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import numpy as np
warnings.filterwarnings("ignore")

dataset1 = pd.read_csv(r'C:\Users\Aditya PC\PycharmProjects\duplicatefinder\scores.csv')
array = dataset1.values

dataset2 = pd.read_csv(r'C:\Users\Aditya PC\PycharmProjects\duplicatefinder\labels.csv')
labels = dataset2.values


x_train, x_test, y_train, y_test = train_test_split(array, labels, test_size=0.2, random_state=1)

models = []

models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto', kernel='linear', max_iter=1000)))  # it's taking too long to train.


# names = []
# results = []
#
# for name, model in models:
#     kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_result = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
#     results.append(cv_result)
#     names.append(name)
#     print('%s: %f, (%f)' % (name, cv_result.mean(), cv_result.std()))
#
# pyplot.boxplot(results, labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()

# BoxPlot generated after this step is shown in BOXPLOT.png looking at the boxplot,
# CART - Decision tree classifier has appeared to perform the best.

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
filename = "saved_DTC.joblib"
joblib.dump(model, filename)

print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# see reports.png

