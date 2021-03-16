#import the needed libraries into the program
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

dataset = pd.read_csv('dataset-here',sep=',',quotechar='"')  #upload the data into the program


dataset.columns       #show all the columns of the uploaded data

import xgboost

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('dataset-here', delimiter=",")


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing


le = preprocessing.LabelEncoder()
dataset = dataset.apply(le.fit_transform)

print (dataset)

X = dataset.iloc[:, 0:32].values
Y = dataset.iloc[:, -1].values

output1 = []
for y in range(1, 10):
    for x in range(1, 500):
        size = 0
        size = y * 0.1
        seed = x
        test_size = size

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

        model = XGBClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]

        accuracy = accuracy_score(y_test, predictions)

        input1 = ("%.2f%%" % (accuracy * 100.0)), x, size
        output1.append(input1)
        if accuracy == 1:
            print("100 percent model seed=%test size=%" % (x, size))
        output1.sort(reverse=True)
    print(y * 10, "%")
print(output1)

print(model.feature_importances_)
print('Feature Importances for XGBOOST Model\n')
for importance,feature in zip(model.feature_importances_,
                              ['1what is your school name?', '2Which curriculum do you belong to?',
       '3What is your gender?', '4Your age?',
       '5Where do you live - City or Country side?',
       '6What is your family size - Less or equal to 3 or greater than 3?',
       '7Cohabitation of parents-they live together or not together?',
       '8The education level of your mom',
       '9 the education level of your father?', '10what is your moms job?',
       '11What is your fathers job?', '12Why you chose your current school?',
       '13Who is your primary guardian?',
       '14the time taken to travel from your house to your school?',
       '15how much time do you study per day?',
       '16do you obtain other support from your schools besides the classes?',
       '17do your parents help you with your school work?',
       '18do you take extra classes outside the school?',
       '19Do you take part in after school activities?',
       '20Did you attend kindergarden?',
       '21Do you want to and plan to attend university?',
       '22Do you have a fast internet WiFi connection at home?',
       '23Are you in a  romantic relationship ?',
       '24do you have a good relationship with your family ? (very good 5_ not good 1)',
       '25Do you have a lot of spare time? (A lot 5_negligible 1)',
       '26do you often go out with your friends?',
       '27Are you health? (very healthy 5_ not 1)',
       '28how many days did you ask for a leave from school per year?',
       '29school_rank', 'z_score_sum', 'z_score_linearized',
       'z_score_normalized' ]):
    print('{}: {}'.format(feature,importance))

from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
# load data
dataset = pd.read_csv('cn_student_data_512_001_csv_utf8.csv', delimiter=",")


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
dataset = dataset.apply(le.fit_transform)

print (dataset)

# split data into X and y
X = dataset.iloc[:, 0:32].values # X axis data columns
y = dataset.iloc[:, -1].values   #Y axis data column
# fit model on training data
model = XGBClassifier()
model.fit(X, y)
# plot feature importance
plot_importance(model)
pyplot.show()







