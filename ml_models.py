import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

# I load the cleaned historical dataset and separate the x's and y columns

train = pd.read_csv("Modified_Datasets/cleaned_historical.csv")
feature_data = np.array(train.iloc[:, 0:-1])
sus_data = np.array(train.iloc[:, -1])


# This is a helper function for me to calculate the f1 score, since there are only
# roughly 5% suspicious transaction, the accuracy measurement is not useful in this situation,
# I will compare F1 score of different models instead:
# Confusion matrix:
#   1. My model gives suspicious, the transaction is suspicious (True Positive)
#   2. My model gives suspicious, the transaction is non-suspicious (False Positive)
#   3. My model gives non-suspicious, the transaction is non-suspicious (True Negative)
#   4. My model gives non-suspicious, the transaction is suspicious (False Negative)
def f1_score(y_predict, y_actual):
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    for i in range(len(y_predict)):
        if y_actual[i] == y_predict[i] == 1:
            true_pos += 1
        if y_predict[i] == 1 and y_actual[i] != y_predict[i]:
            false_pos += 1
        if y_actual[i] == y_predict[i] == 0:
            true_neg += 1
        if y_predict[i] == 0 and y_actual[i] != y_predict[i]:
            false_neg += 1
    # Using the confusion matrix I calculate the precision and recall to calculate F1
    if (true_pos + false_pos) == 0:
        precision = 0
    else:
        precision = true_pos / (true_pos + false_pos)

    if (true_pos + false_neg) == 0:
        recall = 0
    else:
        recall = true_pos / (true_pos + false_neg)

    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * (recall * precision) / (recall + precision)
    return f1
    # return true_pos, false_pos, true_neg, false_neg


# I define the get score function to help me perform k-fold Cross-Validation
# on multiple models
# The function takes in a model and the train/test sets and calculate the score
# as well as the confusion matrix for the model.
def get_scores(model, x_train, x_test, y_Train, y_Test):
    model.fit(x_train, y_Train)
    score = model.score(x_test, y_Test)
    f1 = f1_score(model.predict(x_test), y_Test)
    return f1, score


# We will perform stratifiedKFold with k = 10
kf = StratifiedKFold(n_splits=10)

# We will use the following model:
# 1. Logistic Regression
# 2. Support Vector Classification.
# 3. Random Forest
# 4. Decision Tree
# 5. Naive Bayes

score_logistic = []
f1_logistic = []
score_svc = []
f1_svc = []
score_rf = []
f1_rf = []
score_dt = []
f1_dt = []
score_nb = []
f1_nb = []

# I run all models for each fold, and load the confusion matrices and scores for each model

sus_data = sus_data.astype(int)
for train_index, test_index in kf.split(feature_data, sus_data):
    X_train, X_test, y_train, y_test = feature_data[train_index], feature_data[test_index], sus_data[train_index], \
                                       sus_data[test_index]

    # Logistic Regression: I use liblinear and ovr since data is binary *according to doc
    performance_logistic = get_scores(LogisticRegression(solver='liblinear', multi_class='ovr'), X_train, X_test,
                                      y_train, y_test)
    score_logistic.append(performance_logistic[1])
    f1_logistic.append(performance_logistic[0])

    # Support Vector Classification model is not a good choice from testing and is very slow
    # performance_svc = get_scores(SVC(gamma='auto'), X_train, X_test, y_train, y_test)
    # score_svc.append(performance_svc[1])
    # f1_svc.append(performance_svc[0])

    # Random Forest: I decided to use n_estimators = 30 by testing on jupyter notebook
    performance_rf = get_scores(RandomForestClassifier(n_estimators=30), X_train, X_test, y_train, y_test)
    score_rf.append(performance_rf[1])
    f1_rf.append(performance_rf[0])

    # Decision Tree
    performance_dt = get_scores(DecisionTreeClassifier(), X_train, X_test, y_train, y_test)
    score_dt.append(performance_dt[1])
    f1_dt.append(performance_dt[0])

    # Naive Bayes
    performance_nb = get_scores(GaussianNB(), X_train, X_test, y_train, y_test)
    score_nb.append(performance_nb[1])
    f1_nb.append(performance_nb[0])

print('logistics f1: ', f1_logistic, '\n')
print('Random Forest f1: ', f1_rf, '\n')
print('Decision Tree f1: ', f1_dt, '\n')
print('Naive Bayes f1: ', f1_nb, '\n')
# From running the above testings, I notice decision tree out perform other models in terms of f1
#
# Later I realised that the decision tree with 9 most important variables out perform any other models by far
