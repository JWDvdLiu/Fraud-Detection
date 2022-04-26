import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from sklearn.tree import DecisionTreeClassifier
import joblib

def make_prediction():
    train = np.array(pd.read_csv("Modified_Datasets/super_cleaned_historical_variables.csv"))
    y_data = np.array(pd.read_csv("Modified_Datasets/cleaned_historical.csv").iloc[:, -1])


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
    # The function takes in a model and the train/test sets and calcualte the score
    # as well as the confusion matrix for the model.
    def get_scores(model, x_train, x_test, y_Train, y_Test):
        model.fit(x_train, y_Train)
        score = model.score(x_test, y_Test)
        f1 = f1_score(model.predict(x_test), y_Test)
        return f1, score


    # We will perform stratifiedKFold with k = 10
    kf = StratifiedKFold(n_splits=10)

    # We will use decision tree on this 9 variable data set
    dt = DecisionTreeClassifier()

    # score and f1 for this model
    score_dt = []
    f1_dt = []

    # perform k-fold cross validation with this model
    for train_index, test_index in kf.split(train, y_data):
        X_train, X_test, y_train, y_test = train[train_index], train[test_index], y_data[train_index], y_data[test_index]

        # Decision Tree
        performance_dt = get_scores(dt, X_train, X_test, y_train, y_test)
        score_dt.append(performance_dt[1])
        f1_dt.append(performance_dt[0])

    # This model's f1 out perform other models with > 0.67 f1 score
    # print('Decision Tree f1: ', f1_dt)

    # Making the final model and prediction:
    predict_data = np.array(pd.read_csv("Modified_Datasets/super_cleaned_current_variables.csv"))

    #final_dt = DecisionTreeClassifier()
    #final_dt.fit(train, y_data)
    filename = 'Final_Model/finalized_dt_model.sav'
    #joblib.dump(final_dt, filename)

    # Perform prediction with the model that I saved
    final_dt = joblib.load(filename)
    predict_result = final_dt.predict(predict_data)

    current_transaction = pd.read_csv("current_transaction.csv")
    current_transaction['y'] = predict_result

    current_transaction.to_csv("current_transaction_predicted.csv", index=False)
