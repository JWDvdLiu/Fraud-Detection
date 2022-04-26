import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# I load the cleaned historical dataset and seper.ate the x's and y columns
train = pd.read_csv("Modified_Datasets/cleaned_historical.csv")
feature_data = np.array(train.iloc[:, 0:-1])
sus_data = np.array(train.iloc[:, -1])
# I split the cleaned historical data set into training and testing for Model training.
x_train, x_test, y_train, y_test = train_test_split(feature_data, sus_data, test_size=0.25)

# After some testing, I decided to use two hidden layers with 128 and 64 nodes.
model = Sequential()
model.add(Dense(128, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))

# Last layer uses sigmoid function to output result from (0,1)
model.add(Dense(1, activation='sigmoid'))

model.summary()

# I use the binary_crossentropy function as loss function
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(x=x_train, y=y_train, epochs=20, verbose=1)

results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

final = np.round(model.predict(x_test))


# This is a helper function for me to calculate the confusion matrix:
#   1. My model gives suspicious, the transaction is suspicious (True Positive)
#   2. My model gives suspicious, the transaction is non-suspicious (False Positive)
#   3. My model gives non-suspicious, the transaction is non-suspicious (True Negative)
#   4. My model gives non-suspicious, the transaction is suspicious (False Negative)
def performance_test(y_predict, y_actual):
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


# The f1 score is ~0.43 much lower than decision tree model's 0.66
print(performance_test(final, y_test))
