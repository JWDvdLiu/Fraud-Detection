import data_cleansing
import predict_deci_tree

# In my solution, I first cleaned the datasets, I filled the NaN values for float variables with the column's mean, and for categorical variables with the column's mode.
# Then I checked the pearson correlation between all variables. I removed 4 of the highly correlated variables after confirmed their normality. 
# Then I performed k-fold cross validation over multiple models including Logistic Regression, Support Vector Classification, Random Forest, Decision Tree, and Naive Bayes.
# I also tested using Nerual Network with 2 hidden layers with activation function of relu and sigmoid, and binary cross entropy loss function.
# Since the datasets are heavily imbalanced, there are only ~5% suspicious transactions. Thus we test the models over F1-score instead of accruacy.
# I noticed that decision tree and Nerual Network out performs the other models.

# Then I performed hyperparameter tuning over Decision Tree and the Nerual Networks
# For Nerual Network, I tested different batch size, number of epochs, number of nodes in each layer.
# For Decision Tree, I tested different max_depth, and number of features I feed it. I used the feature_importances_ method from scikit-learn to find the most important features.

# At the end, a decision tree model using only the 9 most important feature yielded the best result, out performs other models by far.


# Prepare both datasets, output the following csv files
# 1. cleaned_historical.csv
# 2. cleaned_current.csv
# 3. super_cleaned_historical_variables (9 features)
# 4. super_cleaned_current_variables (9 Features)
data_cleansing.clean_dataset('historical')
data_cleansing.clean_dataset('current')

# nn_model.py and ml_models.py are the code that I used to test different models.

# Makes prediction using the decision tree model with 9 features.
# Outputs the current_transaction_predicted.csv, which is the prediction result.
predict_deci_tree.make_prediction()