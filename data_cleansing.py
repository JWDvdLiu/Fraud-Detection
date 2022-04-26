import pandas as pd
from scipy import stats
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

def clean_dataset(dataset):

    df = pd.read_csv(f"{dataset}_transaction.csv")

    # I check the datatypes of the columns to find columns with non-numerical values or integer.
    # We want to convert the categorical values to numerical values for easier model building.
    dtypes = df.dtypes.to_dict()
    non_numerical = []
    integer_col = []
    for col_name, typ in dtypes.items():
        if typ == 'object':
            non_numerical.append(col_name)
        if typ == 'int64':
            integer_col.append(col_name)
            # print(f"['{col_name}'].dtype == {typ}")
    #print(non_numerical)
    #print(integer_col)

    # First we deal with x5
    # df['x5'][0:10]
    # we see that this column contains percentage values as string

    # we convert the strings to float numbers
    df['x5'] = [float(x.strip('%')) / 100 if isinstance(x, str) else x for x in df['x5']]
    # print(df.x5.dtype)


    # Then we deal with x6
    # df['x6'][0:10]
    # df['x6'].unique()
    # We see that this column contains the weekdays as strings
    categorical_convert = {
        'x6': {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thurday': 4, 'Friday': 5}
    }
    df = df.replace(categorical_convert)
    # print(df.x6.dtype)


    # Then we deal with x20
    # df['x20'][0:10]
    # df['x20'].unique()
    # We see that this column contains the months as strings
    categorical_convert = {
        'x20': {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5,
                'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sept': 9, 'Oct': 10,
                'Nov': 11, 'Dec': 12}
    }
    df = df.replace(categorical_convert)
    # print(df.x20.dtype)


    # Then we deal with x27
    # df['x27'][0:10]
    # df['x27'].unique()
    # We see that this columns contains the time in the day as strings
    # we convert them in order from morning to night
    categorical_convert = {
        'x27': {'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4}
    }
    df = df.replace(categorical_convert)
    # print(df.x27.dtype)

    # Then we deal with x49
    # df['x49'][0:10]
    # df['x49'].unique()
    # Notice that this column contains Boolean value only.
    # We multiply them by 1 to convert to int
    df['x49'] = df['x49'] * 1
    # df['x49'].value_counts().index

    # Lastly we deal with x57
    # we see that this column contains money value start with dollar sign

    # we convert the strings to float numbers
    df['x57'] = [float(x.strip('$')) if isinstance(x, str) else x for x in df['x57']]
    # print(df.x57.dtype)

    # df.dropna() returns an empty dataframe, which means every row
    # contains at least one NaN value. Thus we have to find another way.
    # df.dropna(how='all')
    # We fill the NaN values with modes or means
    for col in df.columns:
        if col in non_numerical:
            # We replace the NaN values with mode for the categorical values that we converted earlier
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            # We replace the NaN values with the mean of the column for those float value columns
            df[col] = df[col].fillna(df[col].mean())

    # train[(np.abs(stats.zscore(feature_data)) < 3).all(axis=1)]['y'].sum()
    # train['y'].sum()
    # By removing the outliers from the training set, we notice that a large portion of suspicious transactions are removed,
    # plus that the column names are not given, thus we will not deal with outliers.
    # The dataset is now ready for feature selection.

    corr = df.corr().abs()
    s = corr.unstack()
    so = s.sort_values(kind="quicksort", ascending=False)
    so[101:112]

    # Since we are using pearson correlation, we need to make sure the
    # values are normally distributed, we will check their histogram.

    # normality test for x76
    num_bins = 20
    x_val = df['x76']
    plt.hist(x_val, num_bins, facecolor='blue', alpha=0.5)
    plt.show

    # normality test for x32
    num_bins = 20
    x_val = df['x32']
    plt.hist(x_val, num_bins, facecolor='blue', alpha=0.5)
    plt.show

    # normality test for x81
    num_bins = 20
    x_val = df['x81']
    plt.hist(x_val, num_bins, facecolor='blue', alpha=0.5)
    plt.show

    # normality test for x95
    num_bins = 20
    x_val = df['x95']
    plt.hist(x_val, num_bins, facecolor='blue', alpha=0.5)
    plt.show

    # normality test for x35
    num_bins = 20
    x_val = df['x35']
    plt.hist(x_val, num_bins, facecolor='blue', alpha=0.5)
    plt.show

    # normality test for x61
    num_bins = 20
    x_val = df['x61']
    plt.hist(x_val, num_bins, facecolor='blue', alpha=0.5)
    plt.show

    # normality test for x41
    num_bins = 20
    x_val = df['x41']
    plt.hist(x_val, num_bins, facecolor='blue', alpha=0.5)
    plt.show

    # Notice that there are 5 pairs of variables with > 0.9 correlation, while the rest are all below 0.8
    # we remove one from each of the 5 pairs that has more than 0.99 correlation, and x41 since it is highly
    # correlated to both x81 and x95
    df = df.drop(columns=['x76', 'x81', 'x35', 'x41'])

    df.to_csv(f"Modified_Datasets/cleaned_{dataset}.csv", index=False)

    ###############################################################################################################
    # While testing for different models, I achieved very high f1 score surprisingly using DecisionTree
    # over just 7 most important variables.
    # Thus I am making this new dataframe with just 7 most important variables

    # Using the extra trees classifier, I identify the level of importance of each variables
    # model = ExtraTreesClassifier(n_estimators=10)
    # model.fit(df.iloc[:, 0:-1], df.iloc[:, -1])
    # importance_level = model.feature_importances_

    # I find the variables with importance level higher than 0.012
    # very_import = []
    # for i, x in enumerate(importance_level):
    #    if x > 0.012:
    #        very_import.append(i)
    # very_import

    MOST_IMPORTANT = ['x1', 'x9', 'x12', 'x32', 'x54', 'x61', 'x95']
    # I store the variables into a new file
    new_df = df.loc[:, MOST_IMPORTANT]
    new_df.to_csv(f"Modified_Datasets/super_cleaned_{dataset}_variables.csv", index=False)
