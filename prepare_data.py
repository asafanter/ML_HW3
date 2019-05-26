import numpy as np
import pandas as pd


########################################################################################################################
# save and split data
########################################################################################################################


def split_data(data: pd.DataFrame) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    :param data: the data that need to be splitted
    :return: np.ndarray tuple contains (training_indexes, validation_indexes and test_indexes),
    """
    # split [0, 1, ... , ken(data)] to 5 group, with shuffling
    number_of_samples = len(data.index)
    indexes_list = np.arange(number_of_samples)
    np.random.shuffle(indexes_list)
    indexes_partition = np.split(indexes_list, 5)

    # train - 60% (3 groups), validation - 20% (1 group), test - 20% (1 group)
    training_indexes = np.concatenate((indexes_partition[0], indexes_partition[1], indexes_partition[2]))
    validation_indexes = indexes_partition[3]
    test_indexes = indexes_partition[4]

    data.loc[:, 'partition'] = ''
    data.loc[training_indexes, 'partition'] = 'train'
    data.loc[validation_indexes, 'partition'] = 'validation'
    data.loc[test_indexes, 'partition'] = 'test'

    return data, training_indexes, validation_indexes, test_indexes


def save_split_data(data: pd.DataFrame, training_indexes: np.ndarray, validation_indexes: np.ndarray,
                    test_indexes: np.ndarray, path: np.str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    split "data" according the partition (training_indexes,validation_indexes,test_indexes).
    save this partition to files in "path".
    """
    data_ = data.copy().drop(['partition'], axis=1)

    # split data to train, validation and test (type is pd.DataFrame)
    train_df = data_.iloc[training_indexes]
    validation_df = data_.iloc[validation_indexes]
    test_df = data_.iloc[test_indexes]

    # save the partition to files
    train_df.to_csv(path + '/' + path + '_train.csv', header=True, index=False)
    validation_df.to_csv(path + '/' + path + '_validation.csv', header=True, index=False)
    test_df.to_csv(path + '/' + path + '_test.csv', header=True, index=False)


########################################################################################################################
# prepare data: fix outliers, fill missing values, preform data transformation
########################################################################################################################

def is_numeric(series: pd.Series) -> bool:
    """
    the function returns true if a Series is numeric or false if not
    :param series: the Series to be checked
    :return: true if series is numeric, otherwise return false
    """
    return series.dtype != "object"


##########################################
# fix outliers
##########################################

def fix_outliers_aux(arr: np.ndarray, train_arr: np.ndarray, thresh_hold_sigma: int) -> np.ndarray:
    """
    the function gets array with and replaced all negative numbers and outliers with median.
    data is an outlier if > thresh_hold_sigma * standard deviation
    :param arr: an unbalanced array which is need to be fixed
    :param train_arr: like 'arr', only from train set
    :param thresh_hold_sigma: the factor which will be decide if a data is an outlier
    :return: a new fixed array
    """
    train_arr = train_arr[~np.isnan(train_arr)]

    avg = np.average(train_arr)
    sdv = np.std(train_arr)
    median = np.median(train_arr)

    fixed = arr

    with np.errstate(invalid='ignore'):
        fixed[fixed < 0] = median
        fixed[fixed > avg + thresh_hold_sigma * sdv] = median

    return fixed


def fix_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """
    :param data: the data to be fixed
    :return: a fixed DataFrame (we detect outliers only for numeric features)
    """
    tmp_data = data
    training_indexes = data.index[data['partition'] == 'train'].values

    for feature in pd.Categorical(tmp_data).tolist():
        if is_numeric(tmp_data[feature]):
            column_values = tmp_data[feature].values
            column_train_values = tmp_data.loc[training_indexes, feature].values
            tmp_data[feature] = pd.Series(fix_outliers_aux(column_values, column_train_values, 3))

    return tmp_data


##########################################
# fill missing values
##########################################

def fill_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    :param data: the data which is contains missing values
    :return: data without missing values
    """
    training_indexes = data.index[data['partition'] == 'train'].values

    for feature in pd.Categorical(data).tolist():
        if feature == 'partition':
            continue

        column_train_values_no_null = data.loc[training_indexes, feature].dropna()

        # if numeric, fill with median
        if is_numeric(data[feature]):
            replacement = np.median(column_train_values_no_null)
            data[feature] = data[feature].fillna(replacement)

        # if not numeric (i.e, categorical), fill according to majority
        else:
            categories = column_train_values_no_null.value_counts().index
            values = column_train_values_no_null.value_counts().values
            max_id = values.argmax()
            replacement = categories[max_id]
            data[feature] = data[feature].fillna(replacement)

    return data


##########################################
# data transformation
##########################################

def data_transformation_of_numeric(data: pd.DataFrame) -> pd.DataFrame:
    """ perform normalization of numeric features"""
    training_indexes = data.index[data['partition'] == 'train'].values

    numeric_features_ = data.select_dtypes(exclude=["object"]).columns

    for f in numeric_features_:
        column_train_values = data.loc[training_indexes, f].values
        min_on_train = column_train_values.min()
        max_on_train = column_train_values.max()
        data[f] = (data[f] - min_on_train) / (max_on_train - min_on_train)
        data[f] = (data[f] - data[f].min()) / (data[f].max() - data[f].min())
    return data


def data_transformation_of_categorical(data: pd.DataFrame) -> pd.DataFrame:
    """ perform categorical to numeric conversion"""

    # Multi-valued un-ordered:
    for f in ['Vote', 'Most_Important_Issue']:
        data[f] = data[f].astype("category").cat.rename_categories(range(data[f].nunique())).astype(int)

    return data


def data_transformation(data: pd.DataFrame) -> pd.DataFrame:
    """
    on numeric features: perform normalization
    on categorical features: convert to numeric
    """
    data = data_transformation_of_numeric(data)
    data = data_transformation_of_categorical(data)
    return data
