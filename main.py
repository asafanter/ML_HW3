import numpy as np
import pandas as pd

import utils


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
    train_df.to_csv(path + '/' + 'train.csv', header=True, index=False)
    validation_df.to_csv(path + '/' + 'validation.csv', header=True, index=False)
    test_df.to_csv(path + '/' + 'test.csv', header=True, index=False)


def fix_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """
    :param data: the data to be fixed
    :return: a fixed DataFrame (we detect outliers only for numeric features)
    """
    tmp_data = data
    training_indexes = data.index[data['partition'] == 'train'].values

    for feature in pd.Categorical(tmp_data).tolist():
        if utils.is_numeric(tmp_data[feature]):
            column_values = tmp_data[feature].values
            column_train_values = tmp_data.loc[training_indexes, feature].values
            tmp_data[feature] = pd.Series(utils.fix_outliers(column_values, column_train_values, 3))

    return tmp_data


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
        if utils.is_numeric(data[feature]):
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


def data_transformation(data: pd.DataFrame) -> pd.DataFrame:
    """
    on numeric features: perform normalization
    on categorical features: convert to numeric
    """
    data = utils.data_transformation_of_numeric(data)
    data = utils.data_transformation_of_categorical(data)
    return data


def prepareData(data_path: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    divide data set to three groups after preparation: training set, validation set and test set
    :param data_path: a path to the csv data file to be divided
    :return: training set, validation set and test set
    """
    df = pd.read_csv(data_path)
    df, training_set, validation_set, test_set = split_data(df)
    save_split_data(df, training_set, validation_set, test_set, 'Original_data')

    # all_features, numeric_features, categorical_features = map_features(df)
    df = fix_outliers(df)
    df = fill_missing_values(df)
    df = data_transformation(df)

    save_split_data(df, training_set, validation_set, test_set, 'Prepared_data')

    return readSets('Prepared_data')


def readSets(dir_path: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    :param dir_path: a dir which contains a divided data sets
    :return: training set, validation set and test set
    """

    training_set = pd.read_csv('{0}/train.csv'.format(dir_path))
    validation_set = pd.read_csv('{0}/validation.csv'.format(dir_path))
    test_set = pd.read_csv('{0}/test.csv'.format(dir_path))

    return training_set, validation_set, test_set


if __name__ == '__main__':
    # training, validation, test = prepareData('ElectionsData.csv')

    training, validation, test = readSets('Prepared_data')

    print(training.shape)
