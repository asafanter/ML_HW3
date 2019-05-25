import numpy as np
import pandas as pd
# import pylab as p


def fix_outliers(arr: np.ndarray, train_arr: np.ndarray, thresh_hold_sigma: int) -> np.ndarray:
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


def is_numeric(series: pd.Series) -> bool:
    """
    the function returns true if a Series is numeric or false if not
    :param series: the Series to be checked
    :return: true if series is numeric, otherwise return false
    """
    return series.dtype != "object"


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
