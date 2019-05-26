from prepare_data import *

import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


########################################################################################################################
# STEP 1: load the data, split to train/validation/test and save 'original_data'+'prepared_data' (according to HW2)
########################################################################################################################

def prepare_data(data_path: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    divide data set to three groups after preparation: training set, validation set and test set
    :param data_path: a path to the csv data file to be divided
    :return: training set, validation set and test set
    """
    # save the original data (1000 samples x 37 features [+'Vote']):
    df = pd.read_csv(data_path)
    df, training_set, validation_set, test_set = split_data(df)
    save_split_data(df, training_set, validation_set, test_set, 'original_data')

    # save the prepared data (1000 samples x 9 features [+'Vote','partition']):
    df = df[['Vote', 'Most_Important_Issue', 'Avg_monthly_expense_on_pets_or_plants', 'Avg_environmental_importance',
             'Avg_Residancy_Altitude', 'Yearly_ExpensesK', 'Avg_government_satisfaction', 'Weighted_education_rank',
             'Number_of_valued_Kneset_members', 'Avg_education_importance', 'partition']]
    df = fix_outliers(df)
    df = fill_missing_values(df)
    df = data_transformation(df)
    save_split_data(df, training_set, validation_set, test_set, 'prepared_data')

    return read_sets('prepared_data')


def read_sets(dir_path: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    :param dir_path: a dir which contains a divided data sets
    :return: training set, validation set and test set
    """
    training_set = pd.read_csv('{0}/{0}_train.csv'.format(dir_path))
    validation_set = pd.read_csv('{0}/{0}_validation.csv'.format(dir_path))
    test_set = pd.read_csv('{0}/{0}_test.csv'.format(dir_path))

    return training_set, validation_set, test_set


########################################################################################################################
# STEP 2: train models
########################################################################################################################

def find_hyperparameter(scores):
    for (score, parameter, clf) in scores:
        print('\t p={0}\t score={1}'.format(parameter, score))
    (max_score, parameter, clf) = max(scores)
    print('\t >>>>\t p={0} achieved the maximum score of {1}'.format(parameter, max_score))
    return clf


def train_models_with_cross_validation_and_find_hyperparameters(x, y):
    print('\n______ STEP 2: find hyper-parameters ______')
    models = []  # list of models, after choosing for each its best hyper-parameters

    # KNeighborsClassifier:
    print('\nKNeighborsClassifier: p=(n_neighbors,weights)')
    print('  n_neighbors = num of neighbors; default=5')
    print('  weights = weight function used in prediction; default=uniform')
    scores = []
    for n_neighbors in range(1, 20):
        for weights in ['uniform', 'distance']:
            clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
            score = np.mean(cross_val_score(estimator=clf, X=x, y=y, cv=10))
            scores.append((score, (n_neighbors, weights), clf))
    models.append(find_hyperparameter(scores))

    # DecisionTreeClassifier:
    print('\nDecisionTreeClassifier: p=(criterion,min_samples_split)')
    print('  criterion = the function to measure the quality of a split; default=gini')
    print('  min_samples_split = min num of samples required to be at leaf; default=2')
    scores = []
    for criterion in ['gini', 'entropy']:
        for min_samples_split in range(2, 20):
            clf = DecisionTreeClassifier(criterion=criterion, min_samples_split=min_samples_split)
            score = np.mean(cross_val_score(estimator=clf, X=x, y=y, cv=10))
            scores.append((score, (criterion, min_samples_split), clf))
    models.append(find_hyperparameter(scores))

    # RandomForestClassifier:
    print('\nRandomForestClassifier: p=(criterion,min_samples_split)')
    print('  criterion = the function to measure the quality of a split; default=gini')
    print('  min_samples_split = min num of samples required to be at leaf; default=2')
    scores = []
    for criterion in ['gini', 'entropy']:
        for min_samples_split in range(2, 20):
            clf = RandomForestClassifier(criterion=criterion, min_samples_split=min_samples_split)
            score = np.mean(cross_val_score(estimator=clf, X=x, y=y, cv=10))
            scores.append((score, (criterion, min_samples_split), clf))
    models.append(find_hyperparameter(scores))

    return models


########################################################################################################################


if __name__ == '__main__':
    # STEP 1: load the data, split to train/validation/test and save 'original_data'+'prepared_data' (according to HW2):
    # training, validation, test = prepare_data('ElectionsData.csv')  # todo: uncomment this before submission
    training, validation, test = read_sets('prepared_data')
    training_X, training_Y = training.drop(['Vote'], axis=1), training['Vote']
    validation_X, validation_Y = validation.drop(['Vote'], axis=1), validation['Vote']
    test_X, test_Y = test.drop(['Vote'], axis=1), test['Vote']

    # STEP 2: train different models with cross validation and only on the training set, on order to tune the
    # hyper-parameters of each model. the function return list of models objects, after choosing the hyper-parameters:
    models = train_models_with_cross_validation_and_find_hyperparameters(training_X, training_Y)
