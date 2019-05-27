from prepare_data import *
import sys
import numpy

import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

# scikit-learn imports
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from sklearn.metrics import classification_report

# np.set_printoptions(threshold=sys.maxsize)
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 15)


########################################################################################################################
# STEP 1: load the data, split to train/validation/test and save 'original_data'+'prepared_data' (according to HW2).
########################################################################################################################

def prepare_data(data_path: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    divide data set to three groups after preparation: training set, validation set and test set
    :param data_path: a path to the csv data file to be divided
    :return: training set, validation set and test set
    """

    print('\n\n_________________ STEP 1: prepare the data _________________')

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
# STEP 2: train different models with cross validation and only on the training set, in order to tune the
#         hyper-parameters of each model (the func return list of models, after choosing the hyper-parameters).
########################################################################################################################

def find_hyperparameter(scores):
    for (score, parameter, clf) in scores:
        print('\t p={0}\t score={1}'.format(parameter, score))
    (max_score, parameter, clf) = max(scores)
    print('\t >>>>\t p={0} achieved the maximum score of {1}'.format(parameter, max_score))
    return clf, max_score


def train_models_with_cross_validation_in_order_to_find_hyperparameters(x, y):
    print('\n\n_________________ STEP 2: find hyper-parameters _________________')

    models_and_scores = []  # list of models, after choosing for each its best hyper-parameters

    # RandomForestClassifier:
    print('\nRandomForestClassifier: p=(criterion,min_samples_split)')
    print('  criterion = the function to measure the quality of a split; default=gini')
    print('  min_samples_split = min num of samples required to be at leaf; default=2')
    scores = []
    for criterion in ['gini', 'entropy']:
        for min_samples_split in range(2, 15):
            clf = RandomForestClassifier(criterion=criterion, min_samples_split=min_samples_split)
            score = np.mean(cross_val_score(estimator=clf, X=x, y=y, cv=10))
            scores.append((score, (criterion, min_samples_split), clf))
    models_and_scores.append(find_hyperparameter(scores))

    # DecisionTreeClassifier:
    print('\nDecisionTreeClassifier: p=(criterion,min_samples_split)')
    print('  criterion = the function to measure the quality of a split; default=gini')
    print('  min_samples_split = min num of samples required to be at leaf; default=2')
    scores = []
    for criterion in ['gini', 'entropy']:
        for min_samples_split in range(2, 15):
            clf = DecisionTreeClassifier(criterion=criterion, min_samples_split=min_samples_split)
            score = np.mean(cross_val_score(estimator=clf, X=x, y=y, cv=10))
            scores.append((score, (criterion, min_samples_split), clf))
    models_and_scores.append(find_hyperparameter(scores))

    # KNeighborsClassifier:
    print('\nKNeighborsClassifier: p=(n_neighbors,weights)')
    print('  n_neighbors = num of neighbors; default=5')
    print('  weights = weight function used in prediction; default=uniform')
    scores = []
    for n_neighbors in range(1, 15):
        for weights in ['uniform', 'distance']:
            clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
            score = np.mean(cross_val_score(estimator=clf, X=x, y=y, cv=10))
            scores.append((score, (n_neighbors, weights), clf))
    models_and_scores.append(find_hyperparameter(scores))

    # SVC:
    print('\nSVC: p=(kernel)')
    print('  kernel = specifies the kernel type to be used in the algorithm; default=rbf')
    scores = []
    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        clf = SVC(kernel=kernel)
        score = np.mean(cross_val_score(estimator=clf, X=x, y=y, cv=10))
        scores.append((score, (kernel), clf))
    models_and_scores.append(find_hyperparameter(scores))

    return models_and_scores


########################################################################################################################
# STEP 3: train the models with all the training set (without cross validation), and check performance on the
#         validation set.
########################################################################################################################

def report_performance_confusion_matrix(y_true, y_predict, cls_name, step_num):
    conf_mat = confusion_matrix(y_true, y_predict, labels=labels)

    # stdout:
    print(pd.DataFrame(conf_mat, columns=["Predicted " + x for x in labels], index=["Actual " + x for x in labels]))

    # plot:
    classes = labels
    cm = conf_mat
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes,
           title='STEP {0}: confusion matrix of {1}'.format(step_num, cls_name), ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()


def report_performance_histogram(y_true, y_predict, cls_name, step_num):
    sample_num = y_true.count()
    print(sample_num)
    true_data = y_true.value_counts().reindex(labels, fill_value=0).sort_index()
    predict_data = y_predict.value_counts().reindex(labels, fill_value=0).sort_index()

    fig, ax = plt.subplots(figsize=(15, 7))
    index = np.arange(0, len(true_data) * 2, 2)
    bar_width = 0.8

    true_bar = plt.bar(index, true_data, bar_width, color='turquoise', label='True value')
    predict_bar = plt.bar(index + bar_width, predict_data, bar_width, color='lightcoral', label='Prediction')

    plt.xlabel('Party')
    plt.ylabel('Votes')
    plt.title('STEP {0}: histogram of {1}'.format(step_num, cls_name))
    plt.xticks(index + bar_width / 2, labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.legend()

    for rects in [true_bar, predict_bar]:
        for rect in rects:
            height = rect.get_height()
            percent = round((height / sample_num) * 100, 2)
            ax.text(rect.get_x() + rect.get_width() / 2., height,
                    "{0}\n{1}%".format(height, percent), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()


def report_performance(y_true, y_predict, cls_name, step_num):
    print("confusion matrix:")
    report_performance_confusion_matrix(y_true, y_predict, cls_name, step_num)

    print("histogram:")
    report_performance_histogram(y_true, y_predict, cls_name, step_num)

    # print(classification_report(validation_Y, prediction_y, target_names=labels))


def compare_performance(models_and_scores, training_X, training_Y, validation_X, validation_Y):
    print('\n_________________ STEP 3: check performance on validation set _________________')

    for clf, training_cv_score in models_and_scores:
        clf_name = clf.__class__.__name__
        clf.fit(training_X, training_Y)
        prediction_y = pd.Series(clf.predict(validation_X))
        print("\n---------------------- {0} ----------------------".format(clf_name))
        print("(reminder: cross_val_score on the training set: {0})".format(training_cv_score))
        report_performance(validation_Y, prediction_y, clf_name, 3)

    return None


########################################################################################################################
# MAIN:
########################################################################################################################


if __name__ == '__main__':
    labels = ["Blues", "Browns", "Greens", "Greys", "Khakis", "Oranges", "Pinks", "Purples", "Reds",
              "Turquoises", "Violets", "Whites", "Yellows"]

    # STEP 1: load the data, split to train/validation/test and save 'original_data'+'prepared_data' (according to HW2).
    # training, validation, test = prepare_data('ElectionsData.csv')  # todo: uncomment this before submission
    training, validation, test = read_sets('prepared_data')
    training_X, training_Y = training.drop(['Vote'], axis=1), training['Vote']
    validation_X, validation_Y = validation.drop(['Vote'], axis=1), validation['Vote']
    test_X, test_Y = test.drop(['Vote'], axis=1), test['Vote']

    # STEP 2: train different models with cross validation and only on the training set, in order to tune the
    #         hyper-parameters of each model (the func return list of models, after choosing the hyper-parameters).
    # models_and_scores = train_models_with_cross_validation_in_order_to_find_hyperparameters(training_X,
    #                                                                                         training_Y)  # todo: uncomment this before submission
    #
    # for m, s in models_and_scores:
    #     print(m, s)
    models_and_scores = [  # todo: delete before submission
        (RandomForestClassifier(criterion='entropy', min_samples_split=4), 0),
        (DecisionTreeClassifier(criterion='entropy', min_samples_split=6), 0),
        (KNeighborsClassifier(n_neighbors=6, weights='distance'), 0),
        (SVC(kernel='linear'), 0)]

    # STEP 3: train the models with all the training set (without cross validation), and check performance on the
    #         validation set.
    best_model = compare_performance(models_and_scores, training_X, training_Y, validation_X, validation_Y)
