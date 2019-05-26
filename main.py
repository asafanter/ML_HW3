from prepare_data import *


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

if __name__ == '__main__':
    # STEP 1: load the data, split to train/validation/test and save 'original_data'+'prepared_data' (according to HW2):
    # training, validation, test = prepare_data('ElectionsData.csv')  # todo: uncomment it before submission
    training, validation, test = read_sets('prepared_data')
