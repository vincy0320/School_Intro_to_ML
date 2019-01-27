#!/usr/bin/python3
import pandas as pd
import numpy as np
import statistics as st

import naivebayes as nb
import winnow as winnow

import util as util


def get_training_index():
    """
    @return The indexes of the training data used in this dataset. The indices 
            corresponds to rows. 
    """
    return list(range(0, 7)) + list(range(10, 17)) + list(range(20, 27)) + list(range(30, 42))

def get_test_index():
    """
    @return The indexes of the test data used in this dataset. The indices 
            corresponds to rows. 
    """
    return list(range(7, 10)) + list(range(17, 20)) + list(range(27, 30)) + list(range(42, 47))


def normalize(column):
    """
    Normalize the input feature value if it's in the range of lower bound and 
    upper bound.

    @param feature: the feature value
    @param lower_bound: the lower bound
    @param upper_bound: the upper bound
    @return 1 if the feature value is inbound. Otherwise 0.
    """
    value_set = set(column)
    unique_count = len(value_set)
    if unique_count == 1:
        # skip everything in this column. 
        return []
    elif unique_count == 2:
        zero = list(value_set)[0]
        one = list(value_set)[1]
        normalized_column = []
        for value in column:
            normalized_column.append(1 if value == one else 0)
        return [normalized_column]
    else: 
        all_values = list(value_set)
        normalized_column = []

        # expand into multiple columns 
        for index in range(len(all_values)):
            normalized_column.append([])

        for value in column:
            for index in range(len(all_values)):
                normalized_column[index].append(1 if value == all_values[index] else 0)
        
        return normalized_column


def normalize_data(data, class_name):
    """
    Convert the entire data table into 0s and 1s based on the given range and
    class name. The range is used to tell which rows' values should be used
    to calculate means and standard deviation. The calculated value will be
    used to normalize the original values to 0s or 1s. 

    The class name is used to normalize the class name colum. If the name is
    a match, then it's normalized to 1, otherwise, 0. 

    NOTE: This function implicitly assumes that the rows in the range matches
    the given class name. 

    @param data: The unnormalized data read from csv using panda. 
    @param range_lower: the lower range for the section of data to be used.
    @param range_upper: the uppper range for the section of data to be used.
    @param class_name: the name of the classes in this dataset. 
    @return normalized dataset. 
    """
    row_count = len(data.index)
    col_count = len(data.columns)
    normalized_data = []
    for index in range(col_count - 1):
        feature_list = data.iloc[(range(row_count)), index].values
        normalized_data += normalize(feature_list)
    
    normalized_class_list = []

    class_list = data.iloc[(range(row_count)), col_count - 1].values
    for value in class_list:
        normalized_class_list.append(1 if value == class_name else 0)
    normalized_data.append(normalized_class_list)
    
    return normalized_data

    
def train_with_winnow(normalized_data, weights, alpha, theta):
    """
    Train the given normalized dataset using winnow algorithm. 
    @param normalized_data
    @param weights for the model. 
    @param alpha used to tune the model
    @param theta used as a threshold for the model.
    @return the trained weights.
    """
    training_table = util.get_training_table(normalized_data, len(normalized_data) - 1, get_training_index())
    return winnow.train(training_table[0], training_table[1], weights, alpha, theta)

def train_and_test_with_winnow(data, class_names):
    """
    Train data to classify the given class names using winnow algorithm. 
    Then run tests on the test data and print the weights and the results.

    @param data unnormalized ata used for training and testing/ 
    @param class_names the name of the classes. 
    """
    alpha = 7

    # Train Class 1
    class1_theta = 0.5
    class1_normalized_data = normalize_data(data, class_names[0])
    init_weights = [1] * (len(class1_normalized_data) - 1)
    class1_weights = train_with_winnow(class1_normalized_data, init_weights.copy(), alpha, class1_theta)

    # Train Class 2
    class2_theta = 0.5
    class2_normalized_data = normalize_data(data, class_names[1])
    init_weights = [1] * (len(class2_normalized_data) - 1)
    class2_weights = train_with_winnow(class2_normalized_data, init_weights.copy(), alpha, class2_theta)

    # Train Class 3
    class3_theta = 0.5
    class3_normalized_data = normalize_data(data, class_names[2])
    init_weights = [1] * (len(class3_normalized_data) - 1)
    class3_weights = train_with_winnow(class3_normalized_data, init_weights.copy(), alpha, class3_theta)

    # Train Class 4
    class4_theta = 0.5
    class4_normalized_data = normalize_data(data, class_names[3])
    init_weights = [1] * (len(class4_normalized_data) - 1)
    class4_weights = train_with_winnow(class4_normalized_data, init_weights.copy(), alpha, class4_theta)

    class_index = len(class1_normalized_data) - 1
    # Get Class 1 Test Data
    class1_test_feature_table = util.get_test_table(class1_normalized_data, class_index, get_test_index())[0]
    class1_test_classes = util.get_test_table(class1_normalized_data, class_index, get_test_index())[1]
    # Get Class 2 Test Data
    class2_test_feature_table = util.get_test_table(class2_normalized_data, class_index, get_test_index())[0]
    class2_test_classes = util.get_test_table(class2_normalized_data, class_index, get_test_index())[1]
    # Get Class 3 Test Data
    class3_test_feature_table = util.get_test_table(class3_normalized_data, class_index, get_test_index())[0]
    class3_test_classes = util.get_test_table(class3_normalized_data, class_index, get_test_index())[1]
    # Get Class 4 Test Data
    class4_test_feature_table = util.get_test_table(class4_normalized_data, class_index, get_test_index())[0]
    class4_test_classes = util.get_test_table(class4_normalized_data, class_index, get_test_index())[1]

    original_indices = get_test_index()
    # Go through each line of test data and compare results.
    for index in range(len(class1_test_classes)):
        class1_features = util.get_test_features(class1_test_feature_table, index)
        class1_test_result = winnow.get_classification(class1_features, class1_weights, class1_theta)

        class2_features = util.get_test_features(class2_test_feature_table, index)
        class2_test_result = winnow.get_classification(class2_features, class2_weights, class2_theta)

        class3_features = util.get_test_features(class3_test_feature_table, index)
        class3_test_result = winnow.get_classification(class3_features, class3_weights, class3_theta)

        class4_features = util.get_test_features(class4_test_feature_table, index)
        class4_test_result = winnow.get_classification(class4_features, class4_weights, class4_theta)

        expected_classes = [
            class1_test_classes[index], class2_test_classes[index], 
            class3_test_classes[index], class4_test_classes[index]
        ]
        result_classes = [
            class1_test_result, class2_test_result, class3_test_result, class4_test_result
        ]
        util.compare_and_print(expected_classes, result_classes, original_indices, class_names, index)

    util.print_all_weights([class1_weights, class2_weights, class3_weights, class4_weights], class_names)

def train_and_test_with_naive_bayes(data, class_names):
    """
    Train data to classify the given class names using naive bayes algorithm. 
    Then run tests on the test data and print the results.

    @param data unnormalized ata used for training and testing/ 
    @param class_names the name of the classes. 
    """
    # Train class 1
    class1_normalized_data = normalize_data(data, class_names[0])
    class_index = len(class1_normalized_data) - 1
    class1_training_table = util.get_training_table(class1_normalized_data, class_index, get_training_index())
    class1_model = nb.train(class1_training_table[0], class1_training_table[1])

    # Train class 2
    class2_normalized_data = normalize_data(data, class_names[1])
    class2_training_table = util.get_training_table(class2_normalized_data, class_index, get_training_index())
    class2_model = nb.train(class2_training_table[0], class2_training_table[1])

    # Train class 3
    class3_normalized_data = normalize_data(data, class_names[2])
    class3_training_table = util.get_training_table(class3_normalized_data, class_index, get_training_index())
    class3_model = nb.train(class3_training_table[0], class3_training_table[1])

    # Train class 4
    class4_normalized_data = normalize_data(data, class_names[3])
    class4_training_table = util.get_training_table(class4_normalized_data, class_index, get_training_index())
    class4_model = nb.train(class4_training_table[0], class4_training_table[1])

    # Get Class 1 Test Data
    class1_test_feature_table = util.get_test_table(class1_normalized_data, class_index, get_test_index())[0]
    class1_test_classes = util.get_test_table(class1_normalized_data, class_index, get_test_index())[1]
    # Get Class 2 Test Data
    class2_test_feature_table = util.get_test_table(class2_normalized_data, class_index, get_test_index())[0]
    class2_test_classes = util.get_test_table(class2_normalized_data, class_index, get_test_index())[1]
    # Get Class 3 Test Data
    class3_test_feature_table = util.get_test_table(class3_normalized_data, class_index, get_test_index())[0]
    class3_test_classes = util.get_test_table(class3_normalized_data, class_index, get_test_index())[1]
    # Get Class 4 Test Data
    class4_test_feature_table = util.get_test_table(class4_normalized_data, class_index, get_test_index())[0]
    class4_test_classes = util.get_test_table(class4_normalized_data, class_index, get_test_index())[1]


    original_indices = get_test_index()
    # Go through each line of test data and compare results.
    for index in range(len(class1_test_classes)):
        class1_features = util.get_test_features(class1_test_feature_table, index)
        class1_test_result = nb.get_classification(class1_features, class1_model)

        class2_features = util.get_test_features(class2_test_feature_table, index)
        class2_test_result = nb.get_classification(class2_features, class2_model)

        class3_features = util.get_test_features(class3_test_feature_table, index)
        class3_test_result = nb.get_classification(class3_features, class3_model)

        class4_features = util.get_test_features(class4_test_feature_table, index)
        class4_test_result = nb.get_classification(class4_features, class4_model)

        expected_classes = [
            class1_test_classes[index], class2_test_classes[index], 
            class3_test_classes[index], class4_test_classes[index]
        ]
        result_classes = [
            class1_test_result, class2_test_result, class3_test_result, class4_test_result
        ]
        util.compare_and_print(expected_classes, result_classes, original_indices, class_names, index)


def main():
    """
    Main function to kick off the trianing and testing
    """
    data = pd.read_csv('./soybean-small.data', header = None)

    class_names = ["D1", "D2", "D3", "D4"]

    print("\n-- Train and Test with Winnow --\n")
    train_and_test_with_winnow(data, class_names)

    print("\n-- Train and Test with Naive Bayes --\n")
    train_and_test_with_naive_bayes(data, class_names)
    return


if __name__ == '__main__':
    main()
    
