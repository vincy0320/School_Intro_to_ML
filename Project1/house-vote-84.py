#!/usr/bin/python3
import numpy as np
import pandas as pd
import statistics as st

import naivebayes as nb
import winnow as winnow

import util as util

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

    normalized_class_list = []
    class_list = data.iloc[(range(row_count)), 0].values
    for value in class_list:
        normalized_class_list.append(1 if value == class_name else 0)
    normalized_data.append(normalized_class_list)

    for index in range(1, col_count):
        feature_list = data.iloc[(range(row_count)), index].values
        normalized_data += normalize(feature_list)
    
    return normalized_data

def get_training_index():
    """
    @return The indexes of the training data used in this dataset. The indices 
            corresponds to rows. 
    """
    return list(range(0, 305))

def get_test_index():
    """
    @return The indexes of the test data used in this dataset. The indices 
            corresponds to rows. 
    """
    return list(range(305, 435))

def train_with_winnow(normalized_data, weights, alpha, theta):
    """
    Train the given normalized dataset using winnow algorithm. 
    @param normalized_data
    @param weights for the model. 
    @param alpha used to tune the model
    @param theta used as a threshold for the model.
    @return the trained weights.
    """
    training_table = util.get_training_table(normalized_data, 0, get_training_index())
    return winnow.train(training_table[0], training_table[1], weights, alpha, theta)

def train_and_test_with_winnow(data, class_names):
    """
    Train data to classify the given class names using winnow algorithm. 
    Then run tests on the test data and print the weights and the results.

    @param data unnormalized ata used for training and testing/ 
    @param class_names the name of the classes. 
    """
    alpha = 2

    # Train Class
    class_theta = 0.5
    class_normalized_data = normalize_data(data, class_names[0])
    init_weights = [1] * (len(class_normalized_data) - 1)
    class_weights = train_with_winnow(class_normalized_data, init_weights.copy(), alpha, class_theta)

    # Get Class Test Data
    class_index = 0
    class_test_feature_table = util.get_test_table(class_normalized_data, class_index, get_test_index())[0]
    class_test_classes = util.get_test_table(class_normalized_data, class_index, get_test_index())[1]

    original_indices = get_test_index()

    # Go through each line of test data and compare results.
    for index in range(len(class_test_classes)):
        class_features = util.get_test_features(class_test_feature_table, index)
        result_class = winnow.get_classification(class_features, class_weights, class_theta)        
        expected_class = class_test_classes[index]
        matched = result_class == expected_class
        util.print_test_result(original_indices[index], matched, [result_class], expected_class, class_names)

    util.print_all_weights([class_weights, class_weights], class_names)

def train_and_test_with_naive_bayes(data, class_names):
    """
    Train data to classify the given class names using naive bayes algorithm. 
    Then run tests on the test data and print the results.

    @param data unnormalized ata used for training and testing/ 
    @param class_names the name of the classes. 
    """
    # Train data
    class_normalized_data = normalize_data(data, class_names[0])
    class_training_table = util.get_training_table(class_normalized_data, 0, get_training_index())
    class_model = nb.train(class_training_table[0], class_training_table[1])

    # Get Class Test Data
    class_index = 0
    class_test_feature_table = util.get_test_table(class_normalized_data, class_index, get_test_index())[0]
    class_test_classes = util.get_test_table(class_normalized_data, class_index, get_test_index())[1]

    original_indices = get_test_index()
    # Go through each line of test data and compare results.
    for index in range(len(class_test_classes)):
        class_features = util.get_test_features(class_test_feature_table, index)
        result_class = nb.get_classification(class_features, class_model)        
        expected_class = class_test_classes[index]
        matched = result_class == expected_class
        util.print_test_result(original_indices[index], matched, [result_class], expected_class, class_names)

def main():
    """
    Main function to kick off the trianing and testing
    """
    data = pd.read_csv('./house-votes-84.data', header = None)

    class_names = ["republican", "democrat"]

    print("\n-- Train and Test with Winnow --\n")
    train_and_test_with_winnow(data, class_names)

    print("\n-- Train and Test with Naive Bayes --\n")
    train_and_test_with_naive_bayes(data, class_names)
    

if __name__ == '__main__':
    main()
    
