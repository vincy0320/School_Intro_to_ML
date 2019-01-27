#!/usr/bin/python3
import numpy as np
import pandas as pd
import statistics as st

import naivebayes as nb
import winnow as winnow

import util as util


def normalize(feature, lower_bound = -1, upper_bound = -1):
    """
    Normalize the input feature value if it's in the range of lower bound and 
    upper bound.

    @param feature: the feature value
    @param lower_bound: the lower bound
    @param upper_bound: the upper bound
    @return 1 if the feature value is inbound. Otherwise 0.
    """
    if lower_bound < 0 and upper_bound < 0:
        raise Exception("Either lower_bound or upper_bound must be positive.")

    if lower_bound < feature < upper_bound:
        return 1
    else:
        return 0


def normalize_data(data, range_lower, range_upper, class_name):
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
    # Ge all the values for all features and classes as arrays. 
    sepal_l_all = data.iloc[(range(row_count)), 0].values
    petal_l_all = data.iloc[(range(row_count)), 2].values
    petal_w_all = data.iloc[(range(row_count)), 3].values
    class_all = data.iloc[(range(row_count)), 4].values

    # Calculate the standard deviation and the mean to be used to as the 
    # upper bound and lower bound for normalizing the data
    sepal_l_avg = st.mean(data.iloc[range_lower:range_upper,0])
    sepal_l_sd = st.stdev(data.iloc[range_lower:range_upper,0])
    sepal_l_lower_bound = sepal_l_avg - sepal_l_sd
    sepal_l_upper_bound = sepal_l_avg + sepal_l_sd

    petal_l_avg = st.mean(data.iloc[range_lower:range_upper,2])
    petal_l_sd = st.stdev(data.iloc[range_lower:range_upper,2])
    petal_l_lower_bound = petal_l_avg - petal_l_sd
    petal_l_upper_bound = petal_l_avg + petal_l_sd

    petal_w_avg = st.mean(data.iloc[range_lower:range_upper,3])
    petal_w_sd = st.stdev(data.iloc[range_lower:range_upper,3])
    petal_w_lower_bound = petal_w_avg - petal_w_sd
    petal_w_upper_bound = petal_w_avg + petal_w_sd

    sepal_l_normalized = []
    petal_l_normalized = []
    petal_w_normalized = []
    class_normalized = []

    # Loop thru each row of the data and normalize all the data. 
    for index in range(len(sepal_l_all)):
        sepal_l_normalized.append(normalize(sepal_l_all[index], sepal_l_lower_bound, sepal_l_upper_bound))
        petal_l_normalized.append(normalize(petal_l_all[index], petal_l_lower_bound, petal_l_upper_bound))
        petal_w_normalized.append(normalize(petal_w_all[index], petal_w_lower_bound, petal_w_upper_bound))
        class_normalized.append(1 if class_all[index] == class_name else 0)
    
    # return a new table with the normalized data.
    return [sepal_l_normalized, petal_l_normalized, petal_w_normalized, class_normalized]

def get_training_index():
    """
    @return The indexes of the training data used in this dataset. The indices 
            corresponds to rows. 
    """
    return list(range(0, 35)) + list(range(59, 85)) + list(range(100, 135))

def get_test_index():
    """
    @return The indexes of the test data used in this dataset. The indices 
            corresponds to rows. 
    """
    return list(range(36, 50)) + list(range(86, 100)) + list(range(136, 150))

def train_with_winnow(normalized_data, weights, alpha, theta):
    """
    Train the given normalized dataset using winnow algorithm. 
    @param normalized_data
    @param weights for the model. 
    @param alpha used to tune the model
    @param theta used as a threshold for the model.
    @return the trained weights.
    """
    training_table = util.get_training_table(normalized_data, 3, get_training_index())
    return winnow.train(training_table[0], training_table[1], weights, alpha, theta)

def train_and_test_with_winnow(data, class_names):
    """
    Train data to classify the given class names using winnow algorithm. 
    Then run tests on the test data and print the weights and the results.

    @param data unnormalized ata used for training and testing/ 
    @param class_names the name of the classes. 
    """

    alpha = 3
    init_weights = [1, 1, 1]

    # Train Class 1
    class1_theta = 0.5
    class1_normalized_data = normalize_data(data, 0, 34, class_names[0])
    class1_weights = train_with_winnow(class1_normalized_data, init_weights.copy(), alpha, class1_theta)

    # Train Class 2
    class2_theta = 0.1
    class2_normalized_data = normalize_data(data, 50, 84, class_names[1])
    class2_weights = train_with_winnow(class2_normalized_data, init_weights.copy(), alpha, class2_theta)

    # Train Class 3
    class3_theta = 0.2
    class3_normalized_data = normalize_data(data, 100, 134, class_names[2])
    class3_weights = train_with_winnow(class3_normalized_data, init_weights.copy(), alpha, class3_theta)

    # Get Class 1 Test Data
    class1_test_feature_table = util.get_test_table(class1_normalized_data, 3, get_test_index())[0]
    class1_test_classes = util.get_test_table(class1_normalized_data, 3, get_test_index())[1]
    # Get Class 2 Test Data
    class2_test_feature_table = util.get_test_table(class2_normalized_data, 3, get_test_index())[0]
    class2_test_classes = util.get_test_table(class2_normalized_data, 3, get_test_index())[1]
    # Get Class 3 Test Data
    class3_test_feature_table = util.get_test_table(class3_normalized_data, 3, get_test_index())[0]
    class3_test_classes = util.get_test_table(class3_normalized_data, 3, get_test_index())[1]

    original_indices = get_test_index()
    # Go through each line of test data and compare results.
    for index in range(len(class1_test_classes)):
        class1_features = util.get_test_features(class1_test_feature_table, index)
        class1_test_result = winnow.get_classification(class1_features, class1_weights, class1_theta)

        class2_features = util.get_test_features(class2_test_feature_table, index)
        class2_test_result = winnow.get_classification(class2_features, class2_weights, class2_theta)

        class3_features = util.get_test_features(class3_test_feature_table, index)
        class3_test_result = winnow.get_classification(class3_features, class3_weights, class3_theta)

        expected_classes = [class1_test_classes[index], class2_test_classes[index], class3_test_classes[index]]
        result_classes = [class1_test_result, class2_test_result, class3_test_result]
        util.compare_and_print(expected_classes, result_classes, original_indices, class_names, index)

    util.print_all_weights([class1_weights, class2_weights, class3_weights], class_names)


def train_and_test_with_naive_bayes(data, class_names):
    """
    Train data to classify the given class names using naive bayes algorithm. 
    Then run tests on the test data and print the results.

    @param data unnormalized ata used for training and testing/ 
    @param class_names the name of the classes. 
    """

    # Train Class 1
    class1_normalized_data = normalize_data(data, 0, 34, class_names[0])
    class1_training_table = util.get_training_table(class1_normalized_data, 3, get_training_index())
    class1_model = nb.train(class1_training_table[0], class1_training_table[1])
    # Train Class 2
    class2_normalized_data = normalize_data(data, 50, 84, class_names[1])
    class2_training_table = util.get_training_table(class2_normalized_data, 3, get_training_index())
    class2_model = nb.train(class2_training_table[0], class2_training_table[1])
    # Train Class 3
    class3_normalized_data = normalize_data(data, 100, 134, class_names[2])
    class3_training_table = util.get_training_table(class3_normalized_data, 3, get_training_index())
    class3_model = nb.train(class3_training_table[0], class3_training_table[1])

    # Get Class 1 Test Data
    class1_test_feature_table = util.get_test_table(class1_normalized_data, 3, get_test_index())[0]
    class1_test_classes = util.get_test_table(class1_normalized_data, 3, get_test_index())[1]
    # Get Class 2 Test Data
    class2_test_feature_table = util.get_test_table(class2_normalized_data, 3, get_test_index())[0]
    class2_test_classes = util.get_test_table(class2_normalized_data, 3, get_test_index())[1]
    # Get Class 3 Test Data
    class3_test_feature_table = util.get_test_table(class3_normalized_data, 3, get_test_index())[0]
    class3_test_classes = util.get_test_table(class3_normalized_data, 3, get_test_index())[1]

    original_indices = get_test_index()
    # Go through each line of test data and compare results.
    for index in range(len(class1_test_classes)):
        class1_features = util.get_test_features(class1_test_feature_table, index)
        class1_test_result = nb.get_classification(class1_features, class1_model)

        class2_features = util.get_test_features(class2_test_feature_table, index)
        class2_test_result = nb.get_classification(class2_features, class2_model)

        class3_features = util.get_test_features(class3_test_feature_table, index)
        class3_test_result = nb.get_classification(class3_features, class3_model)

        expected_classes = [class1_test_classes[index], class2_test_classes[index], class3_test_classes[index]]
        result_classes = [class1_test_result, class2_test_result, class3_test_result]
        util.compare_and_print(expected_classes, result_classes, original_indices, class_names, index)
        



def main():
    """
    Main function to kick off the trianing and testing
    """
    data = pd.read_csv('./iris.data', header = None)

    class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    print("\n-- Train and Test with Winnow --\n")
    train_and_test_with_winnow(data, class_names)

    print("\n-- Train and Test with Naive Bayes --\n")
    train_and_test_with_naive_bayes(data, class_names)
    

if __name__ == '__main__':
    main()
    
