#!/usr/bin/python3


def get_data_from_indices(list, indices):
    """
    Get the data from the list for the given indices
    """
    return [list[index] for index in indices]

def get_training_table(normalized_data, class_column_index, training_indices):
    """
    Get the training table from normzlied data wnd indices.
    """
    training_feature_table = []
    training_classes = []
    for index in range(len(normalized_data)):
        normalized = get_data_from_indices(normalized_data[index], training_indices)
        if index == class_column_index:
            training_classes = (normalized)
        else:
            training_feature_table.append(normalized)
    return (training_feature_table, training_classes)

def get_test_table(normalized_data, class_column_index, test_indices):
    """
    Get the test table from normzlied data wnd indices.
    """
    test_feature_table = []
    test_classes = []
    for index in range(len(normalized_data)):
        normalized = get_data_from_indices(normalized_data[index], test_indices)
        if index == class_column_index:
            test_classes = normalized
        else:
            test_feature_table.append(normalized)
    return (test_feature_table, test_classes)


def get_test_features(test_feature_table, index): 
    """
    Get the test features from normzlied data wnd indices.
    """
    features = []
    for feature_list in test_feature_table:
        features.append(feature_list[index])
    return features


def print_test_result(index, match, predicted_result_indices, actual_index, class_names):
    """
    Print test result with the given info.
    """
    message = "Index " + str(index)
    message += " Match: " + str(match)
    
    message += " Classified: " 
    if len(predicted_result_indices) == 0:
        message += "None "
    else:
        for index in predicted_result_indices:
            message += str(index) + "-" + class_names[index] + " "

    message += "Actual: " + str(actual_index) + "-" + class_names[actual_index]
    print(message)


def compare_and_print(expected_classes, result_classes, original_indices, class_names, index):
    """
    Compare the given expected classes and the result classes and print the results.
    """
    if expected_classes.count(1) != 1:
        raise Excpetion("Error: Incorrect classification for input classes")

    if result_classes.count(1) == 0:
        print_test_result(original_indices[index], False, [], expected_classes.index(1), class_names)
    elif result_classes.count(1) == 1:
        match = result_classes.index(1) == expected_classes.index(1)
        print_test_result(original_indices[index], match, [result_classes.index(1)], expected_classes.index(1), class_names)
    else:
        result_indices = []
        for result_index in range(len(result_classes)):
            if result_classes[result_index] == 1:
                result_indices.append(result_index)
        print_test_result(original_indices[index], False, result_indices, expected_classes.index(1), class_names)


def print_all_weights(weights_list, class_names):
    """
    Print the given weights used for the model for the class name. 
    """
    print('\n')
    for index in range(len(class_names)):
        print(class_names[index] + " is trained with weights " + str(weights_list[index]))
        print('\n')