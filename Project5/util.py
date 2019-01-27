#!/usr/bin/python3
import math
import random
import time

# Class attribute related helper functions


def get_class_indices_dict(dataset, filter_indices = []):
    """
    Get a dictionary whose key is the class name and the value is a list of 
    row indices in the given dataset that has that class name. 
    When filter indices are provided, skip the row indices that are not in the
    filter indices.
    When filter indices are not provided, don't skip any row.
    """

    filter_indices_set = set(filter_indices)
    if len(filter_indices_set) == 0:
        filter_indices_set = set(range(dataset.size))

    distribution = {}
    for index in range(dataset.size):
        if index not in filter_indices_set:
            continue

        class_name = dataset.get_class(index)
        if class_name in distribution:
            distribution[class_name].append(index)
        else:
            distribution[class_name] = [index]
    return distribution


def get_class_count_dict(class_indices_dict):
    """
    Get a dictionary whose key is the class name and the value is the occurrence
    of that class.
    """

    count_dict = {}
    for class_name in class_indices_dict:
        count_dict[class_name] = len(class_indices_dict[class_name])
    return count_dict


def get_unique_value_in_column(data, column_index):
    """
    Get the unique values in a column.
    """

    values = set()
    for row in data:
        if row[column_index] not in values:
            values.add(row[column_index])
    return list(values) 


def get_majority_class(dataset, filter_indices = []):
    """
    Get the majority number of class
    """

    class_indices_dict = get_class_indices_dict(dataset, filter_indices)
    class_count_dict = get_class_count_dict(class_indices_dict)

    # Get the class by the max vote
    return max(class_count_dict, key=class_count_dict.get)





def get_attribute_dict_list(dataset, filter_indices = [], skip_feature_indices = {}):
    """
    Get a list of dictionaries. 
    At each index, the value is a dictionary whose
        - keys are the unique values (unique_v) in the column and 
        - values are a dictionary whose 
            - keys are the unique value of the class column and 
            - values are the occurrence of that class for the column's value is
              unique_v.
    """

    filter_indices_set = set(filter_indices)
    if len(filter_indices_set) == 0:
        filter_indices_set = set(range(dataset.size))

    # Initialize array with empty dictionaries
    attribute_dict_list = []
    for index in range(dataset.columns):
        attribute_dict_list.append({})

    # Loop through all rows in the dataset
    for index in filter_indices_set:
        if index not in filter_indices_set:
            continue

        # for each row, loop through each feature
        row = dataset.row(index)
        for col_index in range(len(row) - 1):

            attr_dict = attribute_dict_list[col_index]
            class_value = dataset.get_class(index)
            
            if (dataset.is_feature_categorical(col_index) 
                or dataset.is_class_index(col_index)):

                if col_index in skip_feature_indices:
                    continue

                value = row[col_index]
                if value not in attr_dict:
                    attr_dict[value] = {}
                
                if class_value not in attr_dict[value]:
                    attr_dict[value][class_value] = 1
                else:
                    attr_dict[value][class_value] += 1

    return attribute_dict_list


def select_random_n_from_all(all_values, count):
    """
    Get count number of random values from all values
    """

    selected = []
    while count > 0:
        index = random.randint(0, len(all_values) - 1)
        selected.append(all_values[index])
        del all_values[index]
        count -= 1
    return selected

def get_index_of_min_value(values):
    """
    Get the index of the min value in the given list of values
    """

    min_index = -1
    min_value = float("inf")
    for index in range(len(values)):
        val = values[index]
        if val <= min_value:
            min_value = val
            min_index = index
    return min_index


def get_index_of_max_value(values):
    """
    Get the index of the max value in the given list of values
    """

    max_index = -1
    max_value = float("-inf")
    for index in range(len(values)):
        val = values[index]
        if val > max_value:
            max_value = val
            max_index = index
    return max_index
