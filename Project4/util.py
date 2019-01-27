#!/usr/bin/python3
import math
import random
import time


# Distance related helper functions

def distance(a, b):
    """
    Calculates the Euclidean distance between vector a and vector b
    """

    if a is None or b is None:
        raise Exception("Error: Neither a nor b can be none.")
    elif len(a) != len(b):
        raise Exception("Error: a and b should have the same dimension")

    sum = 0
    for index in range(len(a)):
        sum += (a[index] - b[index])**2

    return round(math.sqrt(sum), 2)


def get_distance_matrix(data, class_index = -2):
    """
    Create a distance metrix that contains the distance between any two points 
    in the dataset
    """

    if class_index > -2: 
        data = get_data_without_class(data, class_index)
    
    dist_matrix = []
    size = len(data)
    for i in range(size):
        row = []
        for j in range(size):
            dist = 0
            if i != j:
                dist = distance(data[i], data[j])
            row.append(dist)
        dist_matrix.append(row)
    return dist_matrix


def get_distance(dist_matrix, index_a, index_b):
    """
    Get distance with dist matrix
    """

    return dist_matrix[index_a][index_b]


def get_distances_to_indices(dist_matrix, query_index, indices):
    """
    Get a list of object that represents the distance from the point at 
    query_index to all the points in the indices list.
    """


    # Loop through each point in the train_set and calculate the distance to
    # the query_point
    distances = []
    for index in indices:
        if index != query_index:
            # Calcuate distance
            dist = get_distance(dist_matrix, query_index, index)
            # Record in the distantce list
            distances.append({
                "index": index,
                "dist": dist
            })

    return distances



# Class attribute related helper functions


def get_data_without_class(data, class_index):
    """
    Process the data read from the dataset for feature selection by removing
    the class attribute column
    """

    without_class = []
    for index in range(len(data)):
        # Remove the class column
        without_class.append(get_point_without_class(data, index, class_index))
    return without_class


def get_point_without_class(data, index, class_index):
    """
    Get the point without a class
    """

    if class_index != -1:
        return data[index][0:class_index] + data[index][class_index + 1:]
    else:
        return data[index][0:-1]


def get_point_class(data, index, class_index):
    """
    Get the class of a point
    """
    if data and data[index]:
        return data[index][class_index]
    return None


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

def get_unique_value_in_column(data, column_index):
    """
    Get the unique values in a column.
    """

    values = set()
    for row in data:
        if row[column_index] not in values:
            values.add(row[column_index])
    return list(values) 

def get_one_hot_encoding(value, unique_values):
    """
    Get one-hot encoding for a value, given the list of unique values. 
    """

    return list(map(lambda x: 1 if value == x else 0, unique_values))

def get_majority_class(dataset, filter_indices = []):


    class_indices_dict = get_class_indices_dict(dataset, filter_indices)
    class_count_dict = get_class_count_dict(class_indices_dict)

    # Get the class by the max vote
    return max(class_count_dict, key=class_count_dict.get)

def get_class_count_dict(class_indices_dict):
    """
    Get a dictionary whose key is the class name and the value is the occurrence
    of that class.
    """

    count_dict = {}
    for class_name in class_indices_dict:
        count_dict[class_name] = len(class_indices_dict[class_name])
    return count_dict


def get_feature_indices(data, class_index):
    """
    Get the indices (columns) that are features.
    """

    row = data[0]
    indices = list(range(len(row)))
    feature_indices = list(filter(lambda x: x != class_index, indices))
    return feature_indices


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

    binary_splits = dataset.binary_splits

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
            else: 
                splits = binary_splits[col_index]
                for split_index in range(len(splits)):
                    if split_index not in attr_dict:
                        attr_dict[split_index] = {}
                    
                    if (col_index in skip_feature_indices 
                        and split_index in skip_feature_indices[col_index]):
                        continue
                    
                    value = ">" 
                    if row[col_index] <= splits[split_index]:
                        value = "<="

                    if value not in attr_dict[split_index]:
                        attr_dict[split_index][value] = {}

                    if class_value not in attr_dict[split_index][value]:
                        attr_dict[split_index][value][class_value] = 1
                    else:
                        attr_dict[split_index][value][class_value] += 1

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

# Print related helper functions

def print_separator():
    """
    Print a separator
    """
    print("---------------------------\n")

def print_tree_result(partition_num, name, correctness, tree, prune_diff):
    """
    Print a tree result.
    """
    print(",".join([
        str(partition_num),
        name, 
        str(round(correctness, 4)), 
        str(tree.get_depth()), 
        str(tree.get_size()),
        prune_diff
    ]))