#!/usr/bin/python3
import math
import random


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


def get_varaince(data, indices, class_index):
    """
    Get the variance of a dataset
    """

    index_set = set(indices)
    numbers = []
    for row_index in range(len(data)):
        if row_index in index_set:
            numbers.append(
                get_point_class(data, row_index, class_index))

    mean = sum(numbers) / float(len(numbers))

    mean_square_errors = list(
        map(lambda x: (x - mean)**2, numbers))
    return sum(mean_square_errors) / float(len(mean_square_errors))


def get_random_values(all_values, count):
    """
    Get count number of random values from all values
    """

    selected = []
    while count > 0:
        index = random.randint(0, len(all_values) - 1)
        selected.append(all_values[index])
        del all_values[index]
        count -= 1
    return (selected, all_values)

def vector_scalar_product(scalar, vector):
    """
    Perform a scalar multiplication on a vector
    """

    output = []
    for index in range(len(vector)):
        output.append(scalar * vector[index])
    return output
    
def vector_subtraction(a, b):
    """
    Subtracts given vector a by given vector b.
    """

    if len(a) != len(b):
        raise Exception("Error: Vector subtraction require same length")

    output = []
    for index in range(len(a)):
        output.append(a[index] - b[index])
    return output
    

def get_weighted_sum(weights, vector):
    """
    Get the weighted sum.
    """
    if len(weights) != len(vector):
        raise Exception(
            "Error: Weights and Neuron responses should have same length")

    sum = 1 # w_0 is set to 1
    for index in range(len(vector)):
        sum += weights[index] * vector[index]
    return sum


# Print related helper functions

def print_separator():
    """
    Print a separator
    """
    print("---------------------------\n")