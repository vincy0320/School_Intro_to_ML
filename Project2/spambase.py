#!/usr/bin/python3
import random


def __get_feature_selection_index(data, n):
    """
    Randomly select n index in the dataset for feature seelction. 
    This is specific to spambase because other dataset doesn't have size issue
    """

    if n == len(data):
        return range(0, len(data))

    # randomly select n samples for feature selection. 
    indices = set()
    while n > 0:
        random_index = random.randint(0, len(data) - 1)
        if random_index not in indices:
            indices.add(random_index)
            n -= 1
    return indices


def get_feature_selection_data(data, n):
    """
    Get the feature selection data for Spambase with n rows.
    This is specific to spambase because other dataset doesn't have size issue
    """

    # Get a subset of indices ot use for feature selection
    feature_selection_indices = __get_feature_selection_index(data, n)

    # Get a susbset of rows in the dataset using the indices above
    feature_selection_data = []
    for index in range(len(data)):
        if index in feature_selection_indices:
            feature_selection_data.append(data[index])
    return feature_selection_data