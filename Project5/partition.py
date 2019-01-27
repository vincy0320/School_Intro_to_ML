#!/usr/bin/python3
import random
import math
import util


def __get_class_indices_dict(data, class_index):
    """
    Get a dictionary whose key is the class name and the value is a list of 
    row indices in the given dataset that has that class name.
    """

    return util.get_class_indices_dict(data, class_index)


def get_stratified_partitions(dataset, n):
    """
    Get n-fold stratified partition.
    """

    percentage = round(1 / n, 2)
    percentages = [percentage] * n
    partitions = get_stratified_partitions_by_percentage(dataset, percentages)

    # verify_partition(data, partitions, class_index)

    return partitions


def get_stratified_partitions_by_percentage(dataset, percentages):
    """
    Get stratified partitions based on the given percentages
    """

    # Verity percentage sums to 1
    total = sum(percentages)
    if round(total) != 1:
        raise Exception("Error: Percentages must sum to 1")

    class_indices_dict = util.get_class_indices_dict(dataset)
    count_dict = util.get_class_count_dict(class_indices_dict)

    partitions = []
    for index in range(len(percentages)):
        percentage = percentages[index]
        part = []
        for class_name in class_indices_dict:
            if index == len(percentages) - 1:
                part += class_indices_dict[class_name]
            else:
                count = round(count_dict[class_name] * percentage)
                selection = util.select_random_n_from_all(
                    class_indices_dict[class_name], count)
                part += selection
        partitions.append(part)

    # verify_partition(data, partitions, class_index)

    return partitions


def get_random_partitions(data, n):
    """
    Get n partitions randomly from the given data
    """

    indices = list(range(len(data)))
    partition_size = math.floor(len(indices) / n)

    partitions = []
    while n > 0:
        part = util.select_random_n_from_all(indices, partition_size) 
        partitions.append(part)
        n -= 1
    partitions[-1] += indices
    return partitions




# Test 
# def verify_partition(data, partitions, class_index):
#     for part in partitions:
#         print("part:")
#         count_map = {}
#         for index in part:
#             class_name = data[index][class_index]
#             if class_name in count_map:
#                 count_map[class_name] += 1
#             else:
#                 count_map[class_name] = 0
#         print(count_map)



