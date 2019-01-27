#!/usr/bin/python3
import random
import math
import util


def __get_class_distribution(data, class_index):
    """
    Get the distribution for the class in the dataset so that it can be used
    to create a evenly distributed 5 fold partitions
    """

    distribution = {}
    for index in range(len(data)):
        class_name = data[index][class_index]
        if class_name in distribution:
            distribution[class_name].append(index)
        else:
            distribution[class_name] = [index]
    return distribution


def get_balanced_partitions(data, class_index, n):
    """
    Get 5-fold datasets with balanced partition.
    """

    fold_count = n
    distribution = __get_class_distribution(data, class_index)
    
    fold_count_dict = {}
    for key in distribution:
        fold_count_dict[key] = math.floor(len(distribution[key]) / fold_count)

    partitions = []
    count = fold_count
    while count > 0:
        indices = []
        for key in distribution:
            result = util.get_random_values(
                distribution[key], fold_count_dict[key])
            selected_indices = result[0]
            distribution[key] = result[1]
            indices += selected_indices
        partitions.append(indices)
        count -= 1
    
    # If there is anything left in the distribution, then spread it out to all
    # the dataset
    for key in distribution:
        partition_index = 0
        for index in distribution[key]:
            partition_index = partition_index % len(partitions)
            partitions[partition_index].append(index)
            partition_index += 1

    # verify_partition(data, partitions, class_index)

    return partitions


def get_random_partitions(data, n):
    """
    Get n partitions randomly from the given list
    """

    indices = list(range(len(data)))
    partition_size = math.floor(len(indices) / n)

    partitions = []
    while n > 0:
        part = util.get_random_values(indices, partition_size)[0]
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



