#!/usr/bin/python3
import math

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


def get_clusters_with_index(data, clusters_by_index):
    """
    Get the clusters using the given index representation
    """

    clusters = []
    for cluster_indices in clusters_by_index:
        cluster = []
        for index in cluster_indices:
            cluster.append(data[index])
        clusters.append(cluster)
    return clusters


def process_data_for_features(data, selected_features):
    """
    Process the given data with the given selected features by creating a new
    table with each row contains only the selected feature
    """

    processed = []
    for row in data:
        processed_row = []
        for index in range(len(selected_features)):
            if selected_features[index] == 1:
                processed_row.append(row[index])
        processed.append(processed_row)
    return processed