#!/usr/bin/python3
import math
import random
import util


def find_cluster(x, means):
    """
    Find the cluster that x belongs to by finding the argmin of the distance
    between x and the corresponding mean of the cluster
    """

    cur_min_dist = float('inf')
    cur_min_dist_index = 0
    # Check the point x against each centroid in means
    for index in range(len(means)):
        dist = util.distance(x, means[index])
        if dist < cur_min_dist:
            cur_min_dist_index = index
            cur_min_dist = dist
    return cur_min_dist_index


def __get_cluster_mean(cluster, dimension):
    """
    Get the mean for all the points in the cluster
    """

    if len(cluster) == 0:
        return [0] * dimension

    # sum all the points in the cluster
    sum_x = [0] * len(cluster[0])
    for i in range(len(cluster)):
        for j in range(len(cluster[i])):
            sum_x[j] += cluster[i][j]

    mean_x = list(map(lambda x: round(x/len(cluster), 2), sum_x))
    return mean_x


def __has_mean_changed(old, new):
    """
    Check if the mean has changed
    """

    if len(old) != len(new):
        return True

    # Check if all the elements in the old mean equals the new mean
    for i in range(len(old)):
        for j in range(len(old[i])):
            if old[i][j] != new[i][j]:
                return True

    return False


def __create_initial_means(data, valid_index_set, k, class_index):
    """
    Create an initial set of means by randomly select k points from the dataset
    """

    means = []
    selected_indices = set()
    index = 0
    while index < k:
        # intialize means randomly in a list
        random_index = -1
        while random_index < 0:
            candidate = random.randint(0, len(data) - 1)
            if candidate in valid_index_set:
                random_index = candidate

        if random_index not in selected_indices:
            selected_indices.add(random_index)
            means.append(util.get_point_without_class(
                data, random_index, class_index))
            index += 1
    return means


def get_clusters(data, valid_indices, k, class_index):
    """
    Perform k-means clustering using data and cluster into k clusters
    """

    valid_index_set = set(valid_indices)
    dimension = len(data[0]) - 1
    clusters = []
    clusters_by_index = []

    means = __create_initial_means(data, valid_index_set, k, class_index)

    done = False
    while not done:
        # intialize clusters to contain k empty list, each is a cluster
        clusters = []
        clusters_by_index = []
        index = k
        while index > 0:
            clusters.append([])
            clusters_by_index.append([])
            index -= 1

        # Do the clustering base on the current means
        for index in range(len(data)):
            if index not in valid_index_set:
                # skip the indices that are not valid
                continue
            row = util.get_point_without_class(data, index, class_index)
            cluster_index = find_cluster(row, means)
            clusters[cluster_index].append(row)
            clusters_by_index[cluster_index].append(index)

        # Create a new list of means
        new_means = []
        for cluster in clusters:
            mean = __get_cluster_mean(cluster, dimension)
            new_means.append(mean)

        means_changed = __has_mean_changed(means, new_means)
        means = new_means
        done = not means_changed

    return {
        "clusters_by_index": clusters_by_index,
        "means": means
    }


# Tests
# if __name__ == '__main__':
#     a = [1, 1, 1]
#     b = [0, 0, 0]
#     c = [2, 5, 8]
    # print(dist(a, b))
    # print(__get_cluster_mean([a, b, c]), 3)
