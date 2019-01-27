#!/usr/bin/python3
import util


def __average_distance(x, cluster):
    """
    Calculate the average distance from x to all other objects in the cluster.
    If x is not in the cluster or cluster is empty, then raise exception.
    """

    m = len(cluster)
    if m == 0:
        raise Exception("Error: Cluster length must not be 0")
    elif x not in cluster:
        raise Exception("Error: x must be part of the cluster")
    else:
        # A list of distance to x
        dist_list = []
        for point in cluster:
            if point != x:
                dist_list.append(util.distance(x, point))
        length = len(dist_list)
        if length > 0:
            return sum(dist_list) / length
        else:
            return 0


def __min_average_distance_non_containing_cluster(x, clusters):
    """
    Given x and a list of clusters, find the average distance from x to all of
    the points in that other cluster that doesn't contain x. Then calculate the
    minimum of such values over all of the clusters.
    """

    # A list of average distance to x
    dist_list = []
    for cluster in clusters:
        found = x in cluster
        if found:
            # Skip clusters that contains x
            continue

        sum = 0
        avg = 0
        m = len(cluster)
        if m == 0:
            # Empty cluster has distance as 0
            avg = 0
        else:
            # Calculate the avg distance for non-empty cluster.
            for index in range(m):
                sum += util.distance(x, cluster[index])
            avg = sum / m
        dist_list.append(avg)

    return min(dist_list)


def get_silhouette_coefficient(x, clusters):
    """
    Get the Silhouette Coefficient for x given the clusters.
    """
    
    containing_cluster = None
    non_containing_clusters = []
    for cluster in clusters:
        if x in cluster:
            containing_cluster = cluster
        else:
            non_containing_clusters.append(cluster)

    a = __average_distance(x, containing_cluster)
    b = __min_average_distance_non_containing_cluster(
        x, non_containing_clusters)

    silhouette_coefficient = (b - a) / max([a, b])
    return silhouette_coefficient


def evaluate(clusters):
    """
    Evaluate the overall clustering of the dataset given the clusters by finding
    the average silhouette for the dataset
    """

    # A list to hold all silhouette coefficients for the points in the dataset
    sc_list = []
    for cluster in clusters:
        for x in cluster:
            # Calculate the silhouette coefficient for each point and save to
            # the list
            sc = get_silhouette_coefficient(x, clusters)
            sc_list.append(sc)

    # Calculate the average of the silhouette coefficients
    length = len(sc_list)
    if length > 0:
        return sum(sc_list) / length
    else:
        return 0


# Tests
# if __name__ == '__main__':

#     cluster = [1, 2, 3, 4]
#     clusters = [cluster, [5, 6], [7, 8]]
#     print(__average_distance(1, cluster))
#     print(__min_average_distance_non_containing_cluster(1, clusters))
#     print(get_silhouette_coefficient(1, clusters))
#     print(evaluate(clusters))