#!/usr/bin/python3
import util
import random
import data_provider as dp
import k_means


# [Extra credits: K-Means]
def __get_train_set_from_clusters(data, query_index, clusters, class_index):
    """
    For the given query point, find the cluster it belongs to and use the 
    cluster as the train_set.
    """

    centroids = clusters["means"]
    return centroids
    
    # 
    # query_point = util.get_point_without_class(data, query_index, class_index)
    # cluster_index = k_means.find_cluster(query_point, centroids)
    # return clusters["clusters_by_index"][cluster_index]

# [Extra credits: K-Means]
def __get_distances_to_centroids(data, query_index, class_index, centroids):
    """
    Get distance to centroids
    """

    query_point = util.get_point_without_class(data, query_index, class_index)
    distances = []
    for index in range(len(centroids)):
        dist = util.distance(query_point, centroids[index])
        distances.append({
            "index": index,
            "dist": dist
        })
    return distances

# [Extra credits: K-Means]
def __get_closest_point_to_centroid(data, centroid_index, class_index, clusters):
    """
    Get the closest point to a given centroid index and centroids
    """

    cluster = clusters["clusters_by_index"][centroid_index]
    centroid = clusters["means"][centroid_index]

    cur_min_dist = float('inf')
    cur_min_dist_index = 0
    # Check the point x against each centroid in means
    for index in cluster:
        point = util.get_point_without_class(data, index, class_index)
        dist = util.distance(centroid, point)
        if dist < cur_min_dist:
            cur_min_dist_index = index
            cur_min_dist = dist
    return cur_min_dist_index


def __get_k_nearest_points(data, class_index,
        dist_matrix, query_index, train_set, k, clusters = None):
    """
    Get the k nearest point for the given query index
    """

    new_train_set = train_set.copy()
    distances = []
    if clusters != None:
        # Get the new train set from the clusters, which are the centroids.
        new_train_set = __get_train_set_from_clusters(
            data, query_index, clusters, class_index)

        distances = __get_distances_to_centroids(
            data, query_index, class_index, new_train_set)

    else :
        # Gets the distances from the point at query_index to all points at 
        # indices in the train_set
        distances = util.get_distances_to_indices(
            dist_matrix, query_index, new_train_set)

    # Sort the distance list by distance ascendantly
    sorted_distances = sorted(distances, key=lambda x: x["dist"])
    # Return the first k elements in the list, which are the k smallest item
    return sorted_distances[:k]
    



def condense(data, train_set, class_index, dist_matrix):
    """
    Condense the train set using the Condensed KNN method
    """

    condensed_train_set = []
    # Make a copy of the trainset so the original trainset stays put
    train_set_copy = train_set.copy()
    # Shuffle the train_set so that we can select the points randomly.
    random.shuffle(train_set_copy)

    done = False
    while not done:
        start_size = len(condensed_train_set)

        # Loop through all points in the shuffled train set
        for index in range(len(train_set_copy)):
            train_index = train_set_copy[index]
            # calculate the distances from this point in train set to all the 
            # points in the condensed train set.
            distances = util.get_distances_to_indices(
                dist_matrix, train_index, condensed_train_set)
            
            should_add_to_condensed = len(distances) == 0
            if not should_add_to_condensed:
                # Find the closest point in the condensed train set
                min_distance = min(distances, key=lambda x: x["dist"])
                # Compare the classes of two points
                classification_condensed = util.get_point_class(
                    data, min_distance["index"], class_index)
                classification_train = util.get_point_class(
                    data, train_index, class_index)
                should_add_to_condensed = (
                    classification_condensed != classification_train)
            
            if should_add_to_condensed:
                # If classes are different, add this index to condensed set
                condensed_train_set.append(train_index)
                # and remove it from train set
                del train_set_copy[index]
                # break so that we can go through this process again with the 
                # newly updated condensed set
                break
        
        # If the size didn't change, then it means for all the points in the 
        # train set, their closest point in condensed train set is the same as
        # their class.
        done = len(condensed_train_set) == start_size
        
    return condensed_train_set



def get_classification(data, class_index, query_index, train_set, k, 
        dist_matrix, clusters = None):
    """
    Classify the given query index
    """

    k_nearest = __get_k_nearest_points(data, class_index,
        dist_matrix, query_index, train_set, k, clusters)
    # Loop through the k smallest item and count the occurance of a class
    votes = {}
    for item in k_nearest:
        index = item["index"]
        if clusters != None:
            # When using the cluster approach, get the closest point to the 
            # centroid to be the reference point.
            index = __get_closest_point_to_centroid(
                data, item["index"], class_index, clusters)
        classification = util.get_point_class(data, index, class_index)
        if classification in votes:
            votes[classification] += 1
        else:
            votes[classification] = 0

    # Get the class by the max vote
    return max(votes, key=votes.get)


def get_regression(data, class_index, query_index, train_set, k, dist_matrix, 
        clusters = None):
    """
    Get the regression value for the given query index
    """

    k_nearest = __get_k_nearest_points(data, class_index, 
        dist_matrix, query_index, train_set, k, clusters)
    sum = 0
    for item in k_nearest:
        index = item["index"]
        if clusters != None:
            # When using the cluster approach, get the closest point to the 
            # centroid to be the reference point.
            index = __get_closest_point_to_centroid(
                data, item["index"], class_index, clusters)
        sum += util.get_point_class(data, index, class_index)
    return sum / float(len(k_nearest))



# Test
# if __name__ == "__main__":
#     main()