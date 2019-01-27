#!/usr/bin/python3

# This file is not runnable for now.

import hac
import k_means
import silhouette_coefficient as sc
import util

def __get_clusters_by_index(processed_data, k, use_hac, h):
    """
    Get the index representation of the clusters
    """

    if use_hac:
        if h < 0:
            raise Exception("Error: h must be positive when using HAC")
        return hac.get_clusters(processed_data, k)
    elif k > 0:
        clustering_results = k_means.get_clusters(processed_data, k)
        return clustering_results["clusters_by_index"]
    else:
        raise Exception("Error: k must be positive when using k-means")


def stepwise_forward_selection(data, k = -1, use_hac = False, h = -1):
    """
    Perform Stepwise Forward Selection to select features, using Silhouette
    Coefficient to evaluate performance.
    """

    dimension = len(data[0])
    # Intialize a list of 1s to indicate all featuers are available for 
    # selection
    feature_set = [1] * dimension
    # Intialize a list of 0s to indicate none of the features has been selected
    selected_features = [0] * dimension 
    
    base_perf = float("-inf")
    clusters_by_index = []

    while feature_set.count(1) > 0:
        best_perf = float("-inf")
        best_feature_index = -1

        for feature_index in range(len(feature_set)):
            if feature_set[feature_index] == 0:
                # Feature not available for selection
                continue 
            
            # Try a new feature for clustering
            selected_features[feature_index] = 1
            processed_data = util.process_data_for_features(data, selected_features)
            clusters_by_index = __get_clusters_by_index(processed_data, k, use_hac, h)

            # Evaluate the clustering result
            if len(clusters_by_index) >= 2:
                clusters = util.get_clusters_with_index(data, clusters_by_index)
                curr_perf = sc.evaluate(clusters)
                # Keep track of the best_perf so far
                if curr_perf > best_perf:
                    best_perf = curr_perf
                    best_feature_index = feature_index
            # Remove the feature used above so that we can try the next feature
            selected_features[feature_index] = 0
        
        if best_perf > base_perf:
            # Keep track of the overall best performance
            base_perf = best_perf
            # Mark the feature as unavailable for selection
            feature_set[best_feature_index] = 0
            # Mark the feature as a selected feature
            selected_features[best_feature_index] = 1
        else:
            # Break when we see perf is going down
            break

    return selected_features
