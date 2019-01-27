#!/usr/bin/python3
import argparse
import pandas
import random

import feature_selection as fs
import hac
import k_means
import silhouette_coefficient as sc
import spambase
import util


def parseArgs():
    """
    Parse the arguments from command line
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--glass", help="Specify the input file for the "
                        "Glass dataset")
    parser.add_argument("--iris", help="Specify the input file for the "
                        "Iris dataset")
    parser.add_argument("--spambase", help="Specify the input file for the"
                        "Spambase dataset")
    parser.add_argument("--hac", help="Use HAC for Clustering. "
                        "Otherwise, default to k-means", action="store_true")
    return parser.parse_args()

    
def get_input_file_path(args):
    """
    Get the input file path from the given arguments.
    """

    input_file_path = ""
    if args.glass:
        input_file_path = args.glass
    elif args.iris:
        input_file_path = args.iris
    elif args.spambase:
        input_file_path = args.spambase
    else:
        raise Exception("Error: Unrecognized dataset")
    return input_file_path

def get_proposed_cluster_count(args):
    """
    Gets proposed cluster count for each dataset
    """
    if args.glass:
        return 7 # 7 classes in the Glass dataset
    elif args.iris:
        return 3 # 3 classes in the Iris dataset
    elif args.spambase:
        return 2 # 2 classes in the Spambase dataset
    else:
        raise Exception("Error: Unrecognized dataset")


def get_proposed_hac_height(args):
    """
    Get the proposed hac height for each dataset
    """
    if args.glass:
        return 0.999
    elif args.iris:
        return 0.15
    elif args.spambase:
        return 1
    else:
        raise Exception("Error: Unrecognized dataset")


def get_dataset_name_for_display(args):
    """
    Get the display name for dataset
    """
    
    if args.glass:
        return "Glass"
    elif args.iris:
        return "Iris"
    elif args.spambase:
        return "Spambase"
    else:
        raise Exception("Error: Unrecognized dataset")


def process_data(args, data):
    """
    Process the data read from the dataset for feature selection by removing
    the class attribute column
    """

    if not args.glass and not args.iris and not args.spambase and False:
        raise Exception("Error: Unrecognized dataset")

    processed = []
    for row in data:
        # Remove the class column
        processed.append(row[0:-1])
    return processed


def print_separator():
    """
    Print a separator
    """
    print("---------------------------\n")


def display_results(args, features, clusters, silhouette_coefficient, centroids, height):
    """
    Display results using the given arguments
    """

    print("\n")
    print("Dataset: ", get_dataset_name_for_display(args))
    print("Features: ", features)
    clustering_method = "k-means"
    if args.hac:
        clustering_method = "HAC"
    print("Clustering method: ", clustering_method)
    print("Sihouette Coefficient: ", round(silhouette_coefficient, 2))
    print_separator()

    for index in range(len(clusters)):
        cluster = clusters[index]
        cluster_size = len(cluster)
        centroid = []
        print("Cluster ", index)
        print("Size: ", cluster_size)
        if not args.hac:
            centroid = centroids[index]
            print("Centroid: ", centroid)

        point_closest_to_centroid_count = 0

        class_dict = {}
        for point in cluster:
            # Count the class name
            class_name = point[-1]
            if not class_name in class_dict:
                class_dict[class_name] = 1
            else:
                class_dict[class_name] += 1

            if not args.hac:
                # Get the point with only the selected features
                point_by_feature = []
                for f_index in range(len(features)):
                    if features[f_index] == 1:
                        point_by_feature.append(point[f_index])
                # print("Point: ", point)
                dist_to_centroid = util.distance(point_by_feature, centroid)
                #print("Distance to centroid: ", dist_to_centroid)
                
                dist_to_other_centroids = []
                for m_index in range(len(centroids)):
                    if m_index != index:
                        dist = util.distance(
                            point_by_feature, centroids[m_index])
                        dist_to_other_centroids.append(dist)
                # print("Distance to other centroids: ", dist_to_other_centroids)
                if dist_to_centroid <= min(dist_to_other_centroids):
                    point_closest_to_centroid_count += 1

        if not args.hac and cluster_size != 0:
            print("Point in right cluster rate:", 
                str(point_closest_to_centroid_count * 100 / cluster_size) + "%")

        print("Cluster classes: ")
        for key in class_dict.keys():
            print("- Class", key, ":", class_dict[key])

        print_separator()


def main():
    """ 
    The main function of the program
    """

    args = parseArgs()
    input_file_path = get_input_file_path(args)
    data = None
    try:
        data = pandas.read_csv(input_file_path, header = None)
    except: 
        raise Exception(
            "Error: Unable to open the input file, please try again.")

    # process input data
    data = data.values.tolist()
    processed_data = process_data(args, data)

    # Prepare data for feature selection
    proposed_cluster_count = get_proposed_cluster_count(args)
    data_for_feature_selection = processed_data
    if args.spambase:
        # Spambase is a huge dataset, so we only take a subset
        data_for_feature_selection = spambase.get_feature_selection_data(
            processed_data, 50)
    
    # Perform feature selection
    proposed_hac_height = get_proposed_hac_height(args)
    selected_features = fs.stepwise_forward_selection(
        data_for_feature_selection, proposed_cluster_count, 
        args.hac, proposed_hac_height)

    # Use the selected features to cluster the entire dataset
    featured_data = util.process_data_for_features(
        processed_data, selected_features)

    clusters_by_index = []
    centroids = []
    if args.hac:
        clusters_by_index = hac.get_clusters(featured_data, proposed_cluster_count)
    else:
        clustering_results = k_means.get_clusters(
            featured_data, proposed_cluster_count)
        clusters_by_index = clustering_results["clusters_by_index"]
        centroids = clustering_results["means"]

    # Display the results
    clusters = util.get_clusters_with_index(data, clusters_by_index)
    clusters_without_class = util.get_clusters_with_index(
        processed_data, clusters_by_index)
    s_coefficient = sc.evaluate(clusters_without_class)

    display_results(args, selected_features, clusters, s_coefficient, centroids, 
        proposed_hac_height)
    print_separator()


if __name__ == "__main__":
    main()