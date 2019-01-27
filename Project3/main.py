#!/usr/bin/python3
import argparse

import data_provider as dp
import k_nearest_neighbor as knn
import k_means
import partition
import radial_basis_function_network as rbfn
import util


def parse_args():
    """
    Parse the arguments from command line
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--ecoli", help="Specify the input file for the "
                        "ecoli dataset")
    parser.add_argument("--segmentation", help="Specify the input file for the "
                        "segmentation dataset")
    parser.add_argument("--machine", help="Specify the input file for the"
                        "machine dataset")
    parser.add_argument("--forestfires", help="Specify the input file for the"
                        "forestfires dataset")
    parser.add_argument("--rbf", help="Use RBF Neuro Network", 
                        action="store_true")
    return parser.parse_args()


def verify_args(args):
    """
    Verify dataset args are all legal args
    """
    
    if args.ecoli or args.segmentation:
        return
    elif args.machine or args.forestfires:
        return
    else:
        raise Exception("Error: Unrecognized dataset")



def cross_validate(partitions, data, class_index, dist_matrix, k, 
        is_classification = True, should_condense = False, k_means_k = 0, 
        rbf = None, rbf_k_means_k = 0):
    """
    Perform 5-fold cross validation with the given partition. Other parameters
    are for configuring which algorithm to run for the validation
    """

    # print("Train and Test with k:", k)
    # A list of 1s and 0s where 1 means the classification is corret.
    results = []
    for index in range(len(partitions)):
        parition_result = []
        # Rotate and use each partition as the test set once
        test_set = partitions[index]
        # Rotate and use the other partitions as the training set
        train_set = []
        for partition in partitions[:index] + partitions[index + 1:]:
            train_set += partition

        clusters = None
        if k_means_k > 0:
            # [Extra credits: K-Means]
            clusters = k_means.get_clusters(
                data, train_set, k_means_k, class_index)
        elif should_condense:
            train_set = knn.condense(data, train_set, class_index, dist_matrix)
        
        # [Extra credits: RBF Network]
        if rbf is not None:
            rbf.train(train_set, is_classification)

        # Loop through each example in the test_set
        for test_index in test_set:
            result = -1
            if is_classification:
                # Get classification for each 
                classification = None
                if rbf is not None:
                    classification = rbf.get_prediction(test_index)
                else:
                    classification = knn.get_classification(data, class_index, 
                    test_index, train_set, k, dist_matrix, clusters)
                # Get validation result for each 
                result = validate_classification(
                    data, class_index, test_index, classification)
            else:
                # Get regression value for each 
                regression = knn.get_regression(data, class_index, test_index, 
                    train_set, k, dist_matrix, clusters)
                # Get validation result for each 
                result = get_regression_squared_error(
                    data, class_index, test_index, regression)
            parition_result.append(result)
        # record performance for each train + test set pair by calculating 
        # the rate of correctness
        avg_result = sum(parition_result) / float(len(parition_result))
        results.append(avg_result)

    # calculate the average correctness across all train + test set pair.
    return sum(results) / float(len(results))


def validate_classification(data, class_index, index, classification):
    """
    Validate a classification result.
    """
    if util.get_point_class(data, index, class_index) == classification:
        return 1
    else:
        return 0

def get_regression_squared_error(data, class_index, index, regression_value):
    """
    Get the squared for a regression result.
    """

    truth = util.get_point_class(data, index, class_index)
    square_error = (truth - regression_value)**2
    return square_error


def run_classification(data, class_index, dist_matrix, k_values,k_means_k = 0, 
    rbf = None, rbf_k_means_k = 0):
    """
    Execute a classification task using the given configuration
    """

    # Run the same configuration multiple rounds to get better average
    round = 1
    max_round = 1
    while round <= max_round:
        print("- Round", round)
        partitions = partition.get_balanced_partitions(data, class_index, 5)

        for k in k_values:
            results = []
            print("-- k = " + str(k))
            result_not_condensed = cross_validate(
                partitions, data, class_index, dist_matrix, k, True, False)
            print("-- Not Condensed    ", result_not_condensed)
            results.append(result_not_condensed)

            result_condensed = cross_validate(
                partitions, data, class_index, dist_matrix, k, True, True)
            print("-- Condensed        ", result_condensed)
            results.append(result_condensed)

            # [Extra credits: K-Means]
            result_k_means = -1
            if k_means_k > 0:
                result_k_means = cross_validate(partitions, data, class_index, 
                    dist_matrix, k, True, False, k_means_k)
                print("-- " + str(k_means_k) + "-Means          ", result_k_means)
                results.append(result_k_means)
                
            # Consolidate and print results
            results.sort()
            
            comparison = []
            for result in results:
                if result == result_not_condensed:
                    comparison.append("KNN")
                elif result == result_condensed:
                    comparison.append("Condensed KNN")
                else:
                    comparison.append("KMeans")
            print("-- Accuracy:", " < ".join(map(str, comparison)))

        round += 1
    
        # [Extra credits: RBF 10%]
        result_rbf = -1
        if rbf is not None and result_rbf < 0:
            result_rbf = cross_validate(partitions, data, class_index, 
                dist_matrix, -1, True, False, 0, rbf)
            print("-- RBF 10%        ", result_rbf)

        # [Extra credits: RBF KMeans]
        result_rbf_k_means = -1
        if rbf and rbf_k_means_k > 0 and result_rbf_k_means < 0:
            result_rbf_k_means = cross_validate(partitions, data, class_index, 
                dist_matrix, -1, True, False, 0, rbf, rbf_k_means_k)
            print("-- RBF " + str(rbf_k_means_k) + "-Means    ", 
                result_rbf_k_means)


def run_regression(data, class_index, dist_matrix, k_values, k_means_k = 0, 
    rbf = None, rbf_k_means_k = 0):
    """
    Execute a regression task using the given configuration
    """
    
    # Get the variance of the data set that can be used determine how good a
    # regression result is.
    varaince = util.get_varaince(data, list(range(len(data))), class_index)

    # Run the same configuration multiple rounds to get better average
    round = 1
    max_round = 1
    while round <= max_round:
        print("- Round", round)
        # Get 5 partitions, but not bala
        partitions = partition.get_random_partitions(data, 5)

        for k in k_values:
            print("-- k = " + str(k))
            result = cross_validate(
                partitions, data, class_index, dist_matrix, k, False)
            print("-- Result", result)
            print("-- Diff variance", result - varaince)

            # [Extra credits: K-Means]
            if k_means_k > 0:
                k_mean_result = cross_validate(partitions, data, class_index, 
                    dist_matrix, k, False, False, k_means_k)
                print("-- KMeans K =", k_means_k)
                print("-- KMeans Result", k_mean_result)
                print("-- KMeans Diff variance", k_mean_result - varaince)

                comparison = []
                if k_mean_result < result:
                    comparison = ["KNN", "KMeans"]
                else:
                    comparison = ["KMeans", "KNN"]
                print("-- Accuracy:", " < ".join(map(str, comparison)))

        round += 1

        # [Extra credits: RBF 10%]
        result_rbf = -1
        if rbf is not None and result_rbf < 0:
            result_rbf = cross_validate(partitions, data, class_index, 
                dist_matrix, -1, False, False, 0, rbf)
            print("-- RBF 10% Result", result_rbf)
            print("-- Diff variance", result_rbf - varaince)

        # [Extra credits: RBF KMeans]
        result_rbf_k_means = -1
        if rbf and rbf_k_means_k > 0 and result_rbf_k_means < 0:
            result_rbf_k_means = cross_validate(partitions, data, class_index, 
                dist_matrix, -1, False, False, 0, rbf, rbf_k_means_k)
            
            print("-- RBF " + str(rbf_k_means_k) + "-Means   ", 
                result_rbf_k_means)
            print("-- Diff variance", result_rbf_k_means - varaince)


def main():
    """ 
    The main function of the program
    """

    args = parse_args()
    verify_args(args)

    print("\nDataset:", dp.get_dataset_name_for_display(args))

    # Get data from the given args
    dataInfo = dp.get_data(args)
    data = dataInfo[0]
    class_index = dataInfo[1]
    dist_matrix = util.get_distance_matrix(data, class_index)
    rbf = None
    if args.rbf:
        rbf = rbfn.RBFN(data, class_index, dist_matrix)

    # Define some tuneable values base on dataset
    k_means_k = dp.get_k_means_k(args)
    knn_ks = dp.get_knn_ks(args)
    rbf_k_means_k = dp.get_rbf_k_means_k(args)

    if dp.is_classification(args):
        print("- Type:", "Classification")
        run_classification(data, class_index, dist_matrix, knn_ks, k_means_k, 
            rbf, rbf_k_means_k)
    else:
        # Everything else is regression, otherwise verify_args would've failed.
        print("- Type:", "Regression")
        run_regression(data, class_index, dist_matrix, knn_ks, k_means_k, rbf, 
            rbf_k_means_k)
   
    print("\n")

if __name__ == "__main__":
    main()