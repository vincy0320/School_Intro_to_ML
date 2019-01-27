#!/usr/bin/python3
import argparse

import data_set as ds
import decision_tree as dt
import partition
import statistics as stats
import util


def parse_args():
    """
    Parse the arguments from command line
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--abalone", help="Specify the input file for the "
                        "abalone dataset")
    parser.add_argument("--car", help="Specify the input file for the "
                        "car evaluation dataset")
    parser.add_argument("--segmentation", help="Specify the input file for the "
                        "image segmentation dataset")
    parser.add_argument("--machine", help="Specify the input file for the"
                        "computer hardware dataset")
    parser.add_argument("--forestfires", help="Specify the input file for the"
                        "forest fires dataset")
    parser.add_argument("--wine", help="Specify the input file for the"
                        "wine quality dataset")
    parser.add_argument("--test", help="Specify the input file for the"
                            "test quality dataset")
    parser.add_argument("--test2", help="Specify the input file for the"
                            "test quality dataset")
    # parser.add_argument("--rbf", help="Use RBF Neuro Network", 
    #                     action="store_true")
    
    return parser.parse_args()


def verify_args(args):
    """
    Verify dataset args are all legal args
    """
    
    if (args.abalone or args.segmentation or args.car or args.test or args.test2 or
        args.machine or args.forestfires or args.wine):
        return
    else:
        raise Exception("Error: Unrecognized dataset")


def cross_validate(dataset, cv_partitions, validation_partition):
    """
    Perform cross validation with the given cv_partitions and use the given 
    validation_partion as the validation set
    """

    results_tree_completion = []
    results_tree_pruned = []

    for index in range(len(cv_partitions)):
        util.print_separator()
        # Rotate and use each partition as the test set once
        test_set = cv_partitions[index]
        # Rotate and use the other partitions as the training set
        train_set = []
        for partition in cv_partitions[:index] + cv_partitions[index + 1:]:
            train_set += partition

        # Use the train_set to construct a tree
        tree = dt.DecisionTree(dataset, train_set)
        partition_result = classify_and_validate(tree, dataset, test_set)
        # record performance for each train + test set pair by calculating 
        # the rate of correctness ratio
        correctness = stats.mean(partition_result)
        results_tree_completion.append(correctness)

        print("Partition,Tree,Correctness,Depth,Size,Correctness Improvement over validation")
        util.print_tree_result(index, "Complete", correctness, tree, "N/A")

        # TODO: Implement pruning
        prune_result = reduced_error_prune(
            tree, dataset, validation_partition) 
        # print(prune_result)
        partition_result = classify_and_validate(tree, dataset, test_set)
        correctness = stats.mean(partition_result)
        results_tree_pruned.append(correctness)
        util.print_tree_result(index, "Pruned  ", correctness, tree, 
            str(round(prune_result[0] - prune_result[1], 4)))

    # calculate the average correctness across all train + test set pair.
    return {
        "completion": stats.mean(results_tree_completion),
        "pruned": stats.mean(results_tree_pruned),
    }


def classify_and_validate(tree, dataset, test_set):
    """
    Classify and validate the given tree with the test set, return the result
    of each test example in an array
    """

    results = []
    for test_index in test_set:
        prediction = tree.classify(dataset.row(test_index))[-1]
        result = validate_classification(dataset, test_index, prediction)
        results.append(result)
    return results


def validate_classification(dataset, query_index, prediction):
    """
    Validate a classification result.
    """
    if dataset.get_class(query_index) == prediction:
        return 1
    else:
        return 0

def reduced_error_prune(tree, dataset, validation_set):
    """
    Apply reduced error pruning on the given tree using the given validation set
    """

    # Base result without pruning
    baseline_result = classify_and_validate(tree, dataset, validation_set)
    baseline_avg = stats.mean(baseline_result)

    best_avg = baseline_avg
    done = False

    iteration = 1
    while not done: 
        iteration += 1

        # Prune a descendent one
        pruned_child = tree.prune_next()
        if pruned_child == tree:
            pruned_child.is_pruned = False
            done = True
            break

        if pruned_child is None:
            # Done if nothing can be pruned
            done = True
            break

        # Evaluate correctness
        new_result = classify_and_validate(tree, dataset, validation_set)
        new_avg = stats.mean(new_result)
        if new_avg >= best_avg:
            # Keep the prune is perf is better.
            best_avg = new_avg
            pruned_child.can_prune = True
        else: 
            # Otherwise, unprune the descendent and mark it unprunable
            pruned_child.is_pruned = False
            pruned_child.can_prune = False

    return (best_avg, baseline_avg)


def run_classification(dataset):
    """
    Execute a classification task using the given configuration
    """

    # Define cross validate set and validation set
    validation_percentage = 0.1
    cross_validation_folds = 5
    cross_validate_percentage = (1 - validation_percentage) / cross_validation_folds

    proportions = [validation_percentage]
    proportions += [cross_validate_percentage] * cross_validation_folds

    # Define partitions used for cross validation and separate validation
    partitions = partition.get_stratified_partitions_by_percentage(
        dataset, proportions)
    validation_partition = partitions[0]
    cv_partitions = partitions[1: cross_validation_folds + 1]
    
    validation_result = cross_validate(
        dataset, cv_partitions, validation_partition)

    util.print_separator()
    print("Overall results for Cross Validation")
    print("Completion,Pruned,Pruning Improvment")
    result_completion = round(validation_result["completion"], 4)
    result_pruned = round(validation_result["pruned"], 4)
    print(",".join([
        str(result_completion),
        str(result_pruned),
        str(round(result_pruned - result_completion, 4))
    ]))


def main():
    """ 
    The main function of the program
    """

    args = parse_args()
    verify_args(args)

    # Get data from the given args
    dataset = ds.DataSet(args)
    print("\nClassification Dataset:", dataset.display_name)

    run_classification(dataset)

    # Tests
    # tree = dt.DecisionTree(dataset, list(range(dataset.size)))
    # path = tree.classify(dataset.row(4))
    # print(path)
        
   
    print("\n")



if __name__ == "__main__":
    main()