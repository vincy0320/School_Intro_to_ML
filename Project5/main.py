#!/usr/bin/python3
import argparse
import cross_validate as cv
import data_set as ds
import data_set_info as dsi
import logistic_regression as lr
import naive_bayes as nb
import output_util as output
import partition
import statistics as stats
import util


def parse_args():
    """
    Parse the arguments from command line
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--breast_cancer", help="Specify the input file for the "
                        "breast cancer dataset")
    parser.add_argument("--glass", help="Specify the input file for the "
                        "glass evaluation dataset")
    parser.add_argument("--iris", help="Specify the input file for the "
                        "iris dataset")
    parser.add_argument("--soybean", help="Specify the input file for the"
                        "soybean dataset")
    parser.add_argument("--vote", help="Specify the input file for the"
                        "vote dataset")
    
    return parser.parse_args()


def verify_args(args):
    """
    Verify dataset args are all legal args
    """
    
    if not dsi.is_classification(args):
        raise Exception("Error: Unrecognized dataset")


def cross_validate(dataset, validator):
    """
    Perform cross validation with the given cv_partitions and use the given 
    validation_partion as the validation set
    """

    results_naive_bayes = []
    results_logistic_regression = []
    
    for index in range(validator.size):
        cv_sets = validator.get_sets(index)
        test_set = cv_sets["test"]
        train_set = cv_sets["train"]

        # Train Naive Bayes and Test
        nb_model = nb.NaiveBayes(dataset, train_set)
        nb_pairs = []
        nb_error_rate = classify_and_validate(nb_model, dataset, test_set, nb_pairs)
        results_naive_bayes.append(nb_error_rate)

        # Train Logistic Regression and Test
        lr_model = lr.LogisticRegression(dataset, train_set)
        lr_pairs = []
        lr_error_rate = classify_and_validate(lr_model, dataset, test_set, lr_pairs)
        results_logistic_regression.append(lr_error_rate)

        prefix = "Fold " + str(index)

        output.print_csv_row(["Model", "Error Rate"])
        output.print_csv_row(["NB", nb_error_rate])
        output.print_csv_row(["LR", lr_error_rate])
        output.print_pairs(nb_pairs, prefix + " NB")
        output.print_pairs(lr_pairs, prefix + " LR") 

        nb_model.print_model()
        lr_model.print_model()

    # calculate the average error_rate across all train + test set pair.
    output.print_results({
        "NB": results_naive_bayes,
        "LR": results_logistic_regression
    })


def classify_and_validate(model, dataset, test_set, pairs):
    """
    Classify and validate the given tree with the test set, return the result
    of each test example in an array
    """

    results = []
    for test_index in test_set:
        prediction = model.classify(dataset.row(test_index))
        result = classification_error(dataset, test_index, prediction, pairs)
        results.append(result)
    
    avg_error_rate = round(stats.mean(results), 3)
    return avg_error_rate


def classification_error(dataset, query_index, prediction, pairs):
    """
    Calculate the classification error. 1 is mismatch. 0 is match.
    """
    truth = dataset.get_class(query_index)
    pairs.append([truth, prediction])
    is_equal = (truth == prediction)
    if is_equal:
        return 0
    else:
        return 1


def run_classification(dataset):
    """
    Execute a classification task using the given configuration
    """

    round = 0
    while round < 1:
        validator = cv.CrossValidator(dataset, 5)
        cross_validate(dataset, validator)

        print("")
        round += 1


def main():
    """ 
    The main function of the program
    """

    output.print_separator()
    
    # Start
    args = parse_args()
    verify_args(args)

    # Get data from the given args
    dataset = ds.DataSet(args)
    output.print_dataset_name(dataset)
    run_classification(dataset)

    # End
    output.print_separator()



if __name__ == "__main__":
    main()