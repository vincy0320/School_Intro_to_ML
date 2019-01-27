#!/usr/bin/python3
import argparse
import cross_validate as cv
import data_set as ds
import data_set_info as dsi
import feedforward_neural_network as fnn 
import output_util
import partition
import radial_basis_function_network as rbfn
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


def cross_validate(dataset, validator, args):
    """
    Perform cross validation with the given cv_partitions and use the given 
    validation_partion as the validation set
    """

    results_fn0 = []
    results_fn1 = []
    results_fn2 = []
    results_rbfn = []
    
    for index in range(validator.size):
        cv_sets = validator.get_sets(index)
        test_set = cv_sets["test"]
        train_set = cv_sets["train"]

        # Train and Test Feedforward Neural Network with 0 hidden layer.
        learn_rate = dsi.get_learning_rate(args, 0)
        hidden_layers = dsi.get_hidden_layer_design(args, 0)
        fn0_model = fnn.FeedforwardNeuralNetwork(dataset, train_set, learn_rate, hidden_layers)
        fn0_paris = []
        fn0_error_rate = classify_and_validate(fn0_model, dataset, test_set, fn0_paris)
        results_fn0.append(fn0_error_rate)

        # Train and Test Feedforward Neural Network with 1 hidden layer.
        learn_rate = dsi.get_learning_rate(args, 1)
        hidden_layers = dsi.get_hidden_layer_design(args, 1)
        fn1_model = fnn.FeedforwardNeuralNetwork(dataset, train_set, learn_rate, hidden_layers)
        fn1_paris = []
        fn1_error_rate = classify_and_validate(fn1_model, dataset, test_set, fn1_paris)
        results_fn1.append(fn1_error_rate)
        
        # Train and Test Feedforward Neural Network with 2 hidden layer.
        learn_rate = dsi.get_learning_rate(args, 2)
        hidden_layers = dsi.get_hidden_layer_design(args, 2)
        fn2_model = fnn.FeedforwardNeuralNetwork(dataset, train_set, learn_rate, hidden_layers)
        fn2_paris = []
        fn2_error_rate = classify_and_validate(fn2_model, dataset, test_set, fn2_paris)
        results_fn2.append(fn2_error_rate)

        # Train and test RBFN
        rbfn_model = rbfn.RadialBasisFunctionNetwork(
            dataset, train_set, dsi.get_selectivity_for_rbfn(args)) 
        rbfn_pairs = []
        rbfn_error_rate = classify_and_validate(rbfn_model, dataset, test_set, rbfn_pairs)
        results_rbfn.append(rbfn_error_rate)

        if index == 4:
            prefix = "Fold " + str(index)
            print_results(dataset, "FN0", fn0_error_rate, fn0_paris, prefix, fn0_model)
            print_results(dataset, "FN1", fn1_error_rate, fn1_paris, prefix, fn1_model)
            print_results(dataset, "FN2", fn2_error_rate, fn2_paris, prefix, fn2_model)
            print_results(dataset, "RBFN", rbfn_error_rate, rbfn_pairs, prefix, rbfn_model)

    # calculate the average error_rate across all train + test set pair.
    print_overall_results(dataset, {
        "FN0": results_fn0,
        "FN1": results_fn1,
        "FN2": results_fn2,
        "RBFN": results_rbfn
    })

def print_results(dataset, model_name, error_rate, model_pairs, prefix, model):
    """
    Print results
    """

    file_name = "_".join([dataset.display_name, model_name, "output.csv"])
    printer = output_util.Printer(file_name)
    printer.print_dataset_name(dataset)
    
    printer.print_csv_row(["Model", "Error Rate"])
    printer.print_csv_row([model_name, error_rate])
    printer.print_csv_row([])
    printer.print_csv_row(["Fold of Model", "Category", "Values"])
    printer.print_pairs(model_pairs, prefix + " " + model_name)
    printer.print_csv_row([])
    model.print_model(printer)
    printer.print_csv_row([])

def print_overall_results(dataset, results):

    file_name = "_".join([dataset.display_name, "overall_output.csv"])
    printer = output_util.Printer(file_name)
    printer.print_dataset_name(dataset)
    printer.print_results(results)


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


def run_classification(dataset, args):
    """
    Execute a classification task using the given configuration
    """

    round = 0
    while round < 1:
        validator = cv.CrossValidator(dataset, 5)
        cross_validate(dataset, validator, args)

        print("")
        round += 1


def main():
    """ 
    The main function of the program
    """
    
    # Start
    args = parse_args()
    verify_args(args)

    # Get data from the given args
    dataset = ds.DataSet(args)
    # output_util.print_dataset_name(dataset)
    run_classification(dataset, args)

    # End



if __name__ == "__main__":
    main()