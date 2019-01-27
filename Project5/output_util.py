#!/usr/bin/python3
import statistics as stats

def print_separator():
    """
    Print a separator
    """
    print("---------------------------")

def print_empty_line():
    """
    Print an empty line
    """
    print("")

def print_dataset_name(dataset):
    """
    Print the name of a dataset.
    """
    line = "Dataset:"
    prefix = "Classification"
    if not dataset.is_classification:
        prefix = "Regression"

    print(prefix, line, dataset.display_name)


def print_csv_row(list):
    """
    Print list as a row in csv
    """

    print(",".join(map(str, list)))

def print_pairs(pairs, prefix):
    """
    Print each pair of truth and prediction
    """

    truths = [prefix + " Truth"]
    predictions = [prefix + " Prediction"]
    for pair in pairs:
        truths.append(pair[0])
        predictions.append(pair[1])

    print_csv_row(truths)
    print_csv_row(predictions)

def print_results(results):
    """
    Print the results
    """

    # print header
    header = ["Model"]
    i = 0
    while i < 5:
        header.append("Fold" + str(i))
        i += 1
    header.append("Avg")
    print_csv_row(header)

    # print individual results
    for key in results:
        avg = round(stats.mean(results[key]), 3)
        line = [key] + results[key] + [avg]
        line = [str(word) for word in line]
        print_csv_row(line)