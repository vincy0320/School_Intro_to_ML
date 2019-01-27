#!/usr/bin/python3
import math
import statistics as stats


class Printer: 

    def __init__(self, file_name):
        """
        Constructor
        """

        self.file_name = file_name
        self.file = open(self.file_name, "w+")

    def __print(self, some_string):
        """
        Print to file if exists. Always print to output console.
        """
        print(some_string)
        if self.file:
            self.file.write(some_string + "\n")

    def print_separator(self):
        """
        Print a separator
        """
        print("---------------------------")

    def print_empty_line(self):
        """
        Print an empty line
        """
        print("")

    def print_dataset_name(self, dataset):
        """
        Print the name of a dataset.
        """
        line = "Dataset:"
        prefix = "Classification"
        if not dataset.is_classification:
            prefix = "Regression"

        print(prefix, line, dataset.display_name)


    def print_csv_row(self, list):
        """
        Print list as a row in csv
        """

        self.__print(",".join(map(str, list)))

    def print_pairs(self, pairs, prefix):
        """
        Print each pair of truth and prediction
        """

        truths = [prefix, "Truth"]
        predictions = [prefix, "Prediction"]
        for pair in pairs:
            truths.append(pair[0])
            predictions.append(pair[1])

        self.print_csv_row(truths)
        self.print_csv_row(predictions)

    def print_results(self, results):
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
        self.print_csv_row(header)

        # print individual results
        for key in results:
            avg = round(stats.mean(results[key]), 3)
            line = [key] + results[key] + [avg]
            line = [str(word) for word in line]
            self.print_csv_row(line)


    def print_weights(self, weights, name):
        to_print = []
        if name:
            to_print.append(name)
        to_print += [round(w, 5) for w in weights]
        self.print_csv_row(to_print)