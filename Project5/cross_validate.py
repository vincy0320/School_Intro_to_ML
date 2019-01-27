#!/usr/bin/python3
import partition


class CrossValidator:

    def __init__(self, dataset, fold_count):
        """
        Constructor for the validator with fold count.
        """

        # Define cross validate set and validation set
        self.size = fold_count
        partition_percentage = 1 / self.size

        proportions = [partition_percentage] * self.size

        # Define partitions used for cross validation and separate validation
        self.partitions = partition.get_stratified_partitions_by_percentage(
            dataset, proportions)

    def get_sets(self, index):
        """
        Get train set and test set for the given fold index
        """

        if index >= self.size:
            raise Exception("Error: Index must be smaller than Valdiator size")

        # Make the given index the test set
        test_set = self.partitions[index]
        # Make the rest the train set
        train_set = []
        for partition in self.partitions[:index] + self.partitions[index + 1:]:
            train_set += partition

        return {
            "train": train_set,
            "test": test_set
        }