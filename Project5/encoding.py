#!/usr/bin/python3
import statistics as stats


class Encoder:

    def __init__(self, unique_values, is_categorical):
        """
        Constructor of an Encoder using one-hot-encoding
        """

        self.is_categorical = is_categorical
        self.is_binary = len(unique_values) == 2
        self.unique_values = unique_values
        if not is_categorical and not self.is_binary: 
            # Use the lower bound and the upper bound of standard deviation band
            # to one-hot-encode continuous value
            self.unique_values = self.__get_stdev_band(unique_values)


    def __get_stdev_band(self, unique_values):
        """
        Get the lower bound and upper bound for the standard devaitation band
        for continuous value. 
        """

        mean = stats.mean(unique_values)
        stdev = stats.stdev(unique_values)
        return [mean - stdev, mean + stdev]


    def encode(self, value):
        """
        Get one-hot encoding for a value based on the 
        unique values in this encoder.

        Return a list of 0s except 1 at the index that matches the unique value
        index.
        """

        encoded = []
        if self.is_binary:
            encoded.append(0 if value == self.unique_values[0] else 1)
        elif not self.is_categorical:
            lower_bound = self.unique_values[0]
            upper_bound = self.unique_values[1]
            encoded.append(1 if lower_bound < value < upper_bound else 0)
        else: 
            for index in range(len(self.unique_values)):
                unique = self.unique_values[index]
                encoded.append(1 if value == unique else 0)

        return encoded