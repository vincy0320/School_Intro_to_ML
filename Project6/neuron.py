#!/usr/bin/python3
import math_util

class Neuron:

    def __init__(self, weights):
        """
        Constructor for a Neuron object
        """

        if len(weights) == 0:
            raise Exception("Error: Invalid input weights")
        self._weights = weights

    def get_weights(self):
        return self._weights

    def update_weights(self, weight_delta):
        """
        Update the current weight with the given weight delta
        """

        self._weights = math_util.vector_sum(self._weights, weight_delta) 


    def get_response(self, input_vector):
        """
        Calculate the neuron response from the given input vector
        """

        if len(input_vector) == 0:
            raise Exception("Error: Empty input")

        sum_value = math_util.weighted_sum(input_vector, self._weights)
        # print(sum_value)
        return math_util.logistic_function(sum_value)
