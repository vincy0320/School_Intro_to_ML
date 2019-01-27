#!/usr/bin/python3
import data_provider as dp
import k_means
import math
import random
import util


class RBFN:

    def __init__(self, data, class_index, dist_matrix):
        """
        Contructor for RBF network class
        """

        self.data = data
        self.class_index = class_index
        self.dist_matrix = dist_matrix
        self.neurons = []
        self.output_nodes = []
        self.output_weights = []

    def __reset(self):
        """
        Reset the RBFN so that we can train it again.
        """
        self.neurons = []
        self.output_nodes = []
        self.output_weights = []

    def __activate_neuron(self, input_vector, neuron_vector):
        """
        Activate function for each neurons. aka. the basis function.
        """
        dist = util.distance(input_vector, neuron_vector)
        return math.exp(-1 * 0.05 * dist**2)

    def __get_neuron_responses(self, input_vector):
        """
        Get the neuron response. ie. the output for the input vector after 
        going through each neuron
        """
        neuron_responses = []
        for neuron in self.neurons:
            neuron_responses.append(
                self.__activate_neuron(input_vector, neuron))
        return neuron_responses


    def __init_neurons(self, train_set, k_means_k = 0):
        """
        Initialize neurons. 
        """

        if k_means_k > 0:
            # When k_means_k is provided, perform k-means clustring and use
            # centroids as neuron
            clusters = k_means.get_clusters(
                self.data, train_set, k_means_k, self.class_index)
            self.neurons = clusters["means"]
        else:
            # Use a random 10% of the training set as neurons
            neuron_count = round(len(train_set) * 0.1)
            neuron_indices = util.get_random_values(train_set, neuron_count)[0]
            self.neurons = []
            for index in neuron_indices:
                self.neurons.append(util.get_point_without_class(
                    self.data, index, self.class_index))

    def __init_output_nodes(self, is_classification, train_set):
        """
        Initilize the output layer nodes. 
        """
        if is_classification:
            # Use number of classes in the train_set as number of output nodes
            classes = set()
            for index in train_set:
                classes.add(util.get_point_class(
                    self.data, index, self.class_index))
            self.output_nodes = list(classes)
        else:
            # For regression, only use one node
            self.output_nodes = [[]] 

    def __init_output_weights(self, category_count, neuron_count):
        """
        Initlizie the weights used for each output node.
        """

        i = 0
        while i < category_count:
            weights = []
            j = 0
            while j < neuron_count:
                # Make the initial weight a random value between 0 and 1
                weights.append(random.uniform(0, 1))
                j += 1
            self.output_weights.append(weights)
            i += 1
    
    def __activate_output(self, sum):
        """
        Logistic activation function
        """
        return 1 / (1 + math.exp(-1 * sum))

    def __get_loss(self, target, sum):
        return target - self.__activate_output(sum)

    def __gradient_descent(self, target, activated_sum, neuron_responses, weights):
        """
        Perform gradient descents on the given weight and return the new weights
        """
        coefficient = (target - activated_sum) * activated_sum * (1 - activated_sum)
        gradient_vector = util.vector_scalar_product(coefficient, neuron_responses)
        return util.vector_subtraction(weights, gradient_vector)

    def train(self, train_set, is_classification, k_means_k = -1):
        """
        Train the RBFN with the given training set.
        """

        # Reset for a clean slate
        self.__reset()

        if len(self.neurons) == 0:
            # when there is no neuron, it means the network is not built
            self.__init_neurons(train_set, k_means_k)
            self.__init_output_nodes(is_classification, train_set)
            self.__init_output_weights(len(self.output_nodes), len(self.neurons))

        for index in train_set:
            # Loop through each point in the training set to train.
            input_vector = util.get_point_without_class(
                self.data, index, self.class_index)
            # Go through the hidden layer neurons
            neuron_responses = self.__get_neuron_responses(input_vector)

            input_class = util.get_point_class(
                self.data, index, self.class_index)
            # Find the output node that's suppose to return 1.
            category_index = -1
            if is_classification:
                category_index = self.output_nodes.index(input_class)

            # Go through each output node to adjust weights.
            for weights_index in range(len(self.output_weights)):
                # set the target value to 1 if it's the same class as the 
                # training example
                target = input_class
                if is_classification:
                    target = 1 if weights_index == category_index else 0
                # Calculate an activated weighted sum
                weights = self.output_weights[weights_index]
                sum = util.get_weighted_sum(weights, neuron_responses)
                activated_sum = self.__activate_output(sum)
                # Perform gradient descnet to adjust weights
                new_weights = self.__gradient_descent(
                    target, activated_sum, neuron_responses, weights)
                self.output_weights[weights_index] = new_weights


    def get_prediction(self, query_index):
        """
        Perform a prediction on the given query index.
        """

        # Get the input vector
        input_vector = util.get_point_without_class(
            self.data, query_index, self.class_index)
        # Get the neuron response from the hidden layer.
        neuron_responses = self.__get_neuron_responses(input_vector)

        # Get the activated weighted sum of all output nodes
        weighted_sums = []
        max_index = -1
        max_value = float("-inf")
        for index in range(len(self.output_weights)):
            weights = self.output_weights[index]
            sum = util.get_weighted_sum(weights, neuron_responses)
            activated_sum = self.__activate_output(sum)
            if activated_sum >= max_value:
                max_index = index
                max_value = activated_sum
            weighted_sums.append(activated_sum)

        # Return the highest performing node
        return self.output_nodes[max_index]
