#!/usr/bin/python3
import math
import math_util
import neuron
import output_util
import partition
import random
import util

class RBFNNeuron:

    def __init__(self, point, selectivity):

        self._point = point
        self._selectivity = selectivity # 1 / spread

    def __activate(self, input_vector):
        """
        Activate function for each neurons. aka. the basis function.
        """
        dist = math_util.distance(input_vector, self._point)
        return math.exp(-1 * dist**2 * self._selectivity)

    def get_response(self, input_vector):
        return self.__activate(input_vector)



class RadialBasisFunctionNetwork:

    def __init__(self, dataset, train_set, selectivity):
        """
        Contructor for RBF network class
        """

        self._dataset = dataset
        self._train_set = train_set
        self._class_names = self._dataset.get_all_classes()
        self._neuron_selectivity = selectivity

        self._hidden_responses = []
        self._hidden_layer = self.__build_hidden_layers()
        self._output_layer = self.__build_output_layer(len(self._hidden_layer))

        self.__train()


    def __build_hidden_layers(self):
        """
        Initialize neurons. 
        """
        # Use a random 10% of the training set as hidden layer neurons
        size = round(len(self._train_set) * 0.1)
        hidden_neuron_indices = partition.get_random_n_and_rest(
            self._train_set, size)

        layer = []
        for index in hidden_neuron_indices:
            neuron_point = self._dataset.row(index)
            layer.append(RBFNNeuron(neuron_point, self._neuron_selectivity))
        return layer


    def __build_output_layer(self, hidden_node_count):
        """
        Initilize the output layer nodes. 
        """
        # Use one node for each class
        output_layer = []
        index = 0
        while index < len(self._class_names):
            # initialize weight for each node in the previous layer
            weights = self.__get_init_weights(hidden_node_count)
            output_layer.append(neuron.Neuron(weights))
            index += 1
        return output_layer


    def __get_init_weights(self, length):
        """
        Get the initial weights used by the neurons
        """
        index = 0
        weights = []
        while index < length:
            weights.append(random.uniform(0, 1))
            index += 1
        return weights


    def __train(self):
        """
        Train the RBFN with the given training set.
        """

        for index in self._train_set:
            # Loop through each point in the training set to train.
            input_vector = self._dataset.row(index)
            # Get the responses from the model
            output_respones = self.__feedforward(input_vector)

            true_class = self._dataset.get_class(index)
            truth_index = self._class_names.index(true_class)

            for node_index in range(len(output_respones)):
                neuron = self._output_layer[node_index]
                target = 1 if node_index == truth_index else 0
                response = output_respones[node_index]
                weight_delta = self.__get_weight_delta(
                    target, response, self._hidden_responses)
                neuron.update_weights(weight_delta)


    def __feedforward(self, point):
        """
        Feed an input point through the RBF network
        """
        hidden_responses = []
        for neuron in self._hidden_layer:
            hidden_responses.append(neuron.get_response(point))
        self._hidden_responses = hidden_responses
        output_responses = []
        for neuron in self._output_layer:
            output_responses.append(neuron.get_response(hidden_responses))
        return output_responses


    def __get_weight_delta(self, target, response, hidden_responses):
        coefficient = (target - response) * response * (1 - response)
        return math_util.vector_scalar_product(coefficient, hidden_responses)


    def classify(self, input_vector):
        """
        Get the classification for the given point using the model. 
        """
        output_responses = self.__feedforward(input_vector)
        max_index = util.get_index_of_max_value(output_responses)
        return self._class_names[max_index]

    def print_model(self, printer):
        """
        Print the model
        """

        printer.print_csv_row(["Feedforward Neural Network Model Details"])
        printer.print_csv_row(["Selectivity", self._neuron_selectivity])
        printer.print_csv_row(["Class Count", len(self._class_names)])
        for node_index in range(len(self._output_layer)):
            node = self._output_layer[node_index]
            printer.print_weights(
                node.get_weights(), "Output Neuron " + str(node_index))