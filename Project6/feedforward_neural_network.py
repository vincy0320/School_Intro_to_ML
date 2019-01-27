#!/usr/bin/python3
import math_util
import neuron
import output_util
import random
import util


class FeedforwardNeuralNetwork:

    def __init__(self, dataset, train_set, learn_rate, hidden_layers_node_counts = []):
        """
        Construct an Feedforward Neural Network with backprop.
        """

        self._dataset = dataset
        self._train_set = train_set
        self._learn_rate = learn_rate
        self._class_names = self._dataset.get_all_classes()
        self._is_binary = len(self._class_names) == 2

        # Construct the hidden layer
        self._hidden_layers = self.__build_hidden_layers(
            hidden_layers_node_counts)
        # Prepare a cache to store hidden layer responses, which is useful for
        # backpropagation
        self._hidden_layer_responses = []
        # Construct the output layer
        self._output_layer = self.__build_output_layer()
        self._output_layer_responses = []

        # Train the dataset
        self.__train()


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


    def __build_hidden_layers(self, hidden_layers_node_counts):
        """
        Build Hidden Layers by having the number of nodes specified in 
        hidden_layers_node_counts
        """

        layers = []
        for layer_index in range(len(hidden_layers_node_counts)):
            index = 0
            layer = []
            while index < hidden_layers_node_counts[layer_index]:
                weights = []
                if layer_index == 0:
                    # The length of weights for the first hidden layer depends
                    # on the length of the input
                    weights = self.__get_init_weights(self._dataset.columns)
                else:
                    # The length of weights for the subsequent hidden layer 
                    # depends on the number of neurons in the previous layer
                    weights = self.__get_init_weights(
                        hidden_layers_node_counts[layer_index - 1])
                layer.append(neuron.Neuron(weights))
                index += 1
            if len(layer) > 0:
                layers.append(layer)
        return layers


    def __build_output_layer(self):
        """
        Build the output layer by having one output node
        """

        weight_count = 0
        if len(self._hidden_layers) > 0:
            # if there are hidden layers, then output node's weight count is the
            # number of neurons in the last hidden layer.
            weight_count = len(self._hidden_layers[-1])
        else:
            # otherwise the output node's weight count is the length of the 
            # input vector, ie. the number of columns in the dataset
            weight_count = self._dataset.columns
        
        output_layer = []
        if self._is_binary:
            # If it's binary classification, then use one node.
            weights = self.__get_init_weights(weight_count)
            output_layer = [neuron.Neuron(weights)]
        else:
            # otherwise, use one node for each class
            index = 0
            while index < len(self._class_names):
                weights = self.__get_init_weights(weight_count)
                output_layer.append(neuron.Neuron(weights))
                index += 1
        return output_layer


    def __train(self):
        """
        Train a given feature table to classify the given classes using 
        naive bayes algorithm. 
        """
        for index in self._train_set:
            input_vector = self._dataset.row(index)
            self.__feedforward(input_vector)

            truth_index = self.__get_truth_index(index)

            # Get weight update for output layer
            point = input_vector
            if len(self._hidden_layer_responses) > 0:
                # Use the last hidden layer responses as input for the output 
                # layer
                point = self._hidden_layer_responses[-1]
            output_weight_deltas = self.__get_output_weight_deltas(
                truth_index, point)

            # Get weight update for hidden layers
            hidden_weight_deltas = self.__get_hidden_layer_weight_deltas(
                input_vector, truth_index)
            
            # Update weights in the output layer
            for node_index in range(len(self._output_layer)):
                self._output_layer[node_index].update_weights(
                    output_weight_deltas[node_index])
                # output_util.print_weights(self._output_layer[node_index].get_weights())

            # Update weights for each of the node in each hidden layer
            for layer_index in range(len(self._hidden_layers)):
                hidden_layer = self._hidden_layers[layer_index]
                for node_index in range(len(hidden_layer)):
                    hidden_layer[node_index].update_weights(
                        hidden_weight_deltas[layer_index][node_index])

                    # output_util.print_weights(
                    #     hidden_weight_deltas[layer_index][node_index])


    def __feedforward(self, point):
        """
        Feed an input point through the neural network and store the reponses
        of each neuron
        """
        layer_input = point
        self._hidden_layer_responses = []
        # Loop through each hidden layer
        for layer in self._hidden_layers:
            # Loop through each node in each layer and store all responses from 
            # a layer into a single vector
            layer_vector = []
            for node in layer:
                layer_vector.append(node.get_response(layer_input))
            # store the vector for backprop reference
            self._hidden_layer_responses.append(layer_vector)
            # use all responses in the current layer as the input of the next 
            # layer
            layer_input = layer_vector
        
        self._output_layer_responses = []
        # Loop through each node 
        for node in self._output_layer:
            # store the response from each node for backprop reference
            self._output_layer_responses.append(node.get_response(layer_input))
        
    
    def __get_truth_index(self, index):
        """
        Get the index of the truth value in self._class_names
        """
        true_class = self._dataset.get_class(index)
        return self._class_names.index(true_class)


    def __get_output_weight_deltas(self, truth_index, point):
        """
        Get the weight update for all the output nodes
        """
        delta_vectors = []
        # Loop through each node in output layer
        for node_index in range(len(self._output_layer)):
            response = self._output_layer_responses[node_index]
            # target value is 1 if the index matches, otherwise it's 0.
            target = 1 if node_index == truth_index else 0
            # Store each weight update in a vector for the node
            deltas = []
            for x in point:
                delta = self.__get_output_weight_delta(target, response, x)
                deltas.append(delta)
            # store the weight update vector for each node
            delta_vectors.append(deltas)
        return delta_vectors


    def __get_output_weight_delta(self, target, response, x):
        """
        Return weight update for a output node.
        """
        # delta is eta * (d_j-o_j) * o_j * (1-o_j) * x_ji
        return (self._learn_rate * (target - response) * response * 
            (1 - response) * x)


    def __get_hidden_layer_weight_deltas(self, point, truth_index):
        """
        Get the weight update for all the hidden nodes in each hidden layer
        """
        all_layer_weight_deltas = []
        # Loop through each hidden layer
        for layer_index in range(len(self._hidden_layers)):
            if layer_index > 0:
                # input for each hidden layer is the output of the previous 
                # hidden layer, except for the first hidden layer.
                point = self._hidden_layer_responses[layer_index-1]

            hidden_layer = self._hidden_layers[layer_index]
            # Store weight update vector for the node
            layer_weight_delta = []
            # Loop through each node in the hidden layer
            for node_index in range(len(hidden_layer)):
                ds_delta = self.__get_downstream_delta_for_hidden_node(
                    layer_index, node_index, truth_index)
                # Store each weight update in a vector for the node
                node_deltas = []
                for x in point:
                    weight_delta = -1 * self.__get_hidden_layer_weight_delta(
                        ds_delta, x)
                    node_deltas.append(weight_delta)
                layer_weight_delta.append(node_deltas)
            # store the weight update vector for each node
            all_layer_weight_deltas.append(layer_weight_delta)
        return all_layer_weight_deltas

    
    def __get_downstream_delta_for_hidden_node(self, layer_index, node_index, truth_index):
        """
        Get the downstream delta for hidden node.
        """

        ds_sum = 0
        if layer_index + 1 < len(self._hidden_layers):
            ds_layer = self._hidden_layers[layer_index + 1]
            # Loop through each node in the down stream layer
            for ds_node_index in range(len(ds_layer)):
                ds_delta = self.__get_downstream_delta_for_hidden_node(
                    layer_index + 1, ds_node_index, truth_index)
                ds_weights = ds_layer[ds_node_index].get_weights()
                ds_sum += ds_delta * ds_weights[node_index]
        else:
            # Downstream layer is the output layer
            for output_index in range(len(self._output_layer)):
                response = self._output_layer_responses[output_index]
                # target value is 1 if the index matches, otherwise it's 0.
                target = 1 if node_index == truth_index else 0
                output_delta = (target - response) * response * (1 - response)
                ds_sum += (output_delta * 
                    self._output_layer[output_index].get_weights()[node_index])

        response = self._hidden_layer_responses[layer_index][node_index]
        delta = response * (1 - response) * ds_sum
        return delta 


    def __get_hidden_layer_weight_delta(self, ds_delta, x):
        """
        Return weight update for a hidden node.
        """
        # delta is eta * delta_j * x_ji
        return self._learn_rate * ds_delta * x


    def __get_prediction_index(self):
        """
        Return the prediction as the index of self._class_names
        """
        if self._is_binary:
            response = self._output_layer_responses[0]
            # if the response is greater than 0.5, then consider the prediction
            # to the first index
            return 0 if response >= 0.5 else 1
        else:
            softmax = math_util.softmax(self._output_layer_responses)
            max_index = util.get_index_of_max_value(softmax)
            # return the index with max probability as the predicted index
            return max_index


    def classify(self, point):
        """
        Get the classification for the given point using the model. 
        """

        # Reset to a clean state
        self._hidden_layer_responses = []
        self._output_layer_responses = []

        self.__feedforward(point)
        return self._class_names[self.__get_prediction_index()]


    def print_model(self, printer):
        """
        Print the representation of the model
        """

        printer.print_csv_row(["Feedforward Neural Network Model Details"])
        printer.print_csv_row(["Learning Rate", self._learn_rate])
        printer.print_csv_row(["Hidden Layers", len(self._hidden_layers)])
        printer.print_csv_row(["Class Count", len(self._class_names)])

        for index in range(len(self._hidden_layers)):
            printer.print_csv_row(["Hidden Layer " + str(index) + " Weights"])
            for node_index in range(len(self._hidden_layers[index])):
                node = self._hidden_layers[index][node_index]
                printer.print_weights(
                    node.get_weights(), "Hidden Neuron " + str(node_index))

        printer.print_csv_row(["Output Layer Weights"])
        for node_index in range(len(self._output_layer)):
            node = self._output_layer[node_index]
            printer.print_weights(
                node.get_weights(), "Output Neuron " + str(node_index))