#!/usr/bin/python3
import math_util
import util
import output_util

class LogisticRegression:

    def __init__(self, dataset, train_set):
        """
        Constructor for a naive bayes model
        """
        self._dataset = dataset
        self._train_set = train_set
        self._class_names = self._dataset.get_all_classes()
        self._decision_boundary = 0.5

        self._class_weights_dict = {}
        self._is_binary = len(self._class_names) == 2
        if self._is_binary:
            self._class_weights_dict[self._class_names[0]] = [1]
        else:
            for name in self._class_names:
                self._class_weights_dict[name] = [1]

        # Tuned learning rate that varies per dataset
        self._learn_rate = dataset.get_learning_rate()
        self.__train()


    def classify(self, point):
        """
        Get the classification for the given point using the model. 
        """

        predictions = []
        if self._is_binary:
            prob = self.__get_prediction(
                [1] + point, self._class_names[0])["probability"]
            if prob >= self._decision_boundary:
                return self._class_names[0]
            else:
                return self._class_names[1]
        else:
            for class_name in self._class_names:
                prob = self.__get_prediction([1] + point, class_name)["probability"]
                predictions.append(prob)
            softmax = math_util.softmax(predictions)
            max_index = util.get_index_of_max_value(softmax)
            return self._class_names[max_index]



    def __train(self):
        """
        Train a given feature table to classify the given classes using 
        naive bayes algorithm. 
        """

        for index in self._train_set:
            point = [1] + self._dataset.row(index)
            true_class = self._dataset.get_class(index)
            for class_name in self._class_weights_dict:
                if len(self._class_weights_dict[class_name]) == 1:
                    # Intialize weights to a list of 1s that would match the k 
                    # columns in a row, minus 1 class col, plus 1 w_0 term.
                    self._class_weights_dict[class_name] *= len(point)

                prediction = self.__get_prediction(point, class_name)
                truth = 1
                if prediction["class_name"] != true_class:
                    truth = 0
                
                self._class_weights_dict[class_name] = self.__gradient_descent(
                    point, self._class_weights_dict[class_name], 
                    prediction["probability"], truth)

                # print(self._class_weights_dict[class_name])
                

    def __get_prediction(self, point, class_name):
        """
        """
        
        weights = self._class_weights_dict[class_name]
        weighted_sum = math_util.weighted_sum(point, weights)
        
        # Two class case
        pos_probability = math_util.logistic_function(weighted_sum)
        neg_probability = 1 - pos_probability
        # Compare the size
        matched = 1 if pos_probability >= self._decision_boundary else 0

        return {
            "probability" : max(pos_probability, neg_probability),
            "match" : matched,
            "class_name" : class_name if matched else ""
        }


    def __gradient_descent(self, point, weights, prediction, truth):
        """
        Perform gradient descents on the given weight and return the new weights
        """
        coefficient = (truth - prediction) * self._learn_rate
        gradient_vector = math_util.vector_scalar_product(coefficient, point)
        return math_util.vector_sum(weights, gradient_vector) 


    def print_model(self):
        """
        Print the representation of the model
        """

        output_util.print_csv_row(["LR Model"])
        output_util.print_csv_row(["Learning Rate", self._learn_rate])
        output_util.print_csv_row(["Class", "Weights"])
        if self._is_binary:
            first = self._class_names[0]
            output_util.print_csv_row([
                first, 
                self._class_weights_dict[first]
            ])
            output_util.print_csv_row([
                self._class_names[1], 
                "N/A"
            ])
        else: 
            for name in self._class_weights_dict:
                output_util.print_csv_row([
                    name, self._class_weights_dict[name]
                ])