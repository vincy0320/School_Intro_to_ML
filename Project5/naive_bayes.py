#!/usr/bin/python3
import util
import output_util

class NaiveBayes:

    def __init__(self, dataset, train_set):
        """
        Constructor for a naive bayes model
        """
        self._dataset = dataset
        self._train_set = train_set
        self._model = self.__train()

    
    def __train(self):
        """
        Train a given feature table to classify the given classes using 
        naive bayes algorithm. 
        """

        class_indices_dict = util.get_class_indices_dict(
            self._dataset, self._train_set)
        class_count_dict = util.get_class_count_dict(class_indices_dict)
        class_prob_dict = {}
        for class_name in class_count_dict:
            class_prob_dict[class_name] = round(
                class_count_dict[class_name] / len(self._train_set), 3)

        # class_feature_dict is a dictiionary
        # - whose keys are the class names and the value
        # - whose values are 2-element dictionary
        #   - whose keys are 0 or 1, corrsponds to the feature value 
        #   - whose values are dictionary 
        #     - whose keys corresponds to the feature indexes
        #     - shose values are the probability of 0 or 1 occuring for that class
        class_feature_dict = {}
        for index in self._train_set:
            class_name = self._dataset.get_class(index)
            if class_name not in class_feature_dict:
                class_feature_dict[class_name] = {}

            class_occurence = class_count_dict[class_name]
            row = self._dataset.row(index)
            for f_index in range(len(row)):
                feature_val = row[f_index]
                if feature_val not in class_feature_dict[class_name]:
                    class_feature_dict[class_name][feature_val] = {}
                
                prob_increment = 1 / class_occurence
                if f_index not in class_feature_dict[class_name][feature_val]:
                    class_feature_dict[class_name][feature_val][f_index] = 0
                class_feature_dict[class_name][feature_val][f_index] += prob_increment
                class_feature_dict[class_name][feature_val][f_index] = round(
                    class_feature_dict[class_name][feature_val][f_index], 3)

        return {
            'class_probs': class_prob_dict,
            'class_features': class_feature_dict
        }

        
    def classify(self, point):
        """
        Get the classification for the given point using the model. 
        """

        class_names = list(self._model['class_probs'].keys())
        results = []
        for name in class_names:
            results.append(self._model['class_probs'][name])

        for col_index in range(len(point)):
            feature_val = point[col_index]
            for index in range(len(class_names)):
                class_name = class_names[index]
                prob = self._model['class_features'][class_name].get(
                    feature_val, {}).get(col_index, 0)
                results[index] *= prob
                
        max_index = util.get_index_of_max_value(results)
        return class_names[max_index]

    def print_model(self):
        """
        Print the representation of the model
        """

        output_util.print_csv_row(["NB Model"])
        output_util.print_csv_row(["Class","Probability"])
        class_probs = self._model["class_probs"]
        for name in class_probs:
            output_util.print_csv_row([name, class_probs[name]])

        output_util.print_csv_row(["Class","Value","Feature & Probability"])
        for name in self._model["class_features"]:
            class_values = self._model["class_features"][name]
            for class_val in class_values:
                output_util.print_csv_row([
                    name, class_val, class_values[class_val]
                ])
