#!/usr/bin/python3
import random
import math
import util
import time

class DecisionTree:

    def __init__(self, dataset, train_set, skip_feature_indices = {}, auto_build = True):
        """
        Constructor of decision tree.
        """

        self.dataset = dataset
        self.train_set = train_set

        # A list of indices that should be skipped when calculating expected
        # entropies. Normally, this is for the case when a feature is already
        # used as a node in previous levels of the tree.
        self.skip_feature_indices = skip_feature_indices

        self.feature_index = -1
        self.numeric_feature_split_index = -1
        # Children is a dict whose keys are the values of the feature that
        # corresponds to the feature_index and value is the root of sub decision
        # tree. 
        self.children = {}
        
        self.class_distribution = self.__get_class_distribution()
        self.is_pruned = False
        self.can_prune = True

        if auto_build:
            self.__build_tree()

    def __get_class_distribution(self):
        """
        Get the class distribution of the node
        """

        class_indices_dict = util.get_class_indices_dict(
            self.dataset, self.train_set)
        return util.get_class_count_dict(class_indices_dict)

    def get_majority_class(self):
        """
        Get the class of highest occurence in the tree
        """
        majority = max(self.class_distribution, key=self.class_distribution.get)
        return majority


    def __build_tree(self):
        """
        Build the tree
        """
        self.__id3()


    def __entropy(self, class_count_dict):
        """
        Calculate shannon's entropy for the class attribute
        """

        total = sum(class_count_dict.values())
        entropy = 0
        for class_name in class_count_dict:
            class_count = class_count_dict[class_name]
            ratio = class_count / total
            entropy += - ratio * math.log(ratio, 2)
        return entropy

    def __expected_entropy(self, attr_dict):
        """
        Calculate the expected entropy with the given attribute dictionary
        """

        total = len(self.train_set)
        expected_value = 0
        for attr_value in attr_dict:
            attr_class_dict = attr_dict[attr_value]
            occurence = sum(attr_class_dict.values()) 
            entropy = self.__entropy(attr_class_dict)
            expected_value += occurence / total * entropy
        return expected_value

    def __intrinsic_value(self, attr_dict, total_size):
        """
        Calculate the intrinsic value for the given attribute dict and total 
        size. The returned value will be used to calculate gain ratio.
        """

        intrinsic_value = 0
        for attr_value in attr_dict:
            occurence = sum(attr_dict[attr_value].values())
            ratio = occurence / total_size
            if ratio != 0:
                intrinsic_value += - ratio * math.log(ratio, 2)
        return intrinsic_value


    def __get_numerical_feature_gain_ratio(self, attr_dict, 
        skip_indices, entropy_class):
        """
        Get the expected entropies for numerical features.
        """

        last_one = (len(attr_dict) - len(skip_indices) == 1)

        gain_ratios = []
        for split_index in attr_dict:
            gain_ratio = float("-inf")
            if split_index not in skip_indices:
                exp = self.__expected_entropy(attr_dict[split_index])
                iv = self.__intrinsic_value(
                    attr_dict[split_index], len(self.train_set))
                if iv != 0:
                    gain_ratio = (entropy_class - exp) / iv
                elif last_one:
                    return (gain_ratio, split_index)
            gain_ratios.append(gain_ratio)

        max_gain_ratio_index = util.get_index_of_max_value(gain_ratios)


        max_gain_ratio = float("-inf")
        if len(gain_ratios) != 0:
            max_gain_ratio = gain_ratios[max_gain_ratio_index]
        return (max_gain_ratio, max_gain_ratio_index)


    def __get_gain_ratio_for_feature_index(self, col_index, 
        attr_dict_list, selected_splits, entropy_class):
        """
        Get the gain ratio for the feature at the given feature index
        """

        exp = float("inf")
        gain_ratio = float("-inf")
        if self.dataset.is_class_index(col_index):
            # If the col_index is the class attribute index or if the 
            # col_index should be skiped, add the max entropy possible.
            return gain_ratio
            
        attr_dict = attr_dict_list[col_index]
        if self.dataset.is_feature_categorical(col_index, False):
            iv = self.__intrinsic_value(
                attr_dict, len(self.train_set))
            if iv == 0:
                return gain_ratio
            if col_index not in self.skip_feature_indices:
                # Calcluate the expected entropy for the attribute and column
                exp = self.__expected_entropy(attr_dict)
                gain_ratio = (entropy_class - exp) / iv
        else:
            # Get the split indices that are already used
            skip_split_indices = set()
            if col_index in self.skip_feature_indices:
                skip_split_indices = self.skip_feature_indices[col_index]

            results = self.__get_numerical_feature_gain_ratio(
                attr_dict, skip_split_indices, entropy_class)
            if results[1] != -1:
                gain_ratio = results[0]
                selected_splits[col_index] = results[1]

        return gain_ratio


    def __get_sub_tree_train_set(self, attr_value):
        """
        Get the train set for the next iteration of constructing sub tress. 
        ie. selecting the row that matches the criteria for the given attribute
        value at the selected feature index.
        """
        sub_tree_train_set = []
        for index in self.train_set:
            row = self.dataset.row(index)
            col_value = row[self.feature_index]
            if self.dataset.is_feature_categorical(self.feature_index):
                if col_value == attr_value:
                    sub_tree_train_set.append(index)
            else:
                splits = self.dataset.binary_splits[self.feature_index]
                split_point = splits[self.numeric_feature_split_index]

                if ((attr_value == "<=" and col_value <= split_point) or
                    (attr_value == ">" and col_value > split_point)):
                    sub_tree_train_set.append(index)
        return sub_tree_train_set

    
    def __update_skip_feature_indices(self, selected_splits):
        """
        Update the skip feature indices for the selected feature index
        """
        if self.dataset.is_feature_categorical(self.feature_index):
            # Categorical features are removed once being used.
            self.skip_feature_indices[self.feature_index] = []
        else:
            # Non-categorical features can occur multiple times in a tree
            # because we look for the binary splits that best separate the
            # samples.
            if self.feature_index not in self.skip_feature_indices:
                self.skip_feature_indices[self.feature_index] = set()
            self.skip_feature_indices[self.feature_index].add(
                selected_splits[self.feature_index])


    def __id3(self):
        """
        Construct a tree using id3
        """


        # Get entropy of all classes. ie. I(c1, ..., ck)
        class_indices_dict = util.get_class_indices_dict(
            self.dataset, self.train_set)
        count_dict = util.get_class_count_dict(class_indices_dict)
        entropy_class = self.__entropy(count_dict)

        # start_time = time.time()

        # Construct attribute dictionary
        attr_dict_list = util.get_attribute_dict_list(
            self.dataset, self.train_set, self.skip_feature_indices) 
        # time2 = time.time()
        # print("get_attribute_dict_list:", time2 - start_time)

        # Get a list of grains. ie. I(c1, ..., ck) - E(fk) / iv(fk)
        gain_ratios = []
        # Use a dictionary to keep track of the indices of the selected split 
        # for each feature/column index
        selected_splits = {}
        for col_index in range(len(attr_dict_list)):
            gain_ratio = self.__get_gain_ratio_for_feature_index(
                col_index, attr_dict_list, selected_splits, entropy_class)
            gain_ratios.append(gain_ratio)

        # Get the max gain ratio
        max_gain_ratio_index = util.get_index_of_max_value(gain_ratios)
        # Set the feature_index as the index with max again ratio
        self.feature_index = max_gain_ratio_index
        if self.feature_index == -1:
            # No max gain ratio applicable, thus return
            return

        # Get all possible values for the seelcted feature index and split index
        # if applicable.
        attr_dict = {}
        if self.dataset.is_feature_categorical(self.feature_index):
            attr_dict = attr_dict_list[self.feature_index]
        else:
            self.numeric_feature_split_index = selected_splits[self.feature_index]
            attr_dict = attr_dict_list[self.feature_index][
                self.numeric_feature_split_index]

        # Populate the children for this tree by examine the possible values
        for attr_value in attr_dict:
            attr_class_dict = attr_dict[attr_value]
            if len(attr_class_dict) == 1:
                # If there is only one class in for the attribute, then there is
                # no more decisions to make. Thus it's a leaf node.
                self.children[attr_value] = DecisionTreeLeaf(
                    self.dataset, [], attr_class_dict)
                continue

            # Get a new sub-tree train-set by including only the row indices 
            # whose feature_index column matches the attr_value
            sub_train_set = self.__get_sub_tree_train_set(attr_value)

            # Construct a subtree for the attr_value with the sub-train-set 
            # recursively
            self.__update_skip_feature_indices(selected_splits)
            if len(self.skip_feature_indices) == self.dataset.columns - 1:
                # if all features has been used for creating previous trees
                # then create a leaf because no more feature can be used to make
                # more decisions
                self.children[attr_value] = DecisionTreeLeaf(
                    self.dataset, sub_train_set)
            else:
                self.children[attr_value] = DecisionTree(self.dataset, 
                    sub_train_set, self.skip_feature_indices.copy())


    def classify(self, instance):
        """
        Classifies the given instance. Return the decison path as a list
        """

        path = []
        if self.feature_index == -1:
            return path
            
        feature_value = instance[self.feature_index]
        # Recursively classify using the matching branch's children
        if self.dataset.is_feature_categorical(self.feature_index):
            if feature_value not in self.children:
                # This classification fails from here
                return [self.get_majority_class()]

            child = self.children[feature_value]
            if child.is_pruned:
                return [child.get_majority_class()]

            path.append(
                self.dataset.get_feature_name(self.feature_index) + ":" + 
                feature_value)
        else:
            splits = self.dataset.binary_splits[self.feature_index]
            split_point = splits[self.numeric_feature_split_index]

            branch_name = "<="
            if feature_value > split_point:
                branch_name = ">"

            child = self.children[branch_name]
            if child.is_pruned:
                return [child.get_majority_class()]

            path.append(
                self.dataset.get_feature_name(self.feature_index, False) 
                + ":" + branch_name + str(split_point))

        path += child.classify(instance)
        # print(path)
        return path

    def prune_next(self):
        """
        Prune the next prunable child
        """

        sorted_children_keys = sorted(self.children)
        for key in sorted_children_keys:
            child = self.children[key]
            if not child.is_pruned and child.can_prune:
                pruned = child.prune_next()
                if pruned is not None:
                    return pruned
        
        can_prune = True
        all_children_pruned = True
        for child in self.children.values():
            can_prune = can_prune and child.can_prune
            all_children_pruned = all_children_pruned and child.is_pruned

        self.can_prune = can_prune
        if not self.can_prune:
            return None

        if all_children_pruned:
            self.is_pruned = True
            return self

        # This should never happen, but just for safety
        self.is_pruned = True
        return self

    def get_depth(self):
        """
        Get the depth of the tree.
        """

        if len(self.children) == 0:
            return 1
        
        depths = []
        for key in self.children:
            child = self.children[key]
            if child.is_pruned:
                depths.append(0)
            else:                
                depths.append(child.get_depth())
        
        return 1 + max(depths)

    def get_size(self):
        """
        Get the number of nodes in the tree
        """

        if self.is_pruned:
            return 0

        total = 0
        for key in self.children:
            child = self.children[key]
            total += child.get_size()
        
        return 1 + total


class DecisionTreeLeaf(DecisionTree):
    def __init__(self, dataset, train_set, class_distribution = None):
        DecisionTree.__init__(self, dataset, train_set, {}, False)
        if class_distribution is not None:
            self.class_distribution = class_distribution

    def classify(self, instance):
        return [self.get_majority_class()]



