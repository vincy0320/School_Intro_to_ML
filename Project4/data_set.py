#!/usr/bin/python3
import pandas
import data_set_util as ds_util
import math
import statistics as stats
import util


class DataSet: 

    def __init__(self, args):
        """
        Constructor of data proivder.
        """

        self._args = args
        self.display_name = ds_util.get_dataset_name_for_display(self._args)
        self.is_classification = ds_util.is_classification(self._args)

        self._src_data = self.__read_data()
        
        # Categorical columns doesn't need special hanling in PA#4
        expand_columns = set()
        skip_columns = ds_util.get_columns_to_skip(self._args)
        processed = self.__process_data(
            self._src_data, expand_columns, skip_columns)

        self._data = processed[0]
        self.__post_process_data()

        # Map the source col index to processed col index
        self._col_index_map = processed[1]
        # Map the processed col index to source col index
        self._processed_col_index_map = processed[2]

        self.size = len(self._data)
        self.columns = len(self.row(0))
        self.binary_splits = self.__get_binary_splits()


    def row(self, row_index):
        """
        Get the row in the dataset
        """
        
        return self._data[row_index]

    def cell(self, row_index, col_index, is_source_col_index):
        """
        Get the value in the cell at row index and col index.
        """

        if is_source_col_index:
            col_index = self.__processed_col_index(col_index)
        return self._data[row_index][col_index]


    def is_feature_categorical(self, feature_index, is_source_col_index = False):
        """
        Check whether the feature at the given index is a catgorical feature.
        This method implicitly assume the index is original source index.
        """

        if not is_source_col_index:
            feature_index = self.__src_col_index(feature_index)

        categorical_indices = ds_util.get_categorical_column_indices(self._args)
        return feature_index in categorical_indices

    def get_feature_name(self, feature_index, is_source_col_index = False):
        """
        Get the feature name at the given index
        """

        if not is_source_col_index:
            feature_index = self.__src_col_index(feature_index)
        return ds_util.get_attribute_names(self._args)[feature_index]

    def is_class_index(self, col_index, is_source_col_index = False):
        """
        Check if the given col index is the class index
        """

        if not is_source_col_index:
            col_index = self.__src_col_index(col_index)
        return col_index == ds_util.get_class_index(self._args)

    def get_class_index(self, is_source_col_index = False):
        """
        Get the class index for the dataset
        """

        class_index = ds_util.get_class_index(self._args)
        if is_source_col_index:
            return class_index
        return self.__processed_col_index(class_index)

    def get_class(self, row_index, is_source_col_index = False):
        """
        Get the class at the row
        """

        class_index = self.get_class_index(is_source_col_index)
        return self.cell(row_index, class_index, is_source_col_index)


    def __src_col_index(self, processed_col_index):
        """
        Get the src col index.
        """
        return self._processed_col_index_map[processed_col_index]

    def __processed_col_index(self, col_index): 
        """
        Get the processed col index
        """
        return self._col_index_map[col_index]

    
    def __set_cell(self, row_index, col_index, value, is_source_col_index): #TODO: default to false
        """
        Set the value in the cell
        """

        if is_source_col_index:
            col_index = self.__processed_col_index(col_index)
        self._data[row_index][col_index] = value 


    def __read_data(self):
        """
        Gets the data from the input file given in the args
        """

        input_file_path = ds_util.get_input_file_path(self._args)
        data = None
        try:
            skip_rows = ds_util.get_panda_skip_rows(self._args)
            data = pandas.read_csv(
                input_file_path, skiprows=skip_rows, header=None)
        except: 
            raise Exception(
                "Error: Unable to open the input file, please try again.")

        return data.values.tolist()


    def __process_data(self, raw_data, expand_columns = set(), 
        skip_columns = set()):
        """
        Process the dataset using the given info. Expand or skip column if 
        necessary
        """

        # Get a dictionary for the unique values in the columns that need to be 
        # expanded
        expand_columns_dict = {}
        for index in expand_columns:
            expand_columns_dict[index] = util.get_unique_value_in_column(
                raw_data, index)

        # Find columns with unique values and add it to skip columns
        class_index = ds_util.get_class_index(self._args)
        column_count = len(raw_data[0])
        for index in range(column_count):
            if (index != class_index and index not in expand_columns):
                unique_count = len(util.get_unique_value_in_column(
                    raw_data, index))
                if ((unique_count == len(raw_data) and 
                    index in ds_util.get_categorical_column_indices(self._args))
                    or unique_count == 1): 
                    # every value in this column is unique or every value is the
                    # same, then should skip this column
                    skip_columns.add(index)

        # Create a new data table by expanding or skipping columns
        new_data = []
        for row in raw_data:
            new_row = []
            for index in range(len(row)):
                if index in expand_columns_dict:
                    # Perform one-hot-encoding for categorical columns
                    expanded = util.get_one_hot_encoding(
                        row[index], expand_columns_dict[index])
                    new_row += expanded
                elif index in skip_columns:
                    continue
                else:
                    new_row.append(row[index])
            new_data.append(new_row)

        # Get the index map
        index_map = {}
        processed_index_map = {}
        for col_index in range(column_count):
            new_index = col_index
            if col_index - 1 in index_map:
                # New index should start from the previous index's new index + 1
                new_index = index_map[col_index - 1] + 1

            # Compare col_index with expand_columns
            for exp_index in expand_columns:
                if exp_index < col_index:
                    new_index = new_index - 1 + len(expand_columns_dict[exp_index])
            # Compare col_index with skip columns
            skip_assignment = False
            for skip_index in skip_columns:
                if skip_index == col_index:
                    skip_assignment = True
                    break
                elif col_index - 1 > 0 and col_index - 1 not in index_map:
                    new_index -= 1

            if not skip_assignment:
                index_map[col_index] = new_index
                processed_index_map[new_index] = col_index
        return (new_data, index_map, processed_index_map)


    def __post_process_data(self):
        """
        Post process data if necessary
        """

        if self._args.forestfires:
            # Take logarithm for Forest Fire Data
            class_index = ds_util.get_class_index(self._args)
            for row_index in range(self.size):
                cell_value = self.cell(row_index, class_index, True)
                if cell_value != 0:
                    self.__set_cell(
                        row_index, class_index, math.log(cell_value), True)


    def __get_binary_splits(self):
        """
        Get a dictionary whose key is the feature index (ie. column) and value
        is an array of split points. Only features that are non-categorical
        are applicable for binary splits.
        """

        splits = {}
        
        # A list of sets where each set contains the unique values in the column
        unique_values_dict = {}
        for col_index in range(self.columns):
            unique_values = []
            if not (self.is_class_index(col_index) 
                or self.is_feature_categorical(col_index)):
                unique_values = util.get_unique_value_in_column(
                    self._data, col_index)
            unique_values_dict[col_index] = unique_values
                
        # Loop through each column's unique values
        for col_index in unique_values_dict:
            sorted_values = sorted(unique_values_dict[col_index])
            split_values = set()
            i = 0
            while i < len(sorted_values) - 1:
                mean = round(stats.mean(sorted_values[i:i+2]), 1)
                split_values.add(mean)
                i += 1
            if len(split_values) != 0:
                splits[col_index] = sorted(split_values)
        return splits










    