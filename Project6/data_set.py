#!/usr/bin/python3
import pandas
import data_set_info as dsi
import encoding 
import math
import random
import statistics as stats
import util


class DataSet: 

    def __init__(self, args):
        """
        Constructor of data proivder.
        """

        self._args = args
        # Read data from file
        self._src_data = self.__read_data()
        self.size = len(self._src_data)
        # Fill the missing data in the source data
        self.__fill_missing_src_data()
        
        self.display_name = dsi.get_dataset_name_for_display(self._args)
        self.is_classification = dsi.is_classification(self._args)

        self._col_unique_values = {}
        self._encoders = {}
        self._data = self.__process_data(dsi.get_columns_to_skip(self._args))
        self.columns = len(self.row(0))


    def row(self, row_index):
        """
        Get the row in the dataset
        """
        return self._data[row_index]


    def cell(self, row_index, col_index):
        """
        Get the value in the cell at row index and col index.
        """
        return self._data[row_index][col_index]


    def is_feature_categorical(self, feature_index):
        """
        Check whether the feature at the given index is a catgorical feature.
        This method implicitly assume the index is original source index.
        """

        categorical_indices = dsi.get_categorical_column_indices(self._args)
        return feature_index in categorical_indices

    def get_all_classes(self):
        """
        Get all the classes in the dataset
        """

        class_index = dsi.get_class_index(self._args)
        return self._col_unique_values[class_index]


    def get_class(self, row_index):
        """
        Get the class at the row
        """

        class_index = dsi.get_class_index(self._args)
        return self._src_data[row_index][class_index]
        

    def __read_data(self):
        """
        Gets the data from the input file given in the args
        """

        input_file_path = dsi.get_input_file_path(self._args)
        data = None
        try:
            skip_rows = dsi.get_panda_skip_rows(self._args)
            data = pandas.read_csv(
                input_file_path, skiprows=skip_rows, header=None)
        except: 
            raise Exception(
                "Error: Unable to open the input file, please try again.")

        return data.values.tolist()


    def __fill_missing_src_data(self):
        """
        Fill missing data by randomly assign a value to the missing data
        """

        if not dsi.has_missing_attributes(self._args):
            return

        for row_index in range(self.size):
            row = self._src_data[row_index]
            while "?" in row:
                missing_index = row.index("?")
                random_row_index = random.randint(0, self.size - 1)
                random_row = self._src_data[random_row_index]
                row[missing_index] = random_row[missing_index]


    def __process_data(self, skip_columns = set()):
        """
        Process the dataset using the given info. Expand or skip column if 
        necessary
        """

        src_data_col_count = len(self._src_data[0])
        categorical_columns = dsi.get_categorical_column_indices(self._args)

        self._col_unique_values = {}
        for index in range(src_data_col_count):
            self._col_unique_values[index] = util.get_unique_value_in_column(
                self._src_data, index)

            if index == dsi.get_class_index(self._args):
                # If the index is class index, then we should skip this column
                skip_columns.add(index)
                continue
            unique_count = len(self._col_unique_values[index])
            if unique_count == 1:
                # If all values are the same, then we should skip this column
                skip_columns.add(index)
                continue
            is_categorical_index = index in categorical_columns
            if is_categorical_index and unique_count == self.size:
                # if all values are different for a categorical column, then skip
                skip_columns.add(index)
                continue
            # Create an encoder for each index
            self._encoders[index] = encoding.Encoder(
                self._col_unique_values[index], is_categorical_index)

        # Create a new data table by expanding or skipping columns
        new_data = []
        for row in self._src_data:
            # Loop through each row of the src data
            new_row = []
            for col_index in range(len(row)):
                # if col_index == dsi.get_class_index(self._args):
                #     new_row += [row[col_index]]
                #     continue
                if col_index in skip_columns:
                    continue
                # In PA#5, every column needs to be encoded
                encoded = self._encoders[col_index].encode(row[col_index])
                new_row += encoded
            new_data.append(new_row)
        return new_data
