#!/usr/bin/python3
import pandas
import math

def __get_input_file_path(args):
    """
    Get the input file path from the given arguments.
    """

    input_file_path = ""
    if args.ecoli:
        input_file_path = args.ecoli
    elif args.segmentation:
        input_file_path = args.segmentation
    elif args.machine:
        input_file_path = args.machine
    elif args.forestfires:
        input_file_path = args.forestfires
    
    return input_file_path

def get_knn_ks(args):
    """
    Get tunable k value used for k-nearest neighbor
    """
    if args.ecoli:
        return [13] # Tuned: [1, 5, 10, 12, 13, 15]
    elif args.segmentation:
        return [1] # Tuned: [1, 2, 3, 5]
    elif args.machine:
        return [2] # Tuned [1, 2, 3, 5]
    elif args.forestfires:
        return [240] #Tuned [1, 30, 120, 240, 480]
    
    return 0

# [Extra credits: K-Means]
def get_k_means_k(args):
    """
    Get the k value used for k-means + knn based on dataset.
    """

    if args.ecoli:
        return 5 # Five classes in Ecoli
    elif args.segmentation:
        return 7 # Seven classes in Segmentation
    elif args.machine:
        return 2
    elif args.forestfires:
        return 3
    
    return 0

# [Extra credits: RBF]
def get_rbf_k_means_k(args):
    """
    Get the k value used for k-means + RBF based on dataset.
    """

    if args.ecoli:
        return 5 # Five classes in Ecoli
    elif args.segmentation:
        return 7 # Seven classes in Segmentation
    elif args.machine:
        return 4
    elif args.forestfires:
        return 5
    
    return 0

def __get_unique_value_in_column(data, column_index):
    """
    Get the unique values in a column.
    """

    values = set()
    for row in data:
        if row[column_index] not in values:
            values.add(row[column_index])
    return list(values) 


def __get_one_hot_encoding(value, unique_values):
    """
    Get one-hot encoding for a value, given the list of unique values. 
    """

    return list(map(lambda x: 1 if value == x else 0, unique_values))


def __process_data(data, class_index, category_columns, skip_columns):
    """
    Process the dataset using the given info. Expand or skip column if necessary
    """

    # Expand the columns that has category values
    category_columns_dict = {}
    for index in category_columns:
        category_columns_dict[index] = __get_unique_value_in_column(data, index)
    
    new_data = []
    for row in data:
        new_row = []
        for index in range(len(row)):
            if index in category_columns_dict:
                expanded = __get_one_hot_encoding(
                    row[index], category_columns_dict[index])
                new_row += expanded
            elif index in skip_columns:
                continue
            else:
                new_row.append(row[index])
        new_data.append(new_row)

    # Update the class index
    new_class_index = class_index
    for index in category_columns:
        if index < class_index:
            new_class_index = new_class_index - 1 + len(
                category_columns_dict[index])

    for index in skip_columns:
        if index < class_index:
            new_class_index -= 1
    return (new_data, new_class_index)


def get_dataset_name_for_display(args):
    """
    Get the display name for dataset
    """
    
    if args.ecoli:
        return "Ecoli"
    elif args.segmentation:
        return "Image Segmentation"
    elif args.machine:
        return "Computer Hardware Machine"
    elif args.forestfires:
        return "Forest Fires"
    
    return ""


def get_class_index(args):
    """
    Get the index for the class attribute column for the given dataset.
    """

    if args.ecoli:
        return 8
    elif args.segmentation:
        return 0
    elif args.machine:
        return 8
    elif args.forestfires:
        return 12
    return None # Indicate there is no class column


def is_classification(args):
    """
    Check if the datset is for classification, instead of regression.
    """

    return args.ecoli or args.segmentation


def get_data(args):
    """
    Gets the data from the input file given in the args
    """

    input_file_path = __get_input_file_path(args)
    data = None
    try:
        if args.ecoli:
            # ecoli.data is space separated
            data = pandas.read_csv(input_file_path, sep="\s+", header = None)
        elif args.segmentation:
            # segmentation.data has 5 header or empty rows at the top
            data = pandas.read_csv(input_file_path, skiprows=5, header = None)
        elif args.forestfires:
            data = pandas.read_csv(input_file_path, skiprows=1, header = None)
        else:
            data = pandas.read_csv(input_file_path, header = None)
    except: 
        raise Exception(
            "Error: Unable to open the input file, please try again.")

    data = data.values.tolist()
    class_index = get_class_index(args)

    if args.ecoli:
        temp = []
        for row in data:
            if row[class_index] in ["omL", "imL", "imS"]:
                # Skip rows whose classes that has lower count for ecoli
                continue
            # Get rid of the first column because sequence names are unique
            # for each row, so it doesn't contribute to distance. 
            temp.append(row[1:])
        data = temp
        class_index -= 1

    if args.segmentation:
        # No special handling required for Segmentation dataset
        return (data, class_index)

    if args.machine:
        category_columns = [0]
        # Index 1 is Model names, which are too unique, so assume meaningless
        # Index 9 is ERP, which is only for reference purpose
        skip_columns = [1, 9] 
        processed = __process_data(
            data, class_index, category_columns, skip_columns)
        data = processed[0]
        class_index = processed[1]

    if args.forestfires:
        category_columns = [2, 3]
        skip_columns = []
        processed = __process_data(
            data, class_index, category_columns, skip_columns)
        data = processed[0]
        class_index = processed[1]

        for row in data:
            if row[class_index] != 0:
                row[class_index] = math.log(row[class_index])
    
    return (data, class_index)

