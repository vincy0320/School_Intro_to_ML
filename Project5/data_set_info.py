#!/usr/bin/python3


def get_input_file_path(args):
    """
    Get the input file path from the given arguments.
    """

    input_file_path = ""
    if args.breast_cancer:
        input_file_path = args.breast_cancer
    elif args.iris:
        input_file_path = args.iris
    elif args.glass:
        input_file_path = args.glass
    elif args.soybean:
        input_file_path = args.soybean
    elif args.vote:
        input_file_path = args.vote

    return input_file_path


def get_categorical_column_indices(args):
    """
    Get a list of indices that are categorical for a given dataset. 
    The returned value doesn't include the class column
    """

    if args.breast_cancer:
        return list(range(10))
    elif args.iris:
        return []
    elif args.glass:
        return []
    elif args.soybean:
        return list(range(35))
    elif args.vote:
        return list(range(1, 17))
    return []


def is_classification(args):
    """
    Check if the datset is for classification, instead of regression.
    """

    if (args.breast_cancer or args.glass or args.iris or args.soybean 
        or args.vote):
        return True
    return False


def get_dataset_name_for_display(args):
    """
    Get the display name for dataset
    """

    if args.breast_cancer:
        return "Breast Cancer"
    elif args.iris:
        return "Iris"
    elif args.glass:
        return "Glass"
    elif args.soybean:
        return "Soybean"
    elif args.vote:
        return "Vote"

    return "Unrecognized"


def get_class_index(args):
    """
    Get the index for the class attribute column for the given dataset.
    """

    if args.breast_cancer:
        return 10
    elif args.iris:
        return 4
    elif args.glass:
        return 10
    elif args.soybean:
        return 35
    elif args.vote:
        return 0
    return None  # Indicate there is no class column


def has_missing_attributes(args):
    """
    Return true if there is missing attribute in the datset
    """

    if args.breast_cancer or args.vote:
        return True
    else:
        return False


def get_panda_skip_rows(args):
    """
    Get the number of rows to skip when reading using Panda APIs
    """
    return 0


def get_columns_to_skip(args):
    """
    Get an array of column indices to skip
    """

    skip_columns = set()
    if args.breast_cancer:
        skip_columns = set([0]) # The sample code number column, all unique
    if args.glass:
        skip_columns = set([0]) # THe id number column.
    return skip_columns


def get_learning_rate(args):
    """
    Get the tuned learning rate for the dataset
    """

    if args.breast_cancer:
        return 0.002
    if args.glass:
        return 0.001
    if args.iris:
        return 0.5
    if args.soybean:
        return 0.045
    if args.vote:
        return 0.005
    return 0