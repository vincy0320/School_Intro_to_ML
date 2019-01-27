#!/usr/bin/python3


def get_input_file_path(args):
    """
    Get the input file path from the given arguments.
    """

    input_file_path = ""
    if args.abalone:
        input_file_path = args.abalone
    elif args.car:
        input_file_path = args.car
    elif args.segmentation:
        input_file_path = args.segmentation
    elif args.machine:
        input_file_path = args.machine
    elif args.forestfires:
        input_file_path = args.forestfires
    elif args.wine:
        input_file_path = args.wine
    elif args.test:
        input_file_path = args.test
    elif args.test2:
        input_file_path = args.test2

    return input_file_path


def get_categorical_column_indices(args):
    """
    Get a list of indices that are categorical for a given dataset. 
    """

    if args.abalone:
        return [0]
    elif args.car:
        return [0, 1, 2, 3, 4, 5]
    elif args.segmentation:
        return []
    elif args.machine:
        return [0, 1]
    elif args.forestfires:
        return [2, 3]
    elif args.wine:
        return []
    elif args.test:
        return [0, 1, 2, 3]
    elif args.test2:
        return []
    return []


def is_classification(args):
    """
    Check if the datset is for classification, instead of regression.
    """

    return (args.abalone or args.segmentation or args.car or args.test or args.test2)


def get_dataset_name_for_display(args):
    """
    Get the display name for dataset
    """

    if args.abalone:
        return "Abalone"
    elif args.car:
        return "Car Evaluation"
    elif args.segmentation:
        return "Image Segmentation"
    elif args.machine:
        return "Computer Hardware Machine"
    elif args.forestfires:
        return "Forest Fires"
    elif args.wine:
        return "Wine Quality"

    return ""


def get_class_index(args):
    """
    Get the index for the class attribute column for the given dataset.
    """

    if args.abalone:
        return 8
    elif args.car:
        return 6
    elif args.segmentation:
        return 0
    elif args.machine:
        return 8
    elif args.forestfires:
        return 12
    elif args.wine:
        return 0
    elif args.test:
        return 4
    elif args.test2:
        return 2
    return None  # Indicate there is no class column


def get_panda_skip_rows(args):
    """
    Get the number of rows to skip when reading using Panda APIs
    """

    if args.segmentation:
        return 5
    elif args.forestfires:
        return 1
    else:
        return 0

def get_columns_to_skip(args):
    """
    Get an array of column indices to skip
    """

    skip_columns = set()
    if args.machine:
        # Index 1 is Model names, which are too unique, so assume meaningless
        # Index 9 is ERP, which is only for reference purpose
        skip_columns = set([1, 9])

    return skip_columns


def get_attribute_names(args):
    """
    """

    if args.abalone:
        return [
            "Sex",
            "Length",
            "Diameter",
            "Height",
            "Whole weigh",
            "Shucked weight",
            "Viscera weight",
            "Shell weight",
            "Rings",
        ]

    if args.car:
        return [
            "buying",
            "maint",
            "doors",
            "persons",
            "lug_boot",
            "safety",
            "class"
        ]

    if args.segmentation:
        return [
            "Class",
            "region-centroid-col",
            "region-centroid-row",
            "region-pixel-count",
            "short-line-density-5",
            "short-line-density-2",
            "vedge-mean",
            "vegde-sd",
            "hedge-mean",
            "hedge-sd",
            "intensity-mean",
            "rawred-mean",
            "rawblue-mean",
            "rawgreen-mean",
            "exred-mean",
            "exblue-mean",
            "exgreen-mean",
            "value-mean",
            "saturatoin-mean",
            "hue-mean",
        ]

    if args.test:
        return ["Outlook", "Temp", "Humidity", "Wind", "Numeric", "Class"]
    if args.test2:
        return ["N0", "N1", "Class"]