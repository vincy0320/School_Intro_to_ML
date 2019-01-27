#!/usr/bin/python3


def __run_model(features, weights): 
    """
    Run the model with the given features and weights.
    """
    if len(features) != len(weights):
        raise Exception("Error: Features and weights should have the same length!")
    
    sum = 0
    for index in range(len(features)):
        sum += weights[index] * features[index]
    return sum


def __classify(result, theta):
    """
    Compare the given result to theta to see if it can be considered a match. 
    @return 1 for matched 0 otherwise. 
    """
    if theta <= 0:
        raise Exception("Error: Theta must be positive.")

    if (result > theta):
        return 1
    else:
        return 0


def __demote(features, weights, alpha):
    """
    Demotes the given weight mapped to the features by alpha.
    """
    if len(features) != len(weights):
        raise Exception("Error: Features and weights must have the same length.")
    
    if alpha <= 0:
        raise Exception("Error: Alpha must be positive.")

    for index in range(len(features)):
        if features[index] == 1:
            weights[index] = weights[index] / alpha
    
    return weights
    

def train(feature_table, classes, weights, alpha, theta):
    """
    Train a model using winnow algorith with the given info on the given feature table
    and classes.
    """

    if len(feature_table) != len(weights):
        raise Exception("Error: Feature_table and weights should have the same length")

    for feature_list in feature_table:
        if len(feature_list) != len(classes):
            raise Exception("Error: Each feature list and class list should have "
                            "the same length")

    if alpha <= 0 or theta <= 0:
        raise Exception("Error: Both Alpha and Theta must be positive.")

    for index in range(len(classes)):
        features = []
        for feature_list in feature_table:
            features.append(feature_list[index])
        
        result = __run_model(features, weights)
        result = __classify(result, theta)

        if result != classes[index]:
            weights = __demote(features, weights, alpha)
    return weights


def test(feature_table, classes, trained_weights, theta, original_index):
    """
    Test the given feature table to see if the trained weights work well. 
    """
    if len(feature_table) != len(trained_weights):
        raise Exception("Error: Feature_table and weights should have the same length")

    for feature_list in feature_table:
        if len(feature_list) != len(classes):
            raise Exception("Error: Each feature list and class list should have "
                            "the same length")

    if theta <= 0:
        raise Exception("Error: Theta must be positive.")

    for index in range(len(classes)):
        features = []
        for feature_list in feature_table:
            features.append(feature_list[index])
        
        result = __run_model(features, trained_weights)
        result = __classify(result, theta)

        match = "Match" if result == classes[index] else "No Match"
        print(original_index[index], result, classes[index], match)


def get_classification(features, trained_weights, theta):
    """
    Get the classification for the given feature using the given weights and theta. 
    """
    result = __run_model(features, trained_weights)
    return __classify(result, theta)