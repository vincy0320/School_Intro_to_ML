#!/usr/bin/python3


def train(feature_table, classes):
    """
    Train a given feature table to classify the given classes using 
    naive bayes algorithm. 
    """
    count_class_pos = classes.count(1)
    count_class_neg = classes.count(0)
    prob_class_pos = count_class_pos / len(classes)
    prob_class_neg = count_class_neg / len(classes)

    class_pos_features_prob = []
    class_neg_features_prob = []

    for feature_list in feature_table:
        f_pos_c_pos_count = 0
        f_pos_c_neg_count = 0
        f_neg_c_pos_count = 0
        f_neg_c_neg_count = 0

        for index in range(len(feature_list)):
            if feature_list[index]  == 1 and classes[index]  == 1:
                f_pos_c_pos_count += 1
            elif feature_list[index]  == 1 and classes[index]  == 0:
                f_pos_c_neg_count += 1
            elif feature_list[index]  == 0 and classes[index]  == 1:
                f_neg_c_pos_count += 1
            else:
                f_neg_c_neg_count += 1
        
        class_pos_features_prob.append([
            f_neg_c_pos_count / count_class_pos, 
            f_pos_c_pos_count / count_class_pos
        ])
        class_neg_features_prob.append([
            f_neg_c_neg_count / count_class_neg, 
            f_pos_c_neg_count / count_class_neg
        ])

    return {
        'p_class_positive': prob_class_pos,
        'p_class_pos_features': class_pos_features_prob,
        'p_class_negative': prob_class_neg,
        'p_class_neg_features': class_neg_features_prob
    }


def get_classification(features, model):
    """
    Get the classification for the given feature using the given model. 
    """
    prob_class_pos = model["p_class_positive"]
    prob_class_neg = model["p_class_negative"]

    for index in range(len(features)):
        prob_class_pos *= model["p_class_pos_features"][index][features[index]]
        prob_class_neg *= model["p_class_neg_features"][index][features[index]]
    
    return 1 if prob_class_pos > prob_class_neg else 0
