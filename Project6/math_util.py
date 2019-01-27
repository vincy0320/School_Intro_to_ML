#!/usr/bin/python3
import math


def logistic_function(value):
    """
    Logistic function
    """
    return 1 / (1 + math.exp(-1 * value))


def weighted_sum(vector, weights):
    """
    Get the weighted sum.
    """
    return vector_product(vector, weights)


def vector_scalar_product(scalar, vector):
    """
    Perform a scalar multiplication on a vector
    """

    output = []
    for index in range(len(vector)):
        output.append(scalar * vector[index])
    return output


def vector_sum(a, b):
    """
    Sums given vector a by given vector b.
    """

    if len(a) != len(b):
        raise Exception("Error: Vector addition require same length")

    output = []
    for index in range(len(a)):
        output.append(a[index] + b[index])
    return output

def vector_subtract(a, b):
    """
    Subtract given vector a by given vector b.
    """

    neg_b = vector_scalar_product(-1, b)
    return vector_sum(a, neg_b)


def vector_product(a, b):
    """
    Multiplies the given vector a by given vector b.
    """

    if len(a) != len(b):
        raise Exception("Error: Vector product require same length")

    sum = 0
    for index in range(len(a)):
        sum += a[index] * b[index]
    return sum


def softmax(numbers):
    """
    Get the softmax of the list of numbers
    """

    exps = [math.exp(num) for num in numbers]
    sum_exps = sum(exps)
    return [exp / sum_exps for exp in exps]


def distance(a, b):
    """
    Calculates the Euclidean distance between vector a and vector b
    """
    if a is None or b is None:
        raise Exception("Error: Neither a nor b can be none.")
    elif len(a) != len(b):
        raise Exception("Error: a and b should have the same dimension")
    sum = 0
    for index in range(len(a)):
        sum += (a[index] - b[index])**2
    return round(math.sqrt(sum), 2)