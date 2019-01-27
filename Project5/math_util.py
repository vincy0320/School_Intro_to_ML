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

