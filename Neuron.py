import operator
from math import exp


def activation_curve(x):
    return 1 / (1 + exp(x))


class Neuron:
    def __init__(self, num_inputs):
        """
        :param num_inputs: int
        :return: Neuron
        """
        self.weights = [0] * num_inputs

    def activate(self, inputs):
        """
        :param inputs: list(float)
        :return: int
        """
        return round(activation_curve(sum(map(operator.mul, self.weights, inputs))))
