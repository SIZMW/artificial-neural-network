import operator


class Neuron:
    def __init__(self, num_inputs):
        """
        :param num_inputs: int
        :return: Neuron
        """
        self.weights = [0] * (num_inputs + 1)

    def activate(self, inputs):
        """
        :param inputs: list(float)
        :return: int
        """
        return sum(map(operator.mul, self.weights, [1] + inputs))
