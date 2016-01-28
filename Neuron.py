import operator

_id = 0


class Neuron:
    def __init__(self):
        """
        :return: Neuron
        """
        self.const_weight = 0
        self.weights = {}
        self.input = None
        self.output = None
        global _id
        self._id = _id
        _id += 1

    def connect(self, input_neuron):
        self.weights[input_neuron] = 0

    def activate(self, activation_function, inputs):
        """
        :param inputs: list(float)
        :return: int
        """
        self.input = self.const_weight + sum(map(operator.mul, self.weights.values(), inputs))
        self.output = activation_function(self.input)
        return self.output

    def __hash__(self):
        return _id

    def __eq__(self, other):
        return self._id == other._id

    def __str__(self):
        return "Neuron" + str(self._id)

    def __repr__(self):
        return "Neuron" + str(self._id)
