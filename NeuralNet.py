from Neuron import Neuron


class NeuralNet:
    def __init__(self, *layer_sizes):
        prev_size = layer_sizes[0]
        self.layers = []
        for size in layer_sizes:
            self.layers.append([Neuron(prev_size) for i in range(size)])
            prev_size = size

    def classify(self, inputs):
        inputs = list(inputs)
        for layer in self.layers:
            inputs = [neuron.activate(inputs) for neuron in layer]
        return inputs

    def learn(self):
        pass
