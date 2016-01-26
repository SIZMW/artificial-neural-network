from Neuron import Neuron


class NeuralNet:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        self.input_layer = [Neuron(input_layer_size) for i in range(input_layer_size)]
        self.hidden_layer = [Neuron(input_layer_size) for i in range(hidden_layer_size)]
        self.output_layer = [Neuron(hidden_layer_size) for i in range(output_layer_size)]
        self.layers = (self.input_layer, self.hidden_layer, self.output_layer)

    def classify(self, inputs):
        inputs = list(inputs)
        for layer in self.layers:
            inputs = [neuron.activate(inputs) for neuron in layer]
        return inputs

    def learn(self):
        pass
