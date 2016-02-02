from math import exp

import numpy as np


class NeuralNet:
    def __init__(self, *layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        for i in range(len(self.layer_sizes) - 1):
            self.weights.append(
                [[np.random.randn() for input_node in range(self.layer_sizes[i + 1])] for output_node in
                 range(self.layer_sizes[i] + 1)])
        self.activation_function = lambda x: 1 / (1 + exp(-x))
        self.activation_derivative = lambda x: exp(x) * (1 + exp(x)) ** -2

    def classify(self, input):
        return self.calculate(input)[1][-1]

    def calculate(self, input):
        input = input + [1]
        inputs = [input]
        outputs = [input]
        for w in range(len(self.weights)):
            inputs.append([])
            for input_node in range(self.layer_sizes[w + 1]):
                node_input = 0
                for output_node in range(self.layer_sizes[w]):
                    node_input += outputs[-1][output_node] * \
                                  self.weights[w][output_node][input_node]
                inputs[-1].append(node_input)

            outputs.append([self.activation_function(i) for i in inputs[-1]])

        return inputs, outputs

    def learn(self, rate, input_data, output_data):
        errors = [[0 for node in range(layer_size)] for layer_size in self.layer_sizes]
        for i in range(len(input_data)):
            input_sample = input_data[i] + [1]
            output_sample = output_data[i]

            inputs, outputs = self.calculate(input_sample)

            for output_node in range(self.layer_sizes[-1]):
                errors[-1][output_node] = self.activation_derivative(inputs[-1][output_node]) * \
                                          (output_sample[output_node] - outputs[-1][output_node])

            for layer in reversed(range(1, len(self.layer_sizes) - 1)):
                for output_node in range(self.layer_sizes[layer]):
                    weight_error_sum = 0
                    for input_node in range(self.layer_sizes[layer + 1]):
                        weight_error_sum += self.weights[layer][output_node][input_node] * \
                                            errors[layer + 1][input_node]
                    errors[layer][output_node] = \
                        self.activation_derivative(inputs[layer][output_node]) * weight_error_sum

            for w in range(len(self.weights)):
                for output_node in range(self.layer_sizes[w]):
                    for input_node in range(self.layer_sizes[w + 1]):
                        self.weights[w][output_node][input_node] += \
                            rate * outputs[w][output_node] * errors[w + 1][input_node]
