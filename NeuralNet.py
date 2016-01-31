from math import exp, sqrt
from random import random

import matplotlib.pyplot as plt


class NeuralNet:
    def __init__(self, *layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = None
        self.init_weights()
        self.activation_function = lambda x: 1 / (1 + exp(x))
        self.activation_derivative = lambda x: exp(x) * (1 + exp(x)) ** -2

    def init_weights(self):
        self.weights = []
        for i in range(len(self.layer_sizes) - 1):
            self.weights.append(
                [[random() for output_node in range(self.layer_sizes[i + 1])] for input_node in
                 range(self.layer_sizes[i])])

    def calculate(self, input):
        inputs = [input]
        outputs = [input]
        for w in range(len(self.weights)):
            weights = self.weights[w]
            inputs.append([])
            for input_node in range(self.layer_sizes[w + 1]):
                node_input = 0
                for output_node in range(self.layer_sizes[w]):
                    node_input += outputs[-1][output_node] * \
                                  weights[output_node][input_node]
                inputs[-1].append(node_input)

            outputs.append([self.activation_derivative(i) for i in inputs[-1]])
        return inputs, outputs

    def learn(self, min_error, input_data, output_data):
        graph_data = []
        errors = [[0] * layer_size for layer_size in self.layer_sizes]
        epoch = 0
        rate = 1
        while epoch < 100:
            self.init_weights()

            output_errors = []
            for i in range(len(input_data)):
                input_sample = input_data[i]
                output_sample = output_data[i]

                inputs, outputs = self.calculate(input_sample)

                for output_node in range(self.layer_sizes[-1]):
                    errors[-1][output_node] = self.activation_derivative(inputs[-1][output_node]) * \
                                 (output_sample[output_node] - outputs[-1][output_node])

                for layer in reversed(range(len(self.layer_sizes) - 1)):
                    for input_node in range(self.layer_sizes[layer]):
                        weight_error_sum = 0
                        for output_node in range(self.layer_sizes[layer + 1]):
                            weight_error_sum += self.weights[layer][input_node][output_node] * \
                                                errors[layer + 1][output_node]
                        errors[layer][input_node] = \
                            self.activation_derivative(inputs[layer][input_node]) * weight_error_sum

                output_error = sqrt(sum(o ** 2 for o in outputs[-1]))
                output_errors.append(output_error)
                for w in range(len(self.weights)):
                    for input_node in range(self.layer_sizes[w]):
                        for output_node in range(self.layer_sizes[w + 1]):
                            self.weights[w][input_node][output_node] += \
                                rate * outputs[w][input_node] * errors[w + 1][output_node]

            total_error = sum(output_errors) / len(output_errors)
            graph_data.append(total_error)
            if total_error < min_error:
                break
            epoch += 1

        plt.plot(graph_data)
        plt.show()
