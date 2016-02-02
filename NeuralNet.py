from math import exp

import numpy as np


class NeuralNet:
    def __init__(self, *layer_sizes):
        self.layer_sizes = layer_sizes  # size of each layer
        self.weights = []  # weights between each node in each pair of adjacent layers in the network
        # there is 1 less set of weights than layers
        for i in range(len(self.layer_sizes) - 1):
            # add a set of random weights to each node in the next layer for each node in the current layer
            # an additional set of weights to the next layer is added as bias weights
            self.weights.append(
                [[np.random.randn() for input_node in range(self.layer_sizes[i + 1])] for output_node in
                 range(self.layer_sizes[i] + 1)])
        self.activation_function = lambda x: 1 / (1 + exp(-x))
        self.activation_derivative = lambda x: exp(x) * (1 + exp(x)) ** -2

    def classify(self, input):
        # calculate on the input, then return only the last set of outputs
        return self.calculate(input)[1][-1]

    def calculate(self, input):
        # add a dummy input to match the bias weight
        input = input + [1]
        # the inputs for the first layer are just the input data
        inputs = [input]
        # the output of the first layer is just the input data
        outputs = [input]
        for w in range(len(self.weights)):
            # the inputs for the next layer are the sum of the products of the weights
            # and the output of the previous layer
            inputs.append([])
            for input_node in range(self.layer_sizes[w + 1]):
                node_input = 0
                for output_node in range(self.layer_sizes[w]):
                    node_input += outputs[-1][output_node] * \
                                  self.weights[w][output_node][input_node]
                inputs[-1].append(node_input)

            # the outputs for the next layer are just the activation function applied to its inputs
            outputs.append([self.activation_function(i) for i in inputs[-1]])

        return inputs, outputs

    def learn(self, rate, input_data, output_data):
        # the errors for each node in each layer
        errors = [[0 for node in range(layer_size)] for layer_size in self.layer_sizes]
        for i in range(len(input_data)):
            # add a dummy input for the bias weights
            input_sample = input_data[i] + [1]
            output_sample = output_data[i]

            inputs, outputs = self.calculate(input_sample)

            # set the errors of the output nodes based on the expected output and the actual output
            for output_node in range(self.layer_sizes[-1]):
                errors[-1][output_node] = self.activation_derivative(inputs[-1][output_node]) * \
                                          (output_sample[output_node] - outputs[-1][output_node])

            # set the errors for every other layer (going backwards) based on the errors of the previous layer
            # no need to do this for the input layer since it doesn't have weights
            for layer in reversed(range(1, len(self.layer_sizes) - 1)):
                for output_node in range(self.layer_sizes[layer]):
                    weight_error_sum = 0
                    for input_node in range(self.layer_sizes[layer + 1]):
                        weight_error_sum += self.weights[layer][output_node][input_node] * \
                                            errors[layer + 1][input_node]
                    errors[layer][output_node] = \
                        self.activation_derivative(inputs[layer][output_node]) * weight_error_sum

            # update each of the weights based on the errors, the output of that node, and the learning rate
            for w in range(len(self.weights)):
                for output_node in range(self.layer_sizes[w]):
                    for input_node in range(self.layer_sizes[w + 1]):
                        self.weights[w][output_node][input_node] += \
                            rate * outputs[w][output_node] * errors[w + 1][input_node]

    def copy(self):
        net = NeuralNet(*self.layer_sizes)
        net.weights = self.weights
        return net
