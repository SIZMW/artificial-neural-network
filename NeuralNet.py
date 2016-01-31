import numpy as np


class NeuralNet:
    def __init__(self, *layer_sizes):
        self.weights = []
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]))

    @staticmethod
    def activation_function(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def activation_derivative(x):
        return np.exp(x) * (np.exp(x) + 1) ** -2

    def calculate(self, input):
        inputs = []
        outputs = [input]
        for i in range(len(self.weights)):
            inputs.append(np.dot(outputs[-1], self.weights[i]))
            outputs.append(self.activation_function(inputs[-1]))
        return inputs, outputs

    def get_weight_gradients(self, input, output):
        inputs, outputs = self.calculate(input)

        gradients = [np.zeros(len(inputs))] * len(self.weights)

        delta = np.multiply(-(output - outputs[-1]), self.activation_derivative(inputs[-1]))

        for i in reversed(len(gradients)):
            gradients[i] = np.dot(outputs[i - 1], delta)
            delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(inputs[i])

        return gradients

    def learn(self, min_rate, input, output):
        while True:
            gradients = self.get_weight_gradients(input, output)
            rate = np.linalg.norm(gradients[-1]) / 2.0
            if rate < min_rate:
                break
            for i in range(len(self.weights)):
                self.weights[i] -= rate * gradients[i]
