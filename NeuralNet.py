import numpy as np


class NeuralNet:
    def __init__(self, *layer_sizes):
        self.weights = []
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]))
        self.activation_function = lambda x: np.power(np.add(1, np.exp(-x)), -1)
        self.activation_derivative = lambda x: np.multiply(np.exp(x), np.power(np.add(np.exp(x), 1), -2))

    def calculate(self, input):
        inputs = [input]
        outputs = []
        for i in range(len(self.weights)):
            outputs.append(self.activation_function(inputs[-1]))
            inputs.append(np.dot(outputs[-1], self.weights[i]))
        outputs.append(self.activation_function(inputs[-1]))
        return inputs, outputs

    def get_weight_gradients(self, input, output):
        inputs, outputs = self.calculate(input)

        gradients = [None] * len(self.weights)

        delta = np.multiply(-(output - outputs[-1]), self.activation_derivative(inputs[-1]))

        for i in reversed(range(len(gradients))):
            gradients[i] = np.dot(outputs[i].T, delta)
            if i == 0: break
            delta = np.dot(np.dot(delta, self.weights[i].T), self.activation_derivative(inputs[i]))

        return gradients

    def learn(self, min_rate, input, output):
        while True:
            gradients = self.get_weight_gradients(input, output)
            rate = np.linalg.norm(gradients[-1]) / 2.0
            if rate < min_rate:
                break
            for i in range(len(self.weights)):
                print(self.weights[i].shape)
                print(rate)
                print(gradients[i].shape)
                self.weights[i] -= rate * gradients[i]
