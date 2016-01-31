import numpy as np
import matplotlib.pyplot as plt


class NeuralNet:
    def __init__(self, *layer_sizes):
        self.weights = []
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]))
        self.activation_function = lambda x: np.power(np.add(1, np.exp(-x)), -1)
        self.activation_derivative = lambda x: np.multiply(np.exp(x), np.power(np.add(np.exp(x), 1), -2))

    def calculate(self, input):
        inputs = [input]
        outputs = [input]
        for i in range(len(self.weights)):
            inputs.append(np.dot(outputs[-1], self.weights[i]))
            outputs.append(self.activation_function(inputs[-1]))
        return inputs, outputs

    def learn(self, min_rate, input, output):
        graph_data = []
        epoch = 0
        while epoch < 100:
            inputs, outputs = self.calculate(input)

            gradients = [None] * len(self.weights)
            delta = np.multiply(-(output - outputs[-1]), self.activation_derivative(inputs[-1]))
            for i in reversed(range(len(gradients))):
                gradients[i] = np.dot(outputs[i].T, delta)
                if i == 0:
                    break
                delta = np.multiply(np.dot(delta, self.weights[i].T), self.activation_derivative(inputs[i]))

            error = np.linalg.norm(outputs[-1])
            graph_data.append(error)
            rate = error / 2.0
            if rate < min_rate:
                break
            for i in range(len(self.weights)):
                self.weights[i] -= rate * gradients[i]
            epoch += 1

        plt.plot(graph_data)
        plt.show()
