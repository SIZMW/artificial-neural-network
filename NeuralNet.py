import math
from random import random

import plotly
from plotly.graph_objs import Scatter, Layout

from Neuron import Neuron


class NeuralNet:
    def __init__(self, activation_function, activation_derivative, *layer_sizes):
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.layers = []
        for size in layer_sizes:
            prev_layer = None if len(self.layers) == 0 else self.layers[-1]
            layer = []
            if prev_layer is not None:
                for i in range(size):
                    neuron = Neuron()
                    layer.append(neuron)
                    for prev_neuron in prev_layer:
                        neuron.connect(prev_neuron)
            self.layers.append(layer)

    def classify(self, inputs):
        inputs = list(inputs)
        for layer in self.layers:
            inputs = [neuron.activate(self.activation_function, inputs) for neuron in layer]
        return list(map(self.activation_function, inputs))

    def learn(self, alpha, epsilon, cutoff, data):
        errors = {}
        # for layer in self.layers:
        #     for neuron in layer:
        #         errors[neuron] = 0
        session_errors = []
        for training_session in range(cutoff):
            for layer in self.layers:
                for neuron in layer:
                    for input_neuron in neuron.weights.keys():
                        neuron.weights[input_neuron] = random() * 1e-2
            total_error = 0
            for example in data:
                self.classify(example.inputs)
                i = 0
                for neuron in self.layers[-1]:
                    errors[neuron] = self.activation_derivative(neuron.input) * (example.outputs[i] - neuron.output)
                    i += 1
                for i in reversed(range(0, len(self.layers) - 1)):
                    layer = self.layers[i]
                    for neuron in layer:
                        errors[neuron] = self.activation_derivative(neuron.input) * \
                                         sum(next_neuron.weights[neuron] * errors[next_neuron] for next_neuron in
                                             self.layers[i + 1])
                for layer in self.layers:
                    for neuron in layer:
                        for input_neuron in neuron.weights.keys():
                            neuron.weights[input_neuron] += alpha * input_neuron.output * errors[neuron]
                total_error += math.sqrt(sum(errors[output_neuron] ** 2 for output_neuron in self.layers[-1]))
            total_error /= len(data)
            session_errors.append(total_error)
            if total_error < epsilon:
                break

        plotly.offline.plot({
            "data": [
                Scatter(x=list(range(len(session_errors))), y=session_errors)
            ],
            "layout": Layout(
                    title="da biznis"
            )
        })
