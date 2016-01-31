from argparse import ArgumentParser

import numpy as np

from NeuralNet import NeuralNet


def read_data(file_name):
    """
    :description: Reads the file and returns the matrix of input values and column vector of expected outputs.
    :param file_name: str
    :return: matrix[input, input], matrix[output]
    """
    with open(file_name, 'r') as datafile:
        data = datafile.readlines()

    datafile.close()
    points = []
    outputs = []

    for line in data:
        values = line.strip().split(' ')
        points.append([float(values[0]), float(values[1])])
        outputs.append([float(values[2])])

    return points, outputs


def main():
    parser = ArgumentParser(description='''
    An Artificial Neural Network
    by Daniel Beckwith and Aditya Nivarthi
    for WPI CS 4341
    ''')
    parser.add_argument('filename')
    parser.add_argument('nodesorholdback', nargs='?', default='h')
    parser.add_argument('value', nargs='?', type=float)

    args = parser.parse_args()
    points, expec_output = read_data(args.filename)

    hold_back = 0.2
    node_count = 5
    train_len = 0

    if args.nodesorholdback == 'p':
        hold_back = args.value
        if hold_back < 0.0 or hold_back > 1.0:
            # Invalid percentage
            raise NameError('Hold back percentage is invalid.')

    train_len = int(len(points) * (1.0 - hold_back))

    # Data arrays
    train_data = []
    train_out_data = []
    valid_data = []
    valid_out_data = []
    actual_output = []
    errors = []

    # Set up training data arrays
    for i in range(0, train_len):
        train_data.append(points[i])
        train_out_data.append(expec_output[i])

    # Set up validation data arrays
    for i in range(train_len, len(points)):
        valid_data.append(points[i])
        valid_out_data.append(expec_output[i])

    net = NeuralNet(2, node_count, 1)

    net.learn(1e-3, train_data, train_out_data)

    for example in valid_data:
        actual_output.append(net.calculate(example)[1])

    for i in range(len(actual_output)):
        errors.append(expec_output[i] - actual_output[i])

    corr_percent = 1 - sum(abs(x) for x in errors) / len(valid_out_data)

    print(corr_percent)


if __name__ == '__main__':
    main()
