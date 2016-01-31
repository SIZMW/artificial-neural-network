from argparse import ArgumentParser

import numpy as np


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
        outputs.append(float(values[2]))

    return np.asmatrix(points, float), np.asmatrix(outputs, float)


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

    # Do the neural net here


if __name__ == '__main__':
    main()
