from argparse import ArgumentParser
from math import exp

from DataPoint import DataPoint
from NeuralNet import NeuralNet


def read_data(file_name):
    """
    :param file_name: str
    :return: list(DataPoint)
    """
    with open(file_name, 'r') as datafile:
        data = datafile.readlines()

    datafile.close()
    points = []

    for line in data:
        values = line.strip().split(' ')
        points.append(DataPoint([float(values[0]), float(values[1])], [int(float(values[2]))]))

    return points


def classify(data):
    """
    :param data: list(DataPoint)
    :return: list((DataPoint))
    """
    pass


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

    points = read_data(args.filename)

    hold_back = 0.2
    node_count = 5
    train_len = 0

    if args.nodesorholdback == 'p':
        hold_back = args.value
        if hold_back < 0.0 or hold_back > 1.0:
            # Invalid percentage
            raise NameError('Hold back percentage is invalid.')

    train_len = int(len(points) * (1.0 - hold_back))

    net = NeuralNet(lambda x: 1 / (1 + exp(-x)), lambda x: exp(x) * (exp(x) + 1) ** -2, 2, node_count, 1)
    net.learn(1e-6, 1e-2, 100, points[:train_len])

    for i in range(train_len):
        point = points[i]
        point.classification = list(map(round, net.classify(point.inputs)))
        print(point)


if __name__ == '__main__':
    main()
