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
        points.append(DataPoint(float(values[0]), float(values[1]), int(float(values[2]))))

    return points


def classify(data):
    """
    :param data: list(DataPoint)
    :return: list((DataPoint))
    """
    pass


def threshold_curve(x):
    return 1 / (1 + exp(-x))


def main():
    parser = ArgumentParser(description='''
    An Artificial Neural Network
    by Daniel Beckwith and Aditya Nivarthi
    for WPI CS 4341
    ''')
    parser.add_argument('filename')

    args = parser.parse_args()

    points = read_data(args.filename)

    net = NeuralNet(2, 5, 1)
    net.learn()

    for point in points:
        point.classification = round(threshold_curve(net.classify(point.inputs)[0]))
        print(point)


if __name__ == '__main__':
    main()
