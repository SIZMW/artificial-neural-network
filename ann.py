from DataPoint import DataPoint
from argparse import ArgumentParser


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


def main():
    parser = ArgumentParser(description='''
    An Artificial Neural Network
    by Daniel Beckwith and Aditya Nivarthi
    for WPI CS 4341
    ''')
    parser.add_argument('filename')

    args = parser.parse_args()

    points = read_data(args.filename)
    for point in points:
        print(point)


if __name__ == '__main__':
    main()
