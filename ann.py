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
        points.append(DataPoint(values[0], values[1], values[2]))

    return points


def classify(data):
    """
    :param data: list(DataPoint)
    :return: list((DataPoint))
    """
    pass


if __name__ == '__main__':

    parser = ArgumentParser(description='Get command line arguments')
    parser.add_argument('filename')

    args = parser.parse_args()

    points = read_data(args.filename)
    for point in points:
        print(point)
