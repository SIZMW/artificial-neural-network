from DataPoint import DataPoint


def read_data(file_name):
    """
    :param file_name: str
    :return: list(DataPoint)
    """
    with open(file_name, "r") as datafile:
        data = datafile.readlines()

    datafile.close()
    points = []

    for i in range(0, len(data)):
        values = data[i].strip().split(' ')
        points.append(DataPoint(values[0], values[1], values[2]))

    return points


def classify(data):
    """
    :param data: list(DataPoint)
    :return: list((DataPoint))
    """
    pass


if __name__ == '__main__':

    points = read_data("data.txt")
    for i in range(len(points)):
        print(points[i])
