import sys

import matplotlib.pyplot as plt

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
    filename = sys.argv[1]

    hold_back = 0.2
    node_count = 5

    try:
        if len(sys.argv) > 2:
            if sys.argv[2] == 'h':
                node_count = int(sys.argv[3])
                if len(sys.argv) > 4 and sys.argv[4] == 'p':
                    hold_back = float(sys.argv[5])
            elif sys.argv[2] == 'p':
                hold_back = float(sys.argv[3])
        if node_count <= 0 or hold_back <= 0 or hold_back >= 1:
            raise ValueError
    except (IndexError, TypeError, ValueError):
        print('Usage: ann.py <filename> [h <number of hidden nodes>] [p <holdout proportion>]')
        exit()

    print('Using %d hidden nodes and %1.0f%% data holdout' % (node_count, hold_back * 100))

    points, expec_output = read_data(filename)

    net = NeuralNet(2, node_count, 1)

    valid_size = int(len(points) * hold_back)
    graph_data = []
    for epoch in range(1000):
        num_correct = 0
        validation_samples = 0
        for k in range(int(len(points) / valid_size)):
            valid_data = points[k * valid_size:(k + 1) * valid_size]
            valid_out_data = expec_output[k * valid_size:(k + 1) * valid_size]

            train_data = points[:k * valid_size] + points[(k + 1) * valid_size:]
            train_out_data = expec_output[:k * valid_size] + expec_output[(k + 1) * valid_size:]

            net.learn(0.1, train_data, train_out_data)

            for i in range(len(valid_data)):
                validation_samples += 1
                out = net.classify(valid_data[i])[0]
                if valid_out_data[i][0] == round(out):
                    num_correct += 1
        graph_data.append(num_correct / validation_samples)

    plt.plot(graph_data)
    plt.show()


if __name__ == '__main__':
    main()
