from argparse import ArgumentParser

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
    parser = ArgumentParser(description='''
    An Artificial Neural Network
    by Daniel Beckwith and Aditya Nivarthi
    for WPI CS 4341
    ''')

    parser.add_argument('filename')
    parser.add_argument('h', default='h')
    parser.add_argument('hidden_node_count', type=int)
    parser.add_argument('p', default='p')
    parser.add_argument('percent', type=float)
    args = parser.parse_args()

    points, expec_output = read_data(args.filename)

    hold_back = 0.2
    node_count = 5
    hold_back = args.percent
    node_count = args.hidden_node_count

    if hold_back < 0.0 or hold_back > 1.0:
        # Invalid percentage
        raise NameError('Hold back percentage is invalid.')

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
