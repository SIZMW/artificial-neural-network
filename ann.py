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
    test_errors = []
    validation_errors = []
    best_epoch = 1
    best_error = 1
    epoch = 0
    best_net = net
    while True:
        epoch += 1
        print("Epoch %d, %d epochs since best" % (epoch, epoch - best_epoch))
        test_error = 0
        test_tests = 0
        validation_error = 0
        validation_tests = 0
        for k in range(int(len(points) / valid_size)):
            valid_data = points[k * valid_size:(k + 1) * valid_size]
            valid_out_data = expec_output[k * valid_size:(k + 1) * valid_size]

            train_data = points[:k * valid_size] + points[(k + 1) * valid_size:]
            train_out_data = expec_output[:k * valid_size] + expec_output[(k + 1) * valid_size:]

            net.learn(0.1, train_data, train_out_data)

            for i in range(len(train_data)):
                test_tests += 1
                if train_out_data[i][0] != round(net.classify(train_data[i])[0]):
                    test_error += 1
            for i in range(len(valid_data)):
                validation_tests += 1
                if valid_out_data[i][0] != round(net.classify(valid_data[i])[0]):
                    validation_error += 1
        test_errors.append(test_error / test_tests)
        validation_errors.append(validation_error / validation_tests)
        if validation_errors[-1] < best_error:
            best_epoch = epoch
            best_error = validation_errors[-1]
            best_net = net.copy()
            print("Best error: %1.2f%%" % (best_error * 100))
        elif epoch - best_epoch >= 200:
            print("Error did not improve in 200 epochs, stopping training")
            break

    print("Best error: %1.2f%% at epoch %d" % (best_error * 100, best_epoch))
    print("Best network: %s" % best_net.weights)

    test_plot, = plt.plot(test_errors, label="Test Error")
    validation_plot, = plt.plot(validation_errors, label="Validation Error")
    plt.legend(handles=[test_plot, validation_plot])
    plt.show()


if __name__ == '__main__':
    main()
