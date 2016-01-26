class DataPoint:
    def __init__(self, input_a, input_b, output):
        """
        :param input_a: float
        :param input_b: float
        :param output: int
        :return: None
        """
        self.inputs = (input_a, input_b)
        self.output = output
        self.classification = None

    def __str__(self):
        return 'DataPoint{' + ', '.join('%s: %s' % item for item in vars(self).items()) + '}'
