class DataPoint:
    def __init__(self, inputs, outputs):
        """
        :param inputs: list(float)
        :param outputs: list(int)
        :return: None
        """
        self.inputs = inputs
        self.outputs = outputs
        self.classification = None

    def __str__(self):
        return 'DataPoint{' + ', '.join('%s: %s' % item for item in vars(self).items()) + '}'
