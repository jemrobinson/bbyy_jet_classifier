import os
from .. import utils


class BaseStrategy(object):

    def __init__(self, output_directory):
        self.name = self.__module__.split(".")[-1]
        self.output_directory = os.path.join(output_directory)
        utils.ensure_directory(self.output_directory)

    def train(self, train_data, classification_variables, variable_dict, sample_name):
        raise NotImplementedError("Must be implemented by child class!")

    def test(self, test_data, classification_variables, training_sample):
        raise NotImplementedError("Must be implemented by child class!")
