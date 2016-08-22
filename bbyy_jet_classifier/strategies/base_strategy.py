import os
from ..utils import ensure_directory


class BaseStrategy(object):
    default_output_subdir = None

    def __init__(self, output_directory):
        self.name = self.__module__.split(".")[-1]
        self.output_directory = os.path.join(output_directory, self.default_output_subdir)
        ensure_directory(self.output_directory)

    def train(self, train_data, classification_variables, variable_dict):
        raise NotImplementedError("Must be implemented by child class!")

    def test(self, data, classification_variables, process, train_location):
        raise NotImplementedError("Must be implemented by child class!")
