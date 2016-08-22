import os
from .. import utils


class BaseStrategy(object):
    default_output_subdir = None

    def __init__(self, sample_name):
        self.name = self.__module__.split(".")[-1]
        self.output_directory = self.training_location(sample_name)
        utils.ensure_directory(self.output_directory)

    def training_location(self, sample_name):
        return os.path.join("output", self.default_output_subdir, sample_name)  # , "pickle", "sklBDT_clf.pkl")

    def train(self, train_data, classification_variables, variable_dict):
        raise NotImplementedError("Must be implemented by child class!")

    def test(self, data, classification_variables, process, train_location):
        raise NotImplementedError("Must be implemented by child class!")
