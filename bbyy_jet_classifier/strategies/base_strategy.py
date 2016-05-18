import os


class BaseStrategy(object):
    def __init__(self, output_directory):
        self.name = self.__class__.__name__ # set name to name of the derived class
        self.output_directory = os.path.join(output_directory, self.name)
        self.ensure_directory(self.output_directory)

    @staticmethod
    def ensure_directory(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def train(self, train_data, classification_variables, variable_dict):
        raise NotImplementedError("Must be implemented by child class!")

    def test(self, data, classification_variables, process):
        raise NotImplementedError("Must be implemented by child class!")
