import os
import logging

class BaseStrategy(object):

	def __init__(self, output_directory):
		self.name = self.__module__.split('.')[-1]
		self.output_directory = output_directory if output_directory is not None else self.default_output_location
		self.ensure_directory(self.output_directory)

	def ensure_directory(self, directory):
		if not os.path.exists(directory):
			os.makedirs(directory)

	def train(self, X_train, y_train, w_train, classification_variables, variable_dict):
		raise NotImplementedError("Must be implemented by child class!")


	def test(self, X_test, y_test, w_test, classification_variables, process):
		raise NotImplementedError("Must be implemented by child class!")
