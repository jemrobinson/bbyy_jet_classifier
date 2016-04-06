import os

class BaseStrategy(object) :
  def __init__( self, output_directory ) :
    self.output_directory = output_directory if output_directory is not None else self.default_output_location
    if not os.path.exists(self.output_directory):
      os.makedirs(self.output_directory)

  def run( self, correct_tree, incorrect_tree, variable_dict ) :
    raise NotImplementedError( "Must be implemented by child class!" )
