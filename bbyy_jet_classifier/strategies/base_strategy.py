import os

class BaseStrategy(object) :

  def __init__( self, output_directory ) :
    self.output_directory = output_directory if output_directory is not None else self.default_output_location
    if not os.path.exists(self.output_directory):
      os.makedirs(self.output_directory)


  def load_data( self, input_filename, correct_treename, incorrect_treename, excluded_variables ) :
    raise NotImplementedError( "Must be implemented by child class!" )


  def run( self ) :
    raise NotImplementedError( "Must be implemented by child class!" )
