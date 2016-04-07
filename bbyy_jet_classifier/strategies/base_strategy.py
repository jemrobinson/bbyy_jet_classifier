from ..adaptors import root2python
from root_numpy import root2rec
import os

class BaseStrategy(object) :

  def __init__( self, output_directory ) :
    self.output_directory = output_directory if output_directory is not None else self.default_output_location
    if not os.path.exists(self.output_directory):
      os.makedirs(self.output_directory)


  def load_data( self, input_filename, correct_treename, incorrect_treename, excluded_variables ) :
    self.variable_dict = root2python.get_branch_info( input_filename, correct_treename, excluded_variables )
    self.correct_array = root2rec( input_filename, correct_treename, branches=self.variable_dict.keys() )
    self.incorrect_array = root2rec( input_filename, incorrect_treename, branches=self.variable_dict.keys() )
    self.correct_no_weights = self.correct_array[ [name for name in self.variable_dict.keys() if name != "event_weight"] ]
    self.incorrect_no_weights = self.incorrect_array[ [name for name in self.variable_dict.keys() if name != "event_weight"] ]
    self.correct_weights_only = self.correct_array[ [name for name in self.variable_dict.keys() if name == "event_weight"] ]
    self.incorrect_weights_only = self.incorrect_array[ [name for name in self.variable_dict.keys() if name == "event_weight"] ]


  def run( self ) :
    raise NotImplementedError( "Must be implemented by child class!" )
