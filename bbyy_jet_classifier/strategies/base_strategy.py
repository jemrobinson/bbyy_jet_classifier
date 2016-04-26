from ..adaptors import root2python
from root_numpy import root2rec
import os

class BaseStrategy(object) :

  def __init__( self, output_directory ) :
    self.output_directory = output_directory if output_directory is not None else self.default_output_location
    self.ensure_directory( self.output_directory )


  def ensure_directory( self, directory ) :
    if not os.path.exists(directory):
      os.makedirs(directory)


  def load_data( self, input_filename, correct_treename, incorrect_treename, excluded_variables ) :
    self.variable_dict = root2python.get_branch_info( input_filename, correct_treename, excluded_variables )
    self.correct_array = root2rec( input_filename, correct_treename, branches=self.variable_dict.keys() )
    self.incorrect_array = root2rec( input_filename, incorrect_treename, branches=self.variable_dict.keys() )
    self.classification_variables = sorted( [ name for name in self.variable_dict.keys() if name != "event_weight" ] )
    self.correct_no_weights = self.correct_array[self.classification_variables]
    self.incorrect_no_weights = self.incorrect_array[self.classification_variables]
    self.correct_weights_only = self.correct_array[ ["event_weight"] ]
    self.incorrect_weights_only = self.incorrect_array[ ["event_weight"] ]


  def train_and_test( self, training_fraction ) :
    raise NotImplementedError( "Must be implemented by child class!" )


  def test_only( self ) :
    raise NotImplementedError( "Must be implemented by child class!" )
