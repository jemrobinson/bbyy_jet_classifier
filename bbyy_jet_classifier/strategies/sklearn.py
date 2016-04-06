from . import BaseStrategy
from ..adaptors import root2numpy

class sklearn(BaseStrategy) :
  default_output_location = "output/sklearn"


  def run( self, correct_tree, incorrect_tree, excluded_variables ) :
    root2numpy.tree2array( correct_tree, excluded_variables )
