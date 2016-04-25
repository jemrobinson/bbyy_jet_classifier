#! /usr/bin/env python
import argparse
from bbyy_jet_classifier import strategies, plotting
import os

if __name__ == "__main__" :

  # -- Parse arguments
  parser = argparse.ArgumentParser( description="Run ML algorithms over ROOT TTree input" )
  parser.add_argument( "--input", type=str, help="input file name", required=True )
  parser.add_argument( "--output", type=str, help="output directory", default=None )
  parser.add_argument( "--correct_tree", metavar="NAME_OF_TREE", type=str, help="name of tree containing correctly identified pairs", default="correct")
  parser.add_argument( "--incorrect_tree", metavar="NAME_OF_TREE", type=str, help="name of tree containing incorrectly identified pairs", default="incorrect")
  parser.add_argument( "--exclude", type=str, metavar="VARIABLE_NAME", nargs="+", help="list of variables to exclude", default=[] )
  parser.add_argument( "--ftrain", type=float, help="fraction of events to use for training", default=0.7 )
  parser.add_argument( "--strategy", type=str, help="strategy to use. Options are: RootTMVA, sklBDT.", default="RootTMVA" )
  args = parser.parse_args()

  # -- Check that input file exists
  if not os.path.isfile( args.input ) : raise FileNotFoundError( "{} does not exist!".format( args.input ) )

  # -- Construct dictionary of available strategies
  if not args.strategy in strategies.__dict__.keys() : raise AttributeError( "{} is not a valid strategy".format( args.strategy ) )

  # -- Run appropriate strategy
  ML_strategy = getattr(strategies,args.strategy)( args.output )
  ML_strategy.load_data( args.input, args.correct_tree, args.incorrect_tree, args.exclude )
  ML_strategy.run( args.ftrain )

  # -- Plot distributions
  plotting.plot_inputs( ML_strategy )
  plotting.plot_outputs( ML_strategy )
