#! /usr/bin/env python
import argparse
from bbyy_jet_classifier import strategies, plotting
import logging
import os

if __name__ == "__main__" :
  logging.basicConfig( format="%(levelname)-8s\033[1m%(name)-21s\033[0m: %(message)s" )
  logging.addLevelName( logging.WARNING, "\033[1;35m{:8}\033[1;0m".format(logging.getLevelName(logging.WARNING)) )
  logging.addLevelName( logging.ERROR, "\033[1;31m{:8}\033[1;0m".format(logging.getLevelName(logging.ERROR)) )
  logging.addLevelName( logging.INFO, "\033[1;32m{:8}\033[1;0m".format(logging.getLevelName(logging.INFO)) )
  logging.addLevelName( logging.DEBUG, "\033[1;34m{:8}\033[1;0m".format(logging.getLevelName(logging.DEBUG)) )

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

  # -- Load data for appropriate strategy
  ML_strategy = getattr(strategies,args.strategy)( args.output )
  ML_strategy.load_data( args.input, args.correct_tree, args.incorrect_tree, args.exclude )
  logging.getLogger("RunClassifier").info( "Loaded data for strategy: {}".format(args.strategy) )

  # -- Run appropriate strategy
  if args.ftrain > 0 :
    logging.getLogger("RunClassifier").info( "Preparing to train with {}% of events and then test with the remainder".format( int(100*args.ftrain) ) )
    ML_strategy.train_and_test( args.ftrain )
    # -- Plot distributions
    plotting.plot_training_inputs( ML_strategy )
    plotting.plot_training_outputs( ML_strategy )
  else :
    logging.getLogger("RunClassifier").info( "Preparing to use 100% of sample as testing input" )
    ML_strategy.test_only()
    # -- Plot distributions
    plotting.plot_testing_outputs( ML_strategy, args.input.replace(".root","") )
