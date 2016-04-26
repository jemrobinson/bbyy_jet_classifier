#! /usr/bin/env python
import argparse
from bbyy_jet_classifier import strategies, plotting
import logging
import os

if __name__ == "__main__" :
  logger = logging.getLogger("RunClassifier")

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

  # -- Run appropriate strategy
  if args.ftrain > 0 :
    logger.info( "Preparing to train with {}% of events and then test with the remainder".format( int(100*args.ftrain) ) )
    ML_strategy.train_and_test( args.ftrain )
    # -- Plot distributions
    plotting.plot_training_inputs( ML_strategy )
    plotting.plot_training_outputs( ML_strategy )
  else :
    logger.info( "Preparing to use 100% of sample as testing input" )
    ML_strategy.test_only()
    # -- Plot distributions
    plotting.plot_testing_outputs( ML_strategy, args.input.replace(".root","") )
