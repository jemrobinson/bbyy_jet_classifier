#! /usr/bin/env python
import argparse
from bbyy_jet_classifier import strategies, plotting
import logging
import os
import numpy as np
from viz import calculate_roc, ROC_plotter, add_curve
import cPickle

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
  X_train, X_test, y_train, y_test, w_train, w_test = ML_strategy.load_data( args.input, args.correct_tree, args.incorrect_tree, args.exclude, args.ftrain )

  # -- Training!
  if args.ftrain > 0 :
      logger.info( "Preparing to train with {}% of events and then test with the remainder".format( int(100*args.ftrain) ) )
      
      #-- Plot training distributions
      plotting.plot_inputs( ML_strategy, X_train, y_train, w_train, process = 'training') # plot the feature distributions

      # -- Train classifier
      ML_strategy.train(X_train, y_train, w_train)

      # -- Plot the classifier output as tested on the training set (only useful if you care to check the performance on the training set)
      yhat_train = ML_strategy.test(X_train, y_train, w_train, process = 'training')
      plotting.plot_outputs( ML_strategy, yhat_train, y_train, w_train, process = 'training', fileID = args.input.replace(".root","")) 

  else :
    logger.info( "Preparing to use 100% of sample as testing input" )

  # -- Testing!
  if args.ftrain < 1:

    #-- Plot testing distributions
    plotting.plot_inputs( ML_strategy, X_test, y_test, w_test, process = 'testing') # plot the feature distributions

    yhat_test = ML_strategy.test(X_test, y_test, w_test, process = 'testing')
    
    # -- Plot testing distributions
    plotting.plot_outputs( ML_strategy, yhat_test, y_test, w_test, process = 'testing', fileID = args.input.replace(".root","") )

    # -- Add ROC curve
    logger.info( "Plotting ROC curves..." )
    discs = {}
    add_curve(args.strategy, 'black', calculate_roc(y_test, yhat_test), discs)
    fg = ROC_plotter(discs, min_eff = 0.1, max_eff=1.0, logscale=True)
    fg.savefig('{}/ROC.pdf'.format(ML_strategy.output_directory))

    # -- Save out ROC curve as pickle for later comparison
    ML_strategy.ensure_directory( "{}/pickle/".format(ML_strategy.output_directory) )
    cPickle.dump(discs[args.strategy], open('{}/pickle/{}_ROC.pkl'.format(ML_strategy.output_directory, args.strategy), 'wb'), cPickle.HIGHEST_PROTOCOL)

  else: 
    logger.info("100% of the sample was used for training -- no independent testing can be performed.")




