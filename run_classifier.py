#! /usr/bin/env python
import argparse
import os
import ROOT
from bbyy_jet_classifier import strategies

if __name__ == "__main__" :
  # Set up arguments
  parser = argparse.ArgumentParser( description="Run ML algorithms over ROOT TTree input" )
  parser.add_argument( "--input", type=str, help="input file name" )
  parser.add_argument( "--output", type=str, help="output directory", default=None )
  parser.add_argument( "--strategy", type=str, help="strategy to use", default="RootTMVA" )
  args = parser.parse_args()

  # Check that input file exists
  if not os.path.isfile( args.input ) : raise FileNotFoundError( "{} does not exist!".format( args.input ) )

  # Construct dictionary of available strategies
  if not args.strategy in strategies.__dict__.keys() : raise AttributeError( "{} is not a valid strategy".format( args.strategy ) )

  # Read trees from input file
  f_input = ROOT.TFile( args.input, "READ" )
  correct_tree = f_input.Get( "correct" )
  incorrect_tree = f_input.Get( "incorrect" )

  excluded_variables = [ "event_weight" ]

  # Run appropriate strategy
  ML_strategy = getattr(strategies,args.strategy)( args.output )
  ML_strategy.run( correct_tree, incorrect_tree, excluded_variables )
