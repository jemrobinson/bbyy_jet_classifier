#! /usr/bin/env python

import argparse
import os
import ROOT
import strategies

VARIABLE_TYPES = { # map each input feature to its type
    "m_jb" : "F", 
    "pT_jb" : "F", 
    "eta_jb" : "F",
    "Delta_eta_jb" : "F", 
    "Delta_phi_jb" : "F",
    "pT_j" : "F", 
    "eta_j" : "F",
    "MV2c20_FCBE_70" : "I", 
    "MV2c20_FCBE_77" : "I", 
    "MV2c20_FCBE_85" : "I"
  }

if __name__ == "__main__" :

  # -- Parse up arguments
  parser = argparse.ArgumentParser( description="Run ML algorithms over ROOT TTree input" )
  parser.add_argument( "--input", type=str, help="input file name", required=True )
  parser.add_argument( "--signal_tree", type=str, help="name of the signal tree", default = "correct")
  parser.add_argument( "--bkg_tree", type=str, help="name of the background tree", default = "incorrect")
  parser.add_argument( "--output", type=str, help="output directory", default="output" )
  parser.add_argument( "--strategy", type=str, help="strategy to use", default="RootTMVA" )
  args = parser.parse_args()

  # -- Check that input file exists
  if not os.path.isfile( args.input ) : raise FileNotFoundError( "{} does not exist!".format( args.input ) )

  # -- Select appropriate strategy
  if args.strategy == 'RootTMVA':
    cls = strategies.RootTMVA( args.output )

  elif args.strategy == 'sklBDT':
    cls = strategies.sklBDT( args.output )
    
  else:
    raise AttributeError( "{} is not a valid strategy".format( args.strategy ) )

  # -- Run classification on the input file using the selected classifier method
  cls.run( args.input, args.signal_tree, args.bkg_tree, VARIABLE_TYPES )  

  