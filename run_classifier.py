#! /usr/bin/env python
import argparse
from bbyy_jet_classifier import strategies, plotting
import logging
import os
import numpy as np
from viz import calculate_roc, ROC_plotter, add_curve
import matplotlib.pyplot as plt
import cPickle
 
if __name__ == "__main__":
	logger = logging.getLogger("RunClassifier")

	# -- Parse arguments
	parser = argparse.ArgumentParser(description="Run ML algorithms over ROOT TTree input")
	parser.add_argument("--input", type=str, help="input file name", required=True) 
	parser.add_argument("--output", type=str, help="output directory", default=None)
	parser.add_argument("--correct_tree", metavar="NAME_OF_TREE", type=str, help="name of tree containing correctly identified pairs", default="correct")
	parser.add_argument("--incorrect_tree", metavar="NAME_OF_TREE", type=str, help="name of tree containing incorrectly identified pairs", default="incorrect")
	parser.add_argument("--exclude", type=str, metavar="VARIABLE_NAME", nargs="+", help="list of variables to exclude", default=[])
	parser.add_argument("--ftrain", type=float, help="fraction of events to use for training", default=0.7)
	parser.add_argument("--strategy", type=str, help="strategy to use. Options are: RootTMVA, sklBDT.", default="RootTMVA")
	args = parser.parse_args()

	# -- Check that input file exists
	if not os.path.isfile(args.input): raise FileNotFoundError("{} does not exist!".format(args.input))

	# -- Construct dictionary of available strategies
	if not args.strategy in strategies.__dict__.keys(): raise AttributeError("{} is not a valid strategy".format(args.strategy))

	# -- Load data for appropriate strategy
	ML_strategy = getattr(strategies,args.strategy)(args.output)
	X_train, X_test, y_train, y_test, w_train, w_test, mHmatch_test, pThigh_test = ML_strategy.load_data(
		args.input, args.correct_tree, args.incorrect_tree, args.exclude, args.ftrain)

	# -- Training!
	if args.ftrain > 0:
			logger.info("Preparing to train with {}% of events and then test with the remainder".format(int(100*args.ftrain)))
			
			#-- Plot training distributions
			plotting.plot_inputs(ML_strategy, X_train, y_train, w_train, process='training') # plot the feature distributions

			# -- Train classifier
			ML_strategy.train(X_train, y_train, w_train)

			# -- Plot the classifier output as tested on the training set (only useful if you care to check the performance on the training set)
			yhat_train = ML_strategy.test(X_train, y_train, w_train, process='training')
			plotting.plot_outputs( ML_strategy, yhat_train, y_train, w_train, process='training', fileID=args.input.replace(".root","").split("/")[-1]) 

	else :
		logger.info("Preparing to use 100% of sample as testing input")

	# -- Testing!
	if args.ftrain < 1:

		#-- Plot input testing distributions
		plotting.plot_inputs(ML_strategy, X_test, y_test, w_test, process = 'testing') 

		# -- TEST
		yhat_test = ML_strategy.test(X_test, y_test, w_test, process = 'testing')
		
		# -- Plot output testing distributions from classifier and old strategies
		plotting.plot_outputs(ML_strategy, yhat_test, y_test, w_test, process = 'testing', fileID = args.input.replace(".root","").split("/")[-1] )
		plotting.plot_old_strategy(ML_strategy.output_directory, mHmatch_test, y_test, w_test, 'mHmatch')
		plotting.plot_old_strategy(ML_strategy.output_directory, pThigh_test, y_test, w_test, 'pThigh')

		# -- Add ROC curve and points
		logger.info("Plotting ROC curves...")
		# -- From the older strategies:
		eff_mH_signal = float(sum((mHmatch_test * w_test)[y_test == 1])) / float(sum(w_test[y_test == 1]))
		eff_mH_bkg = float(sum((mHmatch_test * w_test)[y_test == 0])) / float(sum(w_test[y_test == 0]))
		eff_pT_signal = float(sum((pThigh_test * w_test)[y_test == 1])) / float(sum(w_test[y_test == 1]))
		eff_pT_bkg = float(sum((pThigh_test * w_test)[y_test == 0])) / float(sum(w_test[y_test == 0]))
		ML_strategy.ensure_directory("{}/pickle/".format(ML_strategy.output_directory) )
		cPickle.dump({'eff_mH_signal': eff_mH_signal, 'eff_mH_bkg': eff_mH_bkg, 'eff_pT_signal': eff_pT_signal, 'eff_pT_bkg': eff_pT_bkg}, 
			open('{}/pickle/old_strategies_dict.pkl'.format(ML_strategy.output_directory), 'wb'))
		# -- From our ML classifier:
		discs = {}
		add_curve(args.strategy, 'black', calculate_roc(y_test, yhat_test), discs)
		fg = ROC_plotter(discs, min_eff = 0.1, max_eff=1.0, logscale=True)
		plt.plot(eff_mH_signal, 1.0/eff_mH_bkg, marker='o', color='r', label=r'Closest m$_{H}$', linewidth=0) # add point for 'mHmatch' strategy
		plt.plot(eff_pT_signal, 1.0/eff_pT_bkg, marker='o', color='b', label=r'Highest p$_{T}$', linewidth=0) # add point for 'pThigh' strategy
		plt.legend()
		fg.savefig('{}/ROC.pdf'.format(ML_strategy.output_directory))

		# -- Save out ROC curve as pickle for later comparison
		cPickle.dump(discs[args.strategy], open('{}/pickle/{}_ROC.pkl'.format(ML_strategy.output_directory, args.strategy), 'wb'), cPickle.HIGHEST_PROTOCOL)
		
	else: 
		logger.info("100% of the sample was used for training -- no independent testing can be performed.")




