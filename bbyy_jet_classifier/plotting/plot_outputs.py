import matplotlib.pyplot as plt
import numpy as np
import os
import rootpy.plotting as rpp

def plot_training_outputs( strategy ) :
  """
  Definition:
  -----------
    Plots the output distribution for the testing sample, color-coded by target class

  Args:
  -----
    strategy = a completed strategy, containing ndarrays of parameter distributions
  """
  rpp.set_style('ATLAS', mpl=True)

  # -- Ensure output directory exists
  strategy.ensure_directory( "{}/training/".format(strategy.output_directory) )

  # -- Plot distributions of output classifiers and input variables
  for classifier_name in [ x for x in strategy.test_events.dtype.names if x not in strategy.variable_dict.keys() + ["weight","classID"] ] :
    correct_test = strategy.test_events[ strategy.test_events["classID"] == 0 ][classifier_name]
    correct_test_w = strategy.test_events[ strategy.test_events["classID"] == 0 ]["weight"]
    incorrect_test = strategy.test_events[ strategy.test_events["classID"] == 1 ][classifier_name]
    incorrect_test_w = strategy.test_events[ strategy.test_events["classID"] == 1 ]["weight"]

    figure = plt.figure(figsize=(6,6), dpi=100)
    axes = plt.axes()
    bins = np.linspace( min((min(correct_test),min(incorrect_test))), max((max(correct_test),max(correct_test))), 50 )
    plt.hist( correct_test, weights=correct_test_w/sum(correct_test_w), bins=bins, histtype="stepfilled", label="Correct (testing)", color="blue", alpha=0.5 )
    plt.hist( incorrect_test, weights=incorrect_test_w/sum(incorrect_test_w), bins=bins, histtype="stepfilled", label="Incorrect (testing)", color="red", alpha=0.5 )
    plt.legend(loc="upper right")
    plt.xlabel(classifier_name, position=(1., 0), va='bottom', ha='right')
    plt.ylabel("Fraction of events", position=(0, 1.), va='top', ha='right')
    axes.xaxis.set_label_coords(1., -0.15)
    axes.yaxis.set_label_coords(-0.18, 1.)
    figure.savefig( "{}/training/{}.pdf".format(strategy.output_directory,classifier_name) )





def plot_testing_outputs( strategy, input_name ) :
  """
  Definition:
  -----------
    Plots the output distribution for the testing sample, color-coded by target class

  Args:
  -----
    strategy = a completed strategy, containing ndarrays of parameter distributions
  """
  rpp.set_style('ATLAS', mpl=True)

  # -- Ensure output directory exists
  strategy.ensure_directory( "{}/testing/".format(strategy.output_directory) )

  # -- Plot distributions of output classifiers and Receiver Operating Characteristic (ROC)
  for classifier_name in [ x for x in strategy.test_correct_events.dtype.names if x not in strategy.variable_dict.keys() + ["weight","classID"] ] :
    correct_test = strategy.test_correct_events[classifier_name]
    correct_test_w = strategy.test_correct_events["weight"]
    incorrect_test = strategy.test_incorrect_events[classifier_name]
    incorrect_test_w = strategy.test_incorrect_events["weight"]

    # -- Output classifier distributions
    figure = plt.figure(figsize=(6,6), dpi=100)
    axes = plt.axes()
    bins = np.linspace( min((min(correct_test),min(incorrect_test))), max((max(correct_test),max(correct_test))), 25 )
    plt.hist( correct_test, weights=correct_test_w/sum(correct_test_w), bins=bins, histtype="stepfilled", label="Correct (testing)", color="blue", alpha=0.5 )
    plt.hist( incorrect_test, weights=incorrect_test_w/sum(incorrect_test_w), bins=bins, histtype="stepfilled", label="Incorrect (testing)", color="red", alpha=0.5 )
    plt.legend(loc="upper right")
    plt.xlabel(classifier_name, position=(1., 0), va='bottom', ha='right')
    plt.ylabel("Fraction of events", position=(0, 1.), va='top', ha='right')
    axes.xaxis.set_label_coords(1., -0.15)
    axes.yaxis.set_label_coords(-0.18, 1.)
    figure.savefig( "{}/testing/{}_{}.pdf".format( strategy.output_directory, classifier_name, input_name ) )

    # -- Calculate ROC
    true_positive_rate, false_positive_rate, classifier_cut = [], [], []
    nCorrect, _bins =  np.histogram( correct_test, bins=np.linspace( strategy.classifier_range[0], strategy.classifier_range[1], 101 ) )
    nIncorrect, _bins =  np.histogram( incorrect_test, bins=np.linspace( strategy.classifier_range[0], strategy.classifier_range[1], 101 ) )
    for bin_number in range(1,len(_bins)-1) :
      true_positive_rate.append( sum( nCorrect[bin_number:] ) / float(sum(nCorrect)) )
      false_positive_rate.append( sum( nIncorrect[bin_number:] ) / float(sum(nIncorrect)) )
      classifier_cut.append( _bins[bin_number] )
    # -- Plot ROC
    figure = plt.figure(figsize=(6,6), dpi=100)
    axes = plt.axes()
    plt.scatter( false_positive_rate, true_positive_rate, c=classifier_cut )
    plt.plot( [0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6) )
    plt.xlabel("False positive rate", position=(1., 0), va='bottom', ha='right')
    plt.ylabel("True positive rate", position=(0, 1.), va='top', ha='right')
    plt.xlim( 0.0, 1.0 )
    plt.ylim( 0.0, 1.0 )
    axes.xaxis.set_label_coords(1., -0.15)
    axes.yaxis.set_label_coords(-0.18, 1.)
    figure.savefig( "{}/testing/ROC_{}_{}.pdf".format( strategy.output_directory, classifier_name, input_name ) )
