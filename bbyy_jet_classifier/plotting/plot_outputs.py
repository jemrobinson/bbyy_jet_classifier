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
    test_events_correct = strategy.test_events[ strategy.test_events["classID"] == 0 ][classifier_name]
    test_weights_correct = strategy.test_events[ strategy.test_events["classID"] == 0 ]["weight"]
    test_events_incorrect = strategy.test_events[ strategy.test_events["classID"] == 1 ][classifier_name]
    test_weights_incorrect = strategy.test_events[ strategy.test_events["classID"] == 1 ]["weight"]

    plot_classifier( test_events_correct, test_weights_correct, test_events_incorrect, test_weights_incorrect, classifier_name, strategy.output_directory+"/training" )
    plot_ROC( test_events_correct, test_weights_correct, test_events_incorrect, test_weights_incorrect, classifier_name, strategy.classifier_range, strategy.output_directory+"/training" )



def plot_testing_outputs( strategy, input_name ) :
  """
  Definition:
  -----------
    Plots the output distribution for the testing sample, color-coded by target class

  Args:
  -----
    strategy = a completed strategy, containing ndarrays of parameter distributions
  """
  rpp.set_style("ATLAS", mpl=True)

  # -- Ensure output directory exists
  strategy.ensure_directory( "{}/testing/".format(strategy.output_directory) )

  # -- Plot distributions of output classifiers and Receiver Operating Characteristic (ROC)
  for classifier_name in [ x for x in strategy.test_correct_events.dtype.names if x not in strategy.variable_dict.keys() + ["weight","classID"] ] :
    test_events_correct = strategy.test_correct_events[classifier_name]
    test_weights_correct = strategy.test_correct_events["weight"]
    test_events_incorrect = strategy.test_incorrect_events[classifier_name]
    test_weights_incorrect = strategy.test_incorrect_events["weight"]

    plot_classifier( test_events_correct, test_weights_correct, test_events_incorrect, test_weights_incorrect, classifier_name, strategy.output_directory+"/testing", strategy.input_name )
    plot_ROC( test_events_correct, test_weights_correct, test_events_incorrect, test_weights_incorrect, classifier_name, strategy.classifier_range, strategy.output_directory+"/testing", strategy.input_name )



def plot_classifier( events_correct, weights_correct, events_incorrect, weights_incorrect, classifier_name, output_directory, input_name=None ) :
  """
  Definition:
  -----------
    Plots the Receiver Operating Characteristic (ROC)

  Args:
  -----
    events_correct     = Array of events from the sample which are correct
    weights_correct   = Array of weights from the sample for correct events
    events_incorrect   = Array of events from the sample which are incorrect
    weights_incorrect = Array of weights from the sample for incorrect events
    classifier_name    = Name of the classifier
    output_directory   = Output directory
    input_name         = Name of input sample
  """
  # -- Output classifier distributions
  figure = plt.figure(figsize=(6,6), dpi=100)
  axes = plt.axes()
  bins = np.linspace( min((min(events_correct),min(events_incorrect))), max((max(events_correct),max(events_correct))), 25 )
  plt.hist( events_correct, weights=weights_correct/sum(weights_correct), bins=bins, histtype="stepfilled", label="Correct (testing)", color="blue", alpha=0.5 )
  plt.hist( events_incorrect, weights=weights_incorrect/sum(weights_incorrect), bins=bins, histtype="stepfilled", label="Incorrect (testing)", color="red", alpha=0.5 )
  plt.legend(loc="upper right")
  plt.xlabel(classifier_name, position=(1., 0), va='bottom', ha='right')
  plt.ylabel("Fraction of events", position=(0, 1.), va='top', ha='right')
  axes.xaxis.set_label_coords(1., -0.15)
  axes.yaxis.set_label_coords(-0.2, 1.)
  # axes.xaxis.set_label_coords(1., -0.15)
  # axes.yaxis.set_label_coords(-0.2, 1.)
  figure.savefig( "{}/{}{}.pdf".format( output_directory, classifier_name, ["_{}".format(input_name),""][input_name is None] ) )



def plot_ROC( events_correct, weights_correct, events_incorrect, weights_incorrect, classifier_name, classifier_range, output_directory, input_name=None ) :
  """
  Definition:
  -----------
    Plots the Receiver Operating Characteristic (ROC)

  Args:
  -----
    events_correct     = Array of events from the sample which are correct
    weights_correct   = Array of weights from the sample for correct events
    events_incorrect   = Array of events from the sample which are incorrect
    weights_incorrect = Array of weights from the sample for incorrect events
    classifier_name    = Name of the classifier
    classifier_range   = Range of values that the classifier can take
    output_directory   = Output directory
    input_name         = Name of input sample
  """
  # -- Calculate ROC
  true_positive_rate, false_positive_rate, classifier_cut = [], [], []
  nCorrect, _bins =  np.histogram( events_correct, weights=weights_correct, bins=np.linspace( classifier_range[0], classifier_range[1], 101 ) )
  nIncorrect, _bins =  np.histogram( events_incorrect, weights=weights_incorrect, bins=np.linspace( classifier_range[0], classifier_range[1], 101 ) )
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
  axes.yaxis.set_label_coords(-0.2, 1.)
  figure.savefig( "{}/ROC_{}{}.pdf".format( output_directory, classifier_name, ["_{}".format(input_name),""][input_name is None] ) )
