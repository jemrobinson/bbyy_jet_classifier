import matplotlib.pyplot as plt
import numpy as np
import os
import rootpy.plotting as rpp

def plot_outputs( strategy ) :
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
  if not os.path.exists("{}/variables/".format(strategy.output_directory)):
    os.makedirs("{}/variables/".format(strategy.output_directory))

  # -- Plot distributions of output classifiers and input variables
  for classifier_name in [ x for x in strategy.test_events.dtype.names if x not in strategy.variable_dict.keys() + ["weight","classID"] ] :
    correct_test = strategy.test_events[ strategy.test_events["classID"] == 0 ][classifier_name]
    correct_test_w = strategy.test_events[ strategy.test_events["classID"] == 0 ]["weight"]
    incorrect_test = strategy.test_events[ strategy.test_events["classID"] == 1 ][classifier_name]
    incorrect_test_w = strategy.test_events[ strategy.test_events["classID"] == 1 ]["weight"]#/float(len(incorrect_test))

    figure = plt.figure(figsize=(6,6), dpi=100)
    axes = plt.axes()
    bins = np.linspace( 0, 1, 50 )
    plt.hist( correct_test, weights=correct_test_w/sum(correct_test_w), bins=bins, histtype="stepfilled", label="Correct (test)", color="blue", alpha=0.5 )
    plt.hist( incorrect_test, weights=incorrect_test_w/sum(incorrect_test_w), bins=bins, histtype="stepfilled", label="Incorrect (test)", color="red", alpha=0.5 )
    plt.legend(loc="upper right")
    plt.xlabel(classifier_name, position=(1., 0), va='bottom', ha='right')
    plt.ylabel("Fraction of events", position=(0, 1.), va='top', ha='right')
    axes.xaxis.set_label_coords(1., -0.15)
    axes.yaxis.set_label_coords(-0.2, 1.)
    figure.savefig( "{}/variables/{}.pdf".format(strategy.output_directory,classifier_name) )
