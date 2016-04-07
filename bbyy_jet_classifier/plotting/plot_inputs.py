import matplotlib.pyplot as plt
import numpy as np
import os
import rootpy.plotting as rpp

def plot_inputs( strategy ) :
  """
  Definition:
  -----------
    Plots the distributions of input variables for the full sample, color-coded by target class

  Args:
  -----
    strategy = a completed strategy, containing ndarrays of parameter distributions
  """
  rpp.set_style('ATLAS', mpl=True)

  # -- Ensure output directory exists
  strategy.ensure_directory( "{}/variables/".format(strategy.output_directory) )

  # -- Plot distributions of input variables
  for variable in strategy.variable_dict.keys() :
    if variable == "event_weight" : continue
    data_correct, data_incorrect = strategy.correct_array[variable], strategy.incorrect_array[variable]
    figure = plt.figure(figsize=(6,6), dpi=100)
    axes = plt.axes()
    bins = np.linspace( min([min(data_correct),min(data_incorrect)]), max([max(data_correct),max(data_incorrect)]), 50 )
    y_1, _, _ = plt.hist( data_correct, weights=np.ones_like(data_correct)/float(len(data_correct)),
      bins=bins, histtype="stepfilled", label="Correct", color="blue", alpha=0.5)
    y_2, _, _ = plt.hist( data_incorrect, weights=np.ones_like(data_incorrect)/float(len(data_incorrect)),
      bins=bins, histtype="stepfilled", label="Incorrect", color="red", alpha=0.5)
    plt.legend(loc="upper right")
    plt.xlabel(variable, position=(1., 0), va='bottom', ha='right')
    plt.ylabel("Fraction of events", position=(0, 1.), va='top', ha='right')
    axes.xaxis.set_label_coords(1., -0.15)
    axes.yaxis.set_label_coords(-0.2, 1.)
    axes.set_ylim( [0,1.3*max([1e-5,max(y_1),max(y_2)])] )
    figure.savefig( "{}/variables/{}.pdf".format(strategy.output_directory,variable) )
