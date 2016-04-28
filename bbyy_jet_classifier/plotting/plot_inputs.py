import matplotlib.pyplot as plt
import numpy as np
import os
import rootpy.plotting as rpp

def plot_inputs( strategy, X, y, w, process ) :
  """
  Definition:
  -----------
    Plots the distributions of input variables, color-coded by target class

  Args:
  -----
    strategy = a classification method -- here either sklBDT or RootTMVA 
    X = the feature matrix (ndarray)
    y = the target array
    w = the array of weights
    process = string, either 'training' or 'testing', usually

  """
  rpp.set_style('ATLAS', mpl=True)

  # -- Ensure output directory exists
  strategy.ensure_directory( "{}/{}/".format(strategy.output_directory, process) )

  # -- Plot distributions of input variables
  for i, variable in enumerate(strategy.classification_variables) : 
    data_correct = X_train[y_train == 0][:, i]
    data_incorrect = X_train[y_train == 1][:, i]

    figure = plt.figure(figsize=(6,6), dpi=100)
    axes = plt.axes()
    bins = np.linspace( min([min(data_correct),min(data_incorrect)]), max([max(data_correct),max(data_incorrect)]), 50 )

    y_1, _, _ = plt.hist( data_correct, weights = w_train[y_train == 0]/float(sum(w_train[y_train == 0])), 
      bins=bins, histtype="stepfilled", label="Correct", color="blue", alpha=0.5)

    y_2, _, _ = plt.hist( data_incorrect, weights = w_train[y_train == 1]/float(sum(w_train[y_train == 1])), 
      bins=bins, histtype="stepfilled", label="Incorrect", color="red", alpha=0.5)

    plt.legend(loc="upper right")
    plt.xlabel(variable, position=(1., 0), va='bottom', ha='right')
    plt.ylabel("Fraction of events", position=(0, 1.), va='top', ha='right')
    axes.xaxis.set_label_coords(1., -0.15)
    axes.yaxis.set_label_coords(-0.2, 1.)
    axes.set_ylim( [0,1.3*max([1e-5,max(y_1),max(y_2)])] )
    figure.savefig( "{}/{}/{}.pdf".format(strategy.output_directory, process, variable) )
