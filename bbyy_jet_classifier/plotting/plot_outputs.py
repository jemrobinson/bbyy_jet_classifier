import matplotlib.pyplot as plt
import numpy as np
import os
import rootpy.plotting as rpp

def plot_outputs( strategy, yhat, y, w, process, fileID ) :
  """
  Definition:
  -----------
    Plots the classifier output, color-coded by target class

  Args:
  -----
    strategy = a completed strategy, containing ndarrays of parameter distributions
    yhat = 
    y = 
    w = 
    process = string, either 'training' or 'testing', usually
    fileID = 
  """
  rpp.set_style('ATLAS', mpl=True)

  # -- Ensure output directory exists
  strategy.ensure_directory( "{}/{}/".format(strategy.output_directory, process) )

  figure = plt.figure(figsize=(6,6), dpi=100)
  axes = plt.axes()
  bins = np.linspace(min(yhat), max(yhat), 50)

  plt.hist( yhat[y == 0], weights = w[y == 0] / float(sum(w[y == 0])), bins=bins, histtype="stepfilled", label="Correct", color="blue", alpha=0.5 )
  plt.hist( yhat[y == 1], weights = w[y == 1] / float(sum(w[y == 1])), bins=bins, histtype="stepfilled", label="Incorrect", color="red", alpha=0.5 )
  
  plt.legend(loc="upper right")
  plt.xlabel("Classifier Output", position=(1., 0), va='bottom', ha='right')
  plt.ylabel("Fraction of events", position=(0, 1.), va='top', ha='right')
  axes.xaxis.set_label_coords(1., -0.15)
  axes.yaxis.set_label_coords(-0.18, 1.)
  #axes.set_yscale("log", nonposy="clip")
  figure.savefig( "{}/{}/{}_{}.pdf".format( strategy.output_directory, process, "BDT", fileID ) )
