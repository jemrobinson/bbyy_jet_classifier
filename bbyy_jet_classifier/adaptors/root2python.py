from collections import OrderedDict
import glob
import numpy as np
import pandas as pd
import ROOT
from root_numpy import root2rec

class root2python(object) :

  @classmethod
  def get_tree_variables( cls, input_tree, excluded_variables=[] ) :
    """
    Definition:
    -----------
      Retrieve all branches and types from a ROOT tree

    Args:
    -----
      input_tree = a ROOT tree whose branches we are interested in
      excluded_variables = a list of branch names to be excluded from consideration

    Returns:
    --------
      variable_dict = a dictionary which has the branch names to be used for training as keys
    """
    variable_dict = OrderedDict()
    type2char = { "Int_t":"I", "Double_t":"D", "Float_t":"F" }
    for leaf in sorted(input_tree.GetListOfLeaves()) :
      variable_name = leaf.GetName()
      if variable_name not in excluded_variables :
        variable_dict[variable_name] = type2char[leaf.GetTypeName()]
    return variable_dict



  @classmethod
  def get_tree_variables( cls, input_filename, input_treename, excluded_variables=[] ) :
    """
    Definition:
    -----------
      Retrieve all branches and types from a ROOT tree, given the file name and the tree name

    Args:
    -----
      input_tree = a ROOT tree whose branches we are interested in
      excluded_variables = a list of branch names to be excluded from consideration

    Returns:
    --------
      variable_dict = a dictionary which has the branch names to be used for training as keys
    """
    with ROOT.TFile( input_filename, "READ" ) as f_input :
      input_tree = f_input.Get( input_treename )
      variable_dict = cls.get_tree_variables( input_tree, excluded_variables )
    return variable_dict



  @classmethod
  def trees2arrays( cls, input_filename, correct_treename, incorrect_treename, training_fraction=0.7, excluded_variables=[] ):
    """
    Definition:
    -----------
      Turn root input data into useful machine learning ndarrays

    Args:
    -----
      input_filename = path to .root input file, which will have a tree of correct pairs and a tree of incorrect pairs
      correct_treename = name of the signal tree in the input file
      incorrect_treename = name of the bkg tree in the input file
      excluded_variables = a list of branch names to be excluded from the training step

    Returns:
    --------
      X_train = matrix X of dimensions (n_train_events, n_features) for training
      X_test = matrix X of dimensions (n_test_events, n_features) for testing
      y_train = array of truth labels {0, 1} of dimensions (n_train_events) for training
      y_test =  array of truth labels {0, 1} of dimensions (n_test_events) for test
      w_train = array of event weights of dimensions (n_train_events) for training
      w_test = array of event weights of dimensions (n_test_events) for testing
      branch_names = names of features used for training in the order in which they were inserted into X
    """

    # -- import root to array
    files = glob.glob(input_filename)
    correct_arr = np.lib.recfunctions.stack_arrays([root2rec(fpath, correct_treename) for fpath in files])
    incorrect_arr = np.lib.recfunctions.stack_arrays([root2rec(fpath, incorrect_treename) for fpath in files])

    # -- dump into pandas and concatenate + assign target value
    correct_df = pd.DataFrame(correct_arr)
    correct_df["classID"] = 1
    incorrect_df = pd.DataFrame(incorrect_arr)
    incorrect_df["classID"] = 0
    df = pd.concat([correct_df, incorrect_df], ignore_index= True)

    # -- create y
    y = df["classID"].values
    weights = df["event_weight"].values

    # -- create X:
    start = 0
    branch_names = cls.get_tree_variables( input_filename, correct_treename, excluded_variables )
    X = np.zeros((df.shape[0], len(branch_names)))
    unflattened = [df[b] for b in branch_names]

    for i, data in enumerate(zip(*unflattened)):
      data = np.array(data).T
      X[start:(start + data.shape[0])] = data
      start += data.shape[0]

    # -- randomly shuffle samples so that we train on both signal and background events
    ix = range(X.shape[0])
    np.random.shuffle(ix)
    X = X[ix]             # redefine X as shuffled version of itself
    y = y[ix]             # redefine y as shuffled version of itself
    weights = weights[ix] # redefine weights as shuffled version of itself

    # -- split into training and testing according to TRAIN_FRAC
    n_training_examples = int(training_fraction * X.shape[0])
    X_train = X[:n_training_examples]
    y_train = y[:n_training_examples]
    w_train = weights[:n_training_examples]
    X_test  = X[n_training_examples:]
    y_test  = y[n_training_examples:]
    w_test  = weights[n_training_examples:]

    return X_train, X_test, y_train, y_test, w_train, w_test, branch_names
