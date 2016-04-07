from . import BaseStrategy
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


class sklBDT(BaseStrategy) :
  default_output_location = "output/sklBDT"


  def run( self ):
    # -- Convert already-loaded arrays into test and training samples
    self.construct_test_training_arrays()

    # -- ANOVA for feature selection (please, know what you're doing)
    # feature_selection(X_train, y_train, features, 5)

    # -- Train:
    print "Training..."
    classifier = GradientBoostingClassifier(n_estimators=200, min_samples_split=2, max_depth=10, verbose=1)
    classifier.fit(self.X_train, self.y_train, sample_weight=self.w_train)

    # -- Test:
    print "Testing..."
    print "Training accuracy = {:.2f}%".format(100 * classifier.score(self.X_train, self.y_train, sample_weight = self.w_train))
    print "Testing accuracy = {:.2f}%".format(100 * classifier.score(self.X_test, self.y_test, sample_weight = self.w_test))
    yhat_test  = classifier.predict_proba(self.X_test )

    # -- Plot:
    plot(yhat_test, self.y_test)


  def construct_test_training_arrays( self, training_fraction=0.7 ) :
    """
    Definition:
    -----------
      Turn input ndarrays into useful machine learning test and training samples

    Args:
    -----
      training_faction = what proportion of the input to use for training

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

    # # -- import root to array
    # files = glob.glob(input_filename)
    # correct_arr = stack_arrays([root2rec(fpath, correct_treename) for fpath in files])
    # incorrect_arr = stack_arrays([root2rec(fpath, incorrect_treename) for fpath in files])

    # -- dump into pandas and concatenate + assign target value
    correct_df = pd.DataFrame( self.correct_array )
    correct_df["classID"] = 1
    incorrect_df = pd.DataFrame( self.incorrect_array )
    incorrect_df["classID"] = 0
    df = pd.concat([correct_df, incorrect_df], ignore_index= True)

    # -- create y
    y = df["classID"].values
    weights = df["event_weight"].values

    # -- create X:
    start = 0
    self.features = self.variable_dict.keys()
    X = np.zeros((df.shape[0], len(self.features)))
    unflattened = [df[b] for b in self.features]

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
    self.X_train = X[:n_training_examples]
    self.X_test  = X[n_training_examples:]
    self.y_train = y[:n_training_examples]
    self.y_test  = y[n_training_examples:]
    self.w_train = weights[:n_training_examples]
    self.w_test  = weights[n_training_examples:]

    # return X_train, X_test, y_train, y_test, w_train, w_test, branch_names



def feature_selection(X_train, y_train, features, k):
  """
  Definition:
  -----------
    !! ONLY USED FOR INTUITION, IT'S USING A LINEAR MODEL TO DETERMINE IMPORTANCE !!
    Gives an approximate ranking of variable importance and prints out the top k

  Args:
  -----
    X_train = matrix X of dimensions (n_train_events, n_features) for training
    y_train = array of truth labels {0, 1} of dimensions (n_train_events) for training
    features = names of features used for training in the order in which they were inserted into X
    k = int, the function will print the top k features in order of importance
  """

  # -- Select the k top features, as ranked using ANOVA F-score
  from sklearn.feature_selection import SelectKBest, f_classif
  tf = SelectKBest(score_func=f_classif, k=k)
  Xt = tf.fit_transform(X_train, y_train)
  # print("Shape =", Xt.shape)

  # -- Plot support and return names of top features
  print "The {} most important features are {}".format(k, [f for (s, f) in sorted(zip(tf.scores_, features), reverse=True)][:k] )
  # plt.imshow(tf.get_support().reshape(2, -1), interpolation="nearest", cmap=plt.cm.Blues)
  # plt.show()



def plot(yhat_test, y_test, figname="./output/skl_output.pdf"):
  """
  Definition:
  -----------
    Plots the output distribution for the testing sample, color-coded by target class

  Args:
  -----
    yhat_test = array of predicted class probabilities of dimensions (n_test_events, n_classes) for the testing sample
    y_test =  array of truth labels {0, 1} of dimensions (n_test_events) for test
    figname = string, path where the plot will be saved (default = "./output/skl_output.pdf")
  """
  import matplotlib.pyplot as plt

  fg = plt.figure()
  bins = np.linspace(min(yhat_test[:, 1]), max(yhat_test[:, 1]), 40)

  plt.hist(yhat_test[y_test == 1][:, 1],
    bins = bins, histtype = "stepfilled", label = "signal", color = "blue", alpha = 0.5, normed = True)
  plt.hist(yhat_test[y_test == 0][:, 1],
    bins = bins, histtype = "stepfilled", label = "bkg", color = "red", alpha = 0.5, normed = True)

  plt.legend(loc = "upper center")
  plt.title("Scikit-Learn Classifier Output")
  plt.xlabel("Classifier Score")
  plt.ylabel("Arbitrary Units")
  #plt.yscale('log')
  plt.show()
  fg.savefig(figname)
