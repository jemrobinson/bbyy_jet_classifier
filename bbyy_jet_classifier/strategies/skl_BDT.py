from . import BaseStrategy
from ..adaptors import root2python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


class sklBDT(BaseStrategy) :
  default_output_location = "output/sklBDT"


  def run( self ):
    # -- Convert already-loaded arrays into test and training samples
    self.construct_test_training_arrays()

    # -- ANOVA for feature selection (please, know what you're doing)
    # self.feature_selection(5)

    # -- Train:
    print "Training..."
    classifier = GradientBoostingClassifier(n_estimators=200, min_samples_split=2, max_depth=10, verbose=1)
    classifier.fit(self.X_train, self.y_train, sample_weight=self.w_train)

    # -- Test:
    print "Testing..."
    print "Training accuracy = {:.2f}%".format(100 * classifier.score(self.X_train, self.y_train, sample_weight = self.w_train))
    print "Testing accuracy = {:.2f}%".format(100 * classifier.score(self.X_test, self.y_test, sample_weight = self.w_test))
    yhat_test = classifier.predict_proba(self.X_test)
    # print "yhat:",zip( yhat_test, self.y_test )

    # -- Plot:
    plot(yhat_test, self.y_test, figname="{}/skl_BDT_output.pdf".format(self.output_directory))

    # -- Get list of variables and classifier scores
    variables = [ (k,root2python.char2type[v]) for k,v in self.variable_dict.items()+[("weight","F"),("classID","I"),("classifier","F")] if k != "event_weight" ]
    classifier_score_test = np.hsplit( classifier.predict_proba(self.X_test), 2 )[1]
    classifier_score_training = np.hsplit( classifier.predict_proba(self.X_train), 2 )[1]

    # -- Construct sliced arrays of test and training events
    # test_values_with_score = np.append( self.X_test, np.append(np.reshape(self.y_test,(-1,1)), classifier_score_test, axis=1), axis=1 )
    # training_values_with_score = np.append( self.X_train, np.append(np.reshape(self.y_train,(-1,1)), classifier_score_train, axis=1), axis=1 )
    test_values_with_score = np.concatenate( ( self.X_test, np.reshape(self.w_test,(-1,1)), np.reshape(self.y_test,(-1,1)), classifier_score_test ), axis=1 )
    # print variables, len(variables), test_values_with_score.shape
    training_values_with_score = np.concatenate( ( self.X_train, np.reshape(self.w_train,(-1,1)), np.reshape(self.y_train,(-1,1)), classifier_score_training ), axis=1 )
    test_events_sliced = [ test_values_with_score[...,idx].astype(variable[1]) for idx, variable in enumerate( variables ) ]
    training_events_sliced = [ training_values_with_score[...,idx].astype(variable[1]) for idx, variable in enumerate( variables ) ]


    # -- Construct record arrays for plotting
    self.test_events = np.rec.fromarrays( test_events_sliced, names=[ x[0] for x in variables] )
    self.training_events = np.rec.fromarrays( training_events_sliced, names=[ x[0] for x in variables] )


  def construct_test_training_arrays( self, training_fraction=0.5 ) :
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

    # -- dump into pandas and concatenate + assign target value
    correct_df = pd.DataFrame( self.correct_array )
    correct_df["classID"] = 0
    incorrect_df = pd.DataFrame( self.incorrect_array )
    incorrect_df["classID"] = 1
    df = pd.concat([correct_df, incorrect_df], ignore_index= True)

    # print df
    # -- permute into a random order (resetting the index) to mix signal and background
    df = df.sample(frac=1).reset_index(drop=True)
    # print df


    # -- create y
    y = df["classID"].values
    weights = df["event_weight"].values

    # -- create X:
    start = 0
    self.features = [ x for x in self.variable_dict.keys() if x != "event_weight" ]
    X = np.zeros((df.shape[0], len(self.features)))
    unflattened = [df[b] for b in self.features]

    # -- fill X with event-by-event values
    for event in zip(*unflattened):
      event = np.array(event).T
      X[start:(start + event.shape[0])] = event
      start += event.shape[0]

    # # -- randomly shuffle samples so that we train on both signal and background events
    # ix = range(X.shape[0])
    # np.random.shuffle(ix)
    # X = X[ix]             # redefine X as shuffled version of itself
    # y = y[ix]             # redefine y as shuffled version of itself
    # weights = weights[ix] # redefine weights as shuffled version of itself

    # -- split into training and testing according to TRAIN_FRAC
    n_training_examples = int(training_fraction * X.shape[0])
    self.X_train = X[:n_training_examples]
    self.X_test  = X[n_training_examples:]
    self.y_train = y[:n_training_examples]
    self.y_test  = y[n_training_examples:]
    self.w_train = weights[:n_training_examples]
    self.w_test  = weights[n_training_examples:]



  def feature_selection(self, k ):
    """
    Definition:
    -----------
      !! ONLY USED FOR INTUITION, IT'S USING A LINEAR MODEL TO DETERMINE IMPORTANCE !!
      Gives an approximate ranking of variable importance and prints out the top k

    Args:
    -----
      k = int, the function will print the top k features in order of importance
    """

    # -- Select the k top features, as ranked using ANOVA F-score
    from sklearn.feature_selection import SelectKBest, f_classif
    tf = SelectKBest(score_func=f_classif, k=k)
    Xt = tf.fit_transform(self.X_train, self.y_train)
    # print("Shape =", Xt.shape)

    # -- Plot support and return names of top features
    print "The {} most important features are {}".format(k, [f for (s, f) in sorted(zip(tf.scores_, self.features), reverse=True)][:k] )
    # plt.imshow(tf.get_support().reshape(2, -1), interpolation="nearest", cmap=plt.cm.Blues)
    # plt.show()



def plot(yhat_test, y_test, figname="skl_BDT_output.pdf"):
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

  plt.hist(yhat_test[y_test == 0][:, 1],
    bins = bins, histtype = "stepfilled", label = "correct", color = "blue", alpha = 0.5, normed = True)
  plt.hist(yhat_test[y_test == 1][:, 1],
    bins = bins, histtype = "stepfilled", label = "incorrect", color = "red", alpha = 0.5, normed = True)

  plt.legend(loc = "upper center")
  plt.title("Scikit-Learn Classifier Output")
  plt.xlabel("Classifier Score")
  plt.ylabel("Arbitrary Units")
  # #plt.yscale('log')
  # plt.show()
  fg.savefig(figname)
