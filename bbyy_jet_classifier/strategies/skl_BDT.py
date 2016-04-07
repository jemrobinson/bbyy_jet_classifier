from . import BaseStrategy
from ..adaptors import root2python
import numpy as np
# import pandas as pd
from root_numpy import rec2array
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report


class sklBDT(BaseStrategy) :
  default_output_location = "output/sklBDT"


  def run( self ):
    # -- Convert already-loaded arrays into test and training samples
    # self.construct_test_training_arrays()

    # -- Construct array of features (X) and array of categories (y)
    X = rec2array( np.concatenate( (self.correct_no_weights, self.incorrect_no_weights) ) )
    y = np.concatenate( (np.zeros(self.correct_no_weights.shape[0]), np.ones(self.incorrect_no_weights.shape[0])) )
    w = rec2array( np.concatenate( (self.correct_weights_only, self.incorrect_weights_only) ) )

    # -- Construct training and test datasets, automatically permuted
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split( X, y, w, test_size=0.5 )

    # -- ANOVA for feature selection (please, know what you're doing)
    # self.feature_selection(5)

    # -- Train:
    print "Training..."
    classifier = GradientBoostingClassifier(n_estimators=200, min_samples_split=2, max_depth=10, verbose=1)
    classifier.fit( X_train, y_train, sample_weight=w_train )

    # -- Test:
    print "Testing..."
    print "Training accuracy = {:.2f}%".format(100 * classifier.score( X_train, y_train, sample_weight=w_train))
    print classification_report( y_train, classifier.predict(X_train), target_names=["correct","incorrect"]  )
    print "Testing accuracy = {:.2f}%".format(100 * classifier.score( X_test, y_test, sample_weight=w_test))
    print classification_report( y_test, classifier.predict(X_test), target_names=["correct","incorrect"]  )

    # # -- Plot:
    # yhat_test = classifier.predict_proba(X_test)
    # plot(yhat_test, y_test, figname="{}/skl_BDT_output.pdf".format(self.output_directory))

    # -- Get list of variables and classifier scores
    variables = [ (k,root2python.char2type[v]) for k,v in self.variable_dict.items()+[("weight","F"),("classID","I"),("classifier","F")] if k != "event_weight" ]
    classifier_score_test = np.hsplit( classifier.predict_proba(X_test), 2 )[1]
    classifier_score_training = np.hsplit( classifier.predict_proba(X_train), 2 )[1]
    # classifier_score_test = np.reshape(classifier.predict(X_test),(-1,1)) -- to get a yes/no answer
    # classifier_score_training = np.reshape(classifier.predict(X_train),(-1,1)) -- to get a yes/no answer


    # -- Construct sliced arrays of test and training events
    test_values_with_score = np.concatenate( ( X_test, np.reshape(w_test,(-1,1)), np.reshape(y_test,(-1,1)), classifier_score_test ), axis=1 )
    training_values_with_score = np.concatenate( ( X_train, np.reshape(w_train,(-1,1)), np.reshape(y_train,(-1,1)), classifier_score_training ), axis=1 )
    test_events_sliced = [ test_values_with_score[...,idx].astype(variable[1]) for idx, variable in enumerate( variables ) ]
    training_events_sliced = [ training_values_with_score[...,idx].astype(variable[1]) for idx, variable in enumerate( variables ) ]


    # -- Construct record arrays for plotting
    self.test_events = np.rec.fromarrays( test_events_sliced, names=[ x[0] for x in variables] )
    self.training_events = np.rec.fromarrays( training_events_sliced, names=[ x[0] for x in variables] )


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


# def plot(yhat_test, y_test, figname="skl_BDT_output.pdf"):
#   """
#   Definition:
#   -----------
#     Plots the output distribution for the testing sample, color-coded by target class
#
#   Args:
#   -----
#     yhat_test = array of predicted class probabilities of dimensions (n_test_events, n_classes) for the testing sample
#     y_test =  array of truth labels {0, 1} of dimensions (n_test_events) for test
#     figname = string, path where the plot will be saved (default = "./output/skl_output.pdf")
#   """
#   import matplotlib.pyplot as plt
#
#   fg = plt.figure()
#   bins = np.linspace(min(yhat_test[:, 1]), max(yhat_test[:, 1]), 40)
#
#   plt.hist(yhat_test[y_test == 0][:, 1],
#     bins = bins, histtype = "stepfilled", label = "correct", color = "blue", alpha = 0.5, normed = True)
#   plt.hist(yhat_test[y_test == 1][:, 1],
#     bins = bins, histtype = "stepfilled", label = "incorrect", color = "red", alpha = 0.5, normed = True)
#
#   plt.legend(loc = "upper center")
#   plt.title("Scikit-Learn Classifier Output")
#   plt.xlabel("Classifier Score")
#   plt.ylabel("Arbitrary Units")
#   # #plt.yscale('log')
#   # plt.show()
#   fg.savefig(figname)
