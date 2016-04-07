from . import BaseStrategy
from ..adaptors import root2python
from sklearn.ensemble import GradientBoostingClassifier

class sklBDT(BaseStrategy) :
  default_output_location = "output/sklBDT"


  def load_data( self, input_filename, correct_tree_name, incorrect_tree_name, excluded_variables ) :
    self.X_train, self.X_test, self.y_train, self.y_test, self.w_train, self.w_test, self.features = \
      root2python.trees2arrays( input_filename, correct_treename, incorrect_treename, excluded_variables )


  def run( self, input_filename, correct_treename, incorrect_treename, excluded_variables ):
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







  def run( self ) :
    # -- Initialise TMVA tools
    ROOT.TMVA.Tools.Instance()
    f_output = ROOT.TFile( "{}/TMVA_output.root".format( self.output_directory ), "RECREATE" )
    factory = ROOT.TMVA.Factory( "TMVAClassification", f_output, "AnalysisType=Classification" )

    # -- Add variables to the factory:
    for variable, v_type in root2numpy.get_tree_variables( self.correct_tree, excluded_variables=self.excluded_variables).items() :
      factory.AddVariable( variable, v_type )
    factory.SetWeightExpression( "event_weight" );

    # -- Pass signal and background trees:
    factory.AddSignalTree( self.correct_tree )
    factory.AddBackgroundTree( self.incorrect_tree )
    factory.PrepareTrainingAndTestTree( ROOT.TCut(""), ROOT.TCut(""), "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random" )

    # -- Define methods:
    BDT_method = factory.BookMethod( ROOT.TMVA.Types.kBDT, "BDT", ":".join(
      [ "NTrees=800", "MinNodeSize=5", "MaxDepth=3", "BoostType=AdaBoost", "AdaBoostBeta=0.5", "SeparationType=GiniIndex", "nCuts=-1" ]
    ) )

    # -- Where stuff actually happens:
    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()

    # -- Organize output:
    if os.path.isdir( "{}/weights".format(self.output_directory) ) :
      shutil.rmtree( "{}/weights".format(self.output_directory) )
    shutil.move( "weights", self.output_directory )









TRAIN_FRAC = 0.7 # assign 70% of events to training, 30% to testing





def feature_selection(X_train, y_train, features, k):
  '''
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
  '''

  # -- Select the k top features, as ranked using ANOVA F-score
  from sklearn.feature_selection import SelectKBest, f_classif
  tf = SelectKBest(score_func=f_classif, k=k)
  Xt = tf.fit_transform(X_train, y_train)
  # print("Shape =", Xt.shape)

  # -- Plot support and return names of top features
  print 'The {} most important features are {}'.format(k, [f for (s, f) in sorted(zip(tf.scores_, features), reverse=True)][:k] )
  # plt.imshow(tf.get_support().reshape(2, -1), interpolation="nearest", cmap=plt.cm.Blues)
  # plt.show()



def plot(yhat_test, y_test, figname = './output/skl_output.pdf'):
  '''
  Definition:
  -----------
    Plots the output distribution for the testing sample, color-coded by target class

  Args:
  -----
    yhat_test = array of predicted class probabilities of dimensions (n_test_events, n_classes) for the testing sample
    y_test =  array of truth labels {0, 1} of dimensions (n_test_events) for test
    figname = string, path where the plot will be saved (default = './output/skl_output.pdf')
  '''
  import matplotlib.pyplot as plt

  fg = plt.figure()
  bins = np.linspace(min(yhat_test[:, 1]), max(yhat_test[:, 1]), 40)

  plt.hist(yhat_test[y_test == 1][:, 1],
    bins = bins, histtype = 'stepfilled', label = 'signal', color = 'blue', alpha = 0.5, normed = True)
  plt.hist(yhat_test[y_test == 0][:, 1],
    bins = bins, histtype = 'stepfilled', label = 'bkg', color = 'red', alpha = 0.5, normed = True)

  plt.legend(loc = 'upper center')
  plt.title('Scikit-Learn Classifier Output')
  plt.xlabel('Classifier Score')
  plt.ylabel('Arbitrary Units')
  #plt.yscale('log')
  plt.show()
  fg.savefig(figname)
