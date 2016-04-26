from . import BaseStrategy
from ..adaptors import root2python
import array
import os
import shutil
import ROOT
from root_numpy import array2tree, root2rec, tree2array
from numpy.lib import recfunctions

class RootTMVA(BaseStrategy) :
  default_output_location = "output/RootTMVA"


  def train_and_test( self, training_fraction ) :
    # -- Initialise TMVA tools
    ROOT.TMVA.Tools.Instance()
    f_output = ROOT.TFile( "{}/TMVA_output.root".format( self.output_directory ), "RECREATE" )
    factory = ROOT.TMVA.Factory( "TMVAClassification", f_output, "AnalysisType=Classification" )

    # -- Re-construct trees from arrays:
    correct_tree = array2tree( self.correct_array, name="correct" )
    incorrect_tree = array2tree( self.incorrect_array, name="incorrect" )

    # -- Add variables to the factory:
    for v_name, v_type in self.variable_dict.items() :
      if v_name != "event_weight" : factory.AddVariable( v_name, v_type )
    factory.SetWeightExpression( "event_weight" );

    # -- Decide how many events to train/test on
    nTrainSignal, nTrainBackground = int(correct_tree.GetEntries()*training_fraction), int(incorrect_tree.GetEntries()*training_fraction)
    nTestSignal, nTestBackground = int(correct_tree.GetEntries()-nTrainSignal), int(incorrect_tree.GetEntries()-nTrainBackground)

    # -- Pass signal and background trees:
    factory.AddSignalTree( correct_tree )
    factory.AddBackgroundTree( incorrect_tree )
    factory.PrepareTrainingAndTestTree( ROOT.TCut(""), ROOT.TCut(""), ":".join(
      [ "nTrain_Signal={}".format(nTrainSignal), "nTrain_Background={}".format(nTrainBackground),
        "nTest_Signal={}".format(nTestSignal), "nTest_Background={}".format(nTestBackground),
        "SplitMode=Random" ]
    ))

    # -- Define methods:
    BDT_method = factory.BookMethod( ROOT.TMVA.Types.kBDT, "BDT", ":".join(
      [ "NTrees=800", "MinNodeSize=5", "MaxDepth=10", "BoostType=Grad", "SeparationType=GiniIndex" ]
    ))

    # -- Where stuff actually happens:
    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()

    # -- Organize output:
    if os.path.isdir( "{}/weights".format(self.output_directory) ) :
      shutil.rmtree( "{}/weights".format(self.output_directory) )
    shutil.move( "weights", self.output_directory )

    # -- Load test and training trees into arrays
    f_output.Close()
    branches = [ x.replace("event_weight","weight") for x in self.variable_dict.keys() ] + [ "BDT", "classID" ]
    self.test_events = root2rec( "{}/TMVA_output.root".format(self.output_directory), "TestTree", branches=branches )
    self.training_events = root2rec( "{}/TMVA_output.root".format(self.output_directory), "TrainTree", branches=branches )


  def test_only( self ) :
    # -- Setup dictionary of variable names to ROOT-accessible arrays
    variables = {}

    # -- Construct reader and add variables to it:
    reader = ROOT.TMVA.Reader()
    for v_name in self.classification_variables :
      variables[v_name] = array.array( root2python.CHAR_2_ARRAYTYPE[self.variable_dict[v_name]] ,[0])
      reader.AddVariable( v_name, variables[v_name] )
    variables["BDT"] = array.array( "f", [0] )

    # -- Load TMVA results
    reader.BookMVA( "BDT", "{}/weights/TMVAClassification_BDT.weights.xml".format(self.output_directory) )

    # -- Re-construct trees from arrays:
    correct_tree = array2tree( self.correct_array, name="correct" )
    incorrect_tree = array2tree( self.incorrect_array, name="incorrect" )

    # -- Calculate BDT output and add to trees
    for tree in [ correct_tree, incorrect_tree ] :
      BDT_branch = tree.Branch( "BDT", variables["BDT"], "D" )
      for event in tree :
        for v_name in self.classification_variables : variables[v_name][0] = getattr( event, v_name )
        variables["BDT"][0] = reader.EvaluateMVA("BDT")
        BDT_branch.Fill()

    # -- Construct record arrays of correct/incorrect events
    self.test_correct_events = tree2array( correct_tree, branches=self.variable_dict.keys()+["BDT"] )
    self.test_incorrect_events = tree2array( incorrect_tree, branches=self.variable_dict.keys()+["BDT"] )
    self.test_correct_events.dtype.names = [ x.replace("event_weight","weight") for x in self.test_correct_events.dtype.names ]
    self.test_incorrect_events.dtype.names = [ x.replace("event_weight","weight") for x in self.test_incorrect_events.dtype.names ]
