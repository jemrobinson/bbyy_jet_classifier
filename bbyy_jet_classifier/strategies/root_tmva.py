from . import BaseStrategy
import os
import shutil
import ROOT
from root_numpy import array2tree, root2rec

class RootTMVA(BaseStrategy) :
  default_output_location = "output/RootTMVA"


  def run( self ) :
    # -- Initialise TMVA tools
    ROOT.TMVA.Tools.Instance()
    f_output = ROOT.TFile( "{}/TMVA_output.root".format( self.output_directory ), "RECREATE" )
    factory = ROOT.TMVA.Factory( "TMVAClassification", f_output, "AnalysisType=Classification" )

    # -- Re-construct trees from arrays:
    correct_tree = array2tree( self.correct_array, name="correct" )
    incorrect_tree = array2tree( self.incorrect_array, name="incorrect" )

    # -- Add variables to the factory:
    for variable, v_type in self.variable_dict.items() :
      if variable != "event_weight" : factory.AddVariable( variable, v_type )
    factory.SetWeightExpression( "event_weight" );

    # -- Pass signal and background trees:
    factory.AddSignalTree( correct_tree )
    factory.AddBackgroundTree( incorrect_tree )
    factory.PrepareTrainingAndTestTree( ROOT.TCut(""), ROOT.TCut(""), "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random" )

    # -- Define methods:
    BDT_method = factory.BookMethod( ROOT.TMVA.Types.kBDT, "BDT", ":".join(
      [ "NTrees=800", "MinNodeSize=5", "MaxDepth=10", "BoostType=Grad", "SeparationType=GiniIndex" ]
    ) )
    # [ "NTrees=800", "MinNodeSize=5", "MaxDepth=3", "BoostType=AdaBoost", "AdaBoostBeta=0.5", "SeparationType=GiniIndex", "nCuts=-1" ]

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
