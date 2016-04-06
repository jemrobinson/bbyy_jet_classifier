from . import BaseStrategy
from ..adaptors import root2numpy
import os
import shutil
import ROOT

class RootTMVA(BaseStrategy) :
  default_output_location = "output/RootTMVA"


  def run( self, correct_tree, incorrect_tree, excluded_variables ) :
    # Initialise TMVA tools
    ROOT.TMVA.Tools.Instance()
    f_output = ROOT.TFile( "{}/TMVA_output.root".format( self.output_directory ), "RECREATE" )
    factory = ROOT.TMVA.Factory( "TMVAClassification", f_output, "AnalysisType=Classification" )

    for variable, v_type in root2numpy.get_tree_variables(correct_tree, excluded_variables=excluded_variables).items() :
      factory.AddVariable( variable, v_type )
    factory.SetWeightExpression( "event_weight" );

    factory.AddSignalTree( correct_tree )
    factory.AddBackgroundTree( incorrect_tree )
    factory.PrepareTrainingAndTestTree( ROOT.TCut(""), ROOT.TCut(""), "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random" )

    # Define methods
    BDT_method = factory.BookMethod( ROOT.TMVA.Types.kBDT, "BDT", ":".join(
      [ "NTrees=800", "MinNodeSize=5", "MaxDepth=3", "BoostType=AdaBoost", "AdaBoostBeta=0.5", "SeparationType=GiniIndex", "nCuts=-1" ]
    ) )

    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()

    if os.path.isdir( "{}/weights".format(self.output_directory) ) :
      shutil.rmtree( "{}/weights".format(self.output_directory) )
    shutil.move( "weights", self.output_directory )
