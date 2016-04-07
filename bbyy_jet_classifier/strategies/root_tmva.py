from . import BaseStrategy
from ..adaptors import root2python
import os
import shutil
import ROOT

class RootTMVA(BaseStrategy) :
  default_output_location = "output/RootTMVA"


  def load_data( self, input_filename, correct_tree_name, incorrect_tree_name, excluded_variables ) :
    # -- Import data by reading trees from .root file
    f_input = ROOT.TFile( input_filename, "READ" )
    self.correct_tree = f_input.Get( "correct" )
    self.incorrect_tree = f_input.Get( "incorrect" )
    self.excluded_variables = excluded_variables


  def run( self ) :
    # -- Initialise TMVA tools
    ROOT.TMVA.Tools.Instance()
    f_output = ROOT.TFile( "{}/TMVA_output.root".format( self.output_directory ), "RECREATE" )
    factory = ROOT.TMVA.Factory( "TMVAClassification", f_output, "AnalysisType=Classification" )

    # -- Add variables to the factory:
    for variable, v_type in root2python.get_tree_variables( self.correct_tree, excluded_variables=self.excluded_variables).items() :
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
