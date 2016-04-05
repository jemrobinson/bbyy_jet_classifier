from . import BaseStrategy
import shutil
import ROOT

class RootTMVA(BaseStrategy) :

  def run( self, correct_tree, incorrect_tree, variable_dict ) :
    # Initialise TMVA tools
    ROOT.TMVA.Tools.Instance()
    f_output = ROOT.TFile( "{}/TMVA_output.root".format( self.output_directory ), "RECREATE" )
    factory = ROOT.TMVA.Factory( "TMVAClassification", f_output, "AnalysisType=Classification" )

    for variable, v_type in variable_dict.items() :
      factory.AddVariable( variable, v_type )
    factory.SetWeightExpression( "event_weight" );

    factory.AddSignalTree( correct_tree )
    factory.AddBackgroundTree( incorrect_tree )
    factory.PrepareTrainingAndTestTree( ROOT.TCut(""), ROOT.TCut(""), "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random" )

    # Define methods
    BDT_method = factory.BookMethod( ROOT.TMVA.Types.kBDT, "BDT", ":".join(
      [ "NTrees=850", "MinNodeSize=5", "MaxDepth=3", "BoostType=AdaBoost", "AdaBoostBeta=0.5", "SeparationType=GiniIndex" ]
    ) )

    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()

    shutil.move( "weights", self.output_directory )
