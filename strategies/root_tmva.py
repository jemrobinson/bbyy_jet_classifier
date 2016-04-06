#from . import BaseStrategy
import os
import shutil
import ROOT

# def load_data(input_filename, signal_treename, bkg_treename):
#     '''
#     Definition:
#     ------------
#     Args:
#     -----
#     '''
#     print 'Loading trees {} and {} from file {}'.format(signal_treename, bkg_treename, input_filename)
#     f_input = ROOT.TFile( input_filename, "READ" )
#     correct_tree = f_input.Get( signal_treename )
#     incorrect_tree = f_input.Get( bkg_treename )

#     return correct_tree, incorrect_tree

#class RootTMVA(BaseStrategy) :
class RootTMVA(object):

    def __init__( self, output_directory ) :

        self.output_directory = output_directory
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)


    def run( self, input_filename, signal_treename, bkg_treename, variable_dict ) :
        '''
        Definition:
        ------------
        Args:
        -----
        '''
        # -- Import data by loading in .root file
        # correct_tree, incorrect_tree = load_data(input_filename, signal_treename, bkg_treename)
        print 'Loading trees {} and {} from file {}'.format(signal_treename, bkg_treename, input_filename)
        f_input = ROOT.TFile( input_filename, "READ" )
        correct_tree = f_input.Get( signal_treename )
        incorrect_tree = f_input.Get( bkg_treename )

        # -- Initialise TMVA tools
        ROOT.TMVA.Tools.Instance()
        f_output = ROOT.TFile( "{}/TMVA_output.root".format( self.output_directory ), "RECREATE" )
        factory = ROOT.TMVA.Factory( "TMVAClassification", f_output, "AnalysisType=Classification" )

        # -- Add variables to the factory:
        for variable, v_type in variable_dict.items() :
            factory.AddVariable( variable, v_type )

        factory.SetWeightExpression( "event_weight" );

        # -- Pass signal and background trees:
        factory.AddSignalTree( correct_tree )
        factory.AddBackgroundTree( incorrect_tree )
        factory.PrepareTrainingAndTestTree( ROOT.TCut(""), ROOT.TCut(""), "nTrain_Signal=0:nTrain_Background=0:SplitMode=Random" )

        # -- Define methods:
        BDT_method = factory.BookMethod( ROOT.TMVA.Types.kBDT, "BDT", ":".join(
        [ "NTrees=850", "MinNodeSize=5", "MaxDepth=15", "BoostType=AdaBoost", "AdaBoostBeta=0.5", "SeparationType=GiniIndex" ]
        ) )

        # -- Where stuff actually happens:
        factory.TrainAllMethods()
        factory.TestAllMethods()
        factory.EvaluateAllMethods()

        # -- Organize output:
        shutil.move( "weights", self.output_directory )
