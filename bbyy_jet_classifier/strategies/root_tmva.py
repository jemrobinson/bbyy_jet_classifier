from . import BaseStrategy
from ..adaptors import root2python
import array
import os
import shutil
import ROOT
import logging
from root_numpy import array2tree, root2rec, tree2array
from root_numpy.tmva import add_classification_events, evaluate_reader
from numpy.lib import recfunctions

class RootTMVA(BaseStrategy) :
  default_output_location = "output/RootTMVA"


  def train( self, X_train, y_train, w_train ):
    f_output = ROOT.TFile( "{}/TMVA_output.root".format( self.output_directory ), "RECREATE" )
    factory = ROOT.TMVA.Factory( "TMVAClassification", f_output, "AnalysisType=Classification" )

    # -- Add variables to the factory:
    for v_name, v_type in self.variable_dict.items() :
        if v_name != "event_weight" : factory.AddVariable( v_name, v_type )
    
    # Call root_numpy's utility functions to add events from the arrays
    add_classification_events(factory, X_train, y_train, weights=w_train)
    add_classification_events(factory, X_train[0:20], y_train[0:20], weights=w_train[0:20], test=True) # need to put in something or TMVA will complain
    
    # The following line is necessary if events have been added individually:
    factory.PrepareTrainingAndTestTree(ROOT.TCut('1'), 'NormMode=EqualNumEvents')

    #-- Define methods:
    BDT_method = factory.BookMethod( ROOT.TMVA.Types.kBDT, "BDT", ":".join(
      [ "NTrees=800", "MinNodeSize=5", "MaxDepth=10", "BoostType=Grad", "SeparationType=GiniIndex" ]
    ))

    # -- Where stuff actually happens:
    factory.TrainAllMethods()

      # -- Organize output:
    if os.path.isdir( "{}/weights".format(self.output_directory) ) :
        shutil.rmtree( "{}/weights".format(self.output_directory) )
    shutil.move( "weights", self.output_directory )


  
  def test( self, X, y, w, process) :

    logging.getLogger("TMVA_BDT").info("Evaluating Performance...")

    # -- Setup dictionary of variable names to ROOT-accessible arrays
    variables = {}

    # -- Construct reader and add variables to it:
    reader = ROOT.TMVA.Reader()
    for v_name in self.classification_variables :
      #variables[v_name] = array.array( root2python.CHAR_2_ARRAYTYPE[self.variable_dict[v_name]] ,[0])
      #reader.AddVariable( v_name, variables[v_name] )
      #^^ this gives error: --- <FATAL> Reader                   : Reader::AddVariable( const TString& expression, Int_t* datalink ), this function is deprecated, please provide all variables to the reader as floats
      reader.AddVariable(v_name, array.array('f', [0] ) )

    # -- Load TMVA results
    reader.BookMVA("BDT", "{}/weights/TMVAClassification_BDT.weights.xml".format(self.output_directory))

    yhat = evaluate_reader(reader, 'BDT', X)

    # # -- Log classification scores
    # logging.getLogger("TMVA_BDT").info( "{} accuracy = {:.2f}%".format(process, 100 * classifier.score( X, y, sample_weight=w)) )
    # logging.getLogger("TMVA_BDT").info( classification_report( y, classifier.predict(X), target_names=["correct","incorrect"], sample_weight=w) )
    

    return yhat

   